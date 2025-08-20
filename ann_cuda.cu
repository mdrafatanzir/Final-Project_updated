#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <chrono>
#include <string>
#include <numeric>
#include <cctype>
#include <thread>
#include <mutex>
#include <atomic>


#ifdef _OPENMP
#include <omp.h>
#endif

#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
  cudaError_t err = (call); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); \
  } \
} while(0)

struct Options {
  int N = 0, D = 0, H = 64;
  int epochs = 5, batch = 128;
  float lr = 1e-2f;
  unsigned seed = 42u;
  std::string data = "NIR_Data.csv";
  bool target_last = true;
  float val_split = 0.2f;
  bool run_cpu = true;         // run CPU baseline
  bool cpu_omp = false;        // enable OpenMP in CPU baseline (if built with -fopenmp)
  // task-parallel sweep
  std::vector<int> sweep_H;    
  std::vector<float> sweep_lr; 
  int sweep_par = 0;           // number of parallel workers (0 -> use hw concurrency)
};

static std::vector<std::string> split(const std::string& s, char delim) {
  std::vector<std::string> out; out.reserve(16);
  size_t i=0, j=0;
  while(j<=s.size()){
    if(j==s.size() || s[j]==delim){ if(j>i) out.emplace_back(s.substr(i,j-i)); i=j+1; }
    ++j;
  }
  return out;
}
static void parse_csv_list(const std::string& s, std::vector<int>& out){
  out.clear(); if(s.empty()) return;
  for(auto &t: split(s, ',')) out.push_back(std::stoi(t));
}
static void parse_csv_list_f(const std::string& s, std::vector<float>& out){
  out.clear(); if(s.empty()) return;
  for(auto &t: split(s, ',')) out.push_back(std::stof(t));
}

static Options parse(int argc, char** argv){
  Options o;
  for(int i=1;i<argc;i++){
    std::string a = argv[i];
    auto next = [&](){ if(i+1>=argc){ fprintf(stderr,"Missing value after %s\n", a.c_str()); exit(1);} return std::string(argv[++i]); };
    if(a=="--h") o.H = std::stoi(next());
    else if(a=="--epochs") o.epochs = std::stoi(next());
    else if(a=="--batch") o.batch = std::stoi(next());
    else if(a=="--lr") o.lr = std::stof(next());
    else if(a=="--data") o.data = next();
    else if(a=="--target-last") o.target_last = std::stoi(next())!=0;
    else if(a=="--val") o.val_split = std::stof(next());
    else if(a=="--no-cpu"){ o.run_cpu = false; }
    else if(a=="--omp"){ o.cpu_omp = true; }
    else if(a=="--sweep-h"){ parse_csv_list(next(), o.sweep_H); }
    else if(a=="--sweep-lr"){ parse_csv_list_f(next(), o.sweep_lr); }
    else if(a=="--sweep-par"){ o.sweep_par = std::stoi(next()); }
  }
  return o;
}


// CSV load


static bool read_csv_matrix(const std::string& path,std::vector<float>& M,int& rows,int& cols){
  FILE* f = fopen(path.c_str(), "r");
  if(!f){ perror(("open "+path).c_str()); return false; }
  std::vector<float> data; data.reserve(1<<20);
  rows = 0; cols = -1;
  std::string line; line.reserve(1<<15);
  char buf[1<<15];
  bool header_checked = false;
  while(fgets(buf,sizeof(buf),f)){
    line.assign(buf);
    if(line.empty()) continue;
    if(!header_checked){
      header_checked = true;
      unsigned char c = static_cast<unsigned char>(line[0]);
      if(!std::isdigit(c) && line[0]!='-' && line[0]!='+') continue; // skip header
    }
    int ccount=0;
    const char* s=line.c_str();
    char* end=nullptr;
    while(*s){
      while(*s==' '||*s=='\t'||*s==','||*s=='\n'||*s=='\r'){ if(*s=='\n'||*s=='\r') break; ++s; }
      if(*s=='\n'||*s=='\r'||*s=='\0') break;
      float v = strtof(s,&end);
      if(end==s){ ++s; continue; }
      data.push_back(v); ++ccount; s=end;
    }
    if(ccount==0) continue;
    if(cols==-1) cols=ccount;
    else if(ccount!=cols){ fprintf(stderr,"Irregular columns at row %d\n", rows); fclose(f); return false; }
    ++rows;
  }
  fclose(f);
  M.swap(data);
  return true;
}


// Standardization

static void standardize_features(std::vector<float>& X,int N,int D,std::vector<float>& mean,std::vector<float>& stdv){
  mean.assign(D,0.f); stdv.assign(D,0.f);
  for(int j=0;j<D;++j){ double s=0; for(int i=0;i<N;++i) s+=X[i*D+j]; mean[j]=static_cast<float>(s/N); }
  for(int j=0;j<D;++j){ double v=0; for(int i=0;i<N;++i){ double d=X[i*D+j]-mean[j]; v+=d*d; } stdv[j]=static_cast<float>(std::sqrt(v/(N>1?N-1:N))); if(stdv[j]==0) stdv[j]=1; }
  for(int i=0;i<N;++i) for(int j=0;j<D;++j) X[i*D+j]=(X[i*D+j]-mean[j])/stdv[j];
}
static void standardize_target(std::vector<float>& y,float& mean,float& stdv){
  double s=0; for(float v:y) s+=v; mean=static_cast<float>(s/y.size());
  double vv=0; for(float v:y){ double d=v-mean; vv+=d*d; }
  stdv=static_cast<float>(std::sqrt(vv/(y.size()>1?y.size()-1:y.size()))); if(stdv==0) stdv=1;
  for(float& v:y) v=(v-mean)/stdv;
}

// Metrics

static void rmse_r2(const std::vector<float>& y,const std::vector<float>& yhat,double& rmse,double& r2){
  assert(y.size()==yhat.size());
  int n=int(y.size());
  double se=0.0, sy=0.0, mean=0.0; for(float v:y) mean+=v; mean/=n;
  for(int i=0;i<n;++i){ double d=yhat[i]-y[i]; se+=d*d; double dy=y[i]-mean; sy+=dy*dy; }
  rmse=std::sqrt(se/n); r2=sy>0?1.0-se/sy:0.0;
}


// CPU baseline 

static void train_cpu_serial(const std::vector<float>& X,const std::vector<float>& y,
                             int N,int D,int H,int epochs,int batch,float lr){
  std::mt19937 rng(123); std::normal_distribution<float> nd(0.f,0.1f);
  std::vector<float> W1(D*H), b1(H,0.f), W2(H), b2(1,0.f);
  for(auto&w:W1) w=nd(rng); for(auto&w:W2) w=nd(rng);

  std::vector<int> idx(N); std::iota(idx.begin(),idx.end(),0);
  for(int ep=1; ep<=epochs; ++ep){
    std::shuffle(idx.begin(),idx.end(),rng); double loss_ep=0.0;
    for(int s=0; s<N; s+=batch){
      int B=std::min(batch,N-s);
      std::vector<float>dW1(D*H,0.f),db1(H,0.f),dW2(H,0.f); float db2=0.f;
      for(int i=0;i<B;++i){
        int id=idx[s+i]; std::vector<float> h(H);
        for(int j=0;j<H;++j){ float sum=b1[j]; for(int r=0;r<D;++r) sum+=X[id*D+r]*W1[r*H+j]; h[j]=std::max(0.f,sum); }
        float yhat=b2[0]; for(int j=0;j<H;++j) yhat+=h[j]*W2[j];
        float diff=yhat-y[id]; loss_ep+=0.5*diff*diff;
        for(int j=0;j<H;++j) dW2[j]+=h[j]*diff; db2+=diff;
        for(int j=0;j<H;++j){ float dz=(h[j]>0?1.f:0.f)*W2[j]*diff; db1[j]+=dz; for(int r=0;r<D;++r) dW1[r*H+j]+=X[id*D+r]*dz; }
      }
      float invB=1.f/B;
      for(int j=0;j<H;++j){ W2[j]-=lr*dW2[j]*invB; b1[j]-=lr*db1[j]*invB; for(int r=0;r<D;++r) W1[r*H+j]-=lr*dW1[r*H+j]*invB; }
      b2[0]-=lr*db2*invB;
    }
    std::cout<<"[CPU-serial] Epoch "<<ep<<"\n";
  }
}


// CPU baseline with OpenMP 

static void train_cpu_omp(const std::vector<float>& X,const std::vector<float>& y,
                          int N,int D,int H,int epochs,int batch,float lr){
  std::mt19937 rng(123); std::normal_distribution<float> nd(0.f,0.1f);
  std::vector<float> W1(D*H), b1(H,0.f), W2(H), b2(1,0.f);
  for(auto&w:W1) w=nd(rng); for(auto&w:W2) w=nd(rng);

  std::vector<int> idx(N); std::iota(idx.begin(),idx.end(),0);

  for(int ep=1; ep<=epochs; ++ep){
    std::shuffle(idx.begin(),idx.end(),rng);
    for(int s=0; s<N; s+=batch){
      int B=std::min(batch,N-s);

      std::vector<float>dW1(D*H,0.f),db1(H,0.f),dW2(H,0.f); float db2=0.f;

      #pragma omp parallel
      {
        std::vector<float> dW1_local(D*H,0.f), db1_local(H,0.f), dW2_local(H,0.f);
        float db2_local = 0.f;

        #pragma omp for schedule(static)
        for(int i=0;i<B;++i){
          int id=idx[s+i];
          std::vector<float> h(H);
          for(int j=0;j<H;++j){
            float sum=b1[j];
            for(int r=0;r<D;++r) sum+=X[id*D+r]*W1[r*H+j];
            h[j]=sum>0?sum:0.f;
          }
          float yhat=b2[0]; for(int j=0;j<H;++j) yhat+=h[j]*W2[j];
          float diff=yhat-y[id];

          for(int j=0;j<H;++j) dW2_local[j]+=h[j]*diff;
          db2_local+=diff;

          for(int j=0;j<H;++j){
            float dz=(h[j]>0?1.f:0.f)*W2[j]*diff;
            db1_local[j]+=dz;
            for(int r=0;r<D;++r) dW1_local[r*H+j]+=X[id*D+r]*dz;
          }
        }

        // reduction
        #pragma omp critical
        {
          for(int j=0;j<H;++j){
            dW2[j]+=dW2_local[j];
            db1[j]+=db1_local[j];
          }
          db2+=db2_local;
          for(int t=0;t<D*H;++t) dW1[t]+=dW1_local[t];
        }
      } // parallel

      float invB=1.f/B;
      for(int j=0;j<H;++j){
        W2[j]-=lr*dW2[j]*invB;
        b1[j]-=lr*db1[j]*invB;
        for(int r=0;r<D;++r) W1[r*H+j]-=lr*dW1[r*H+j]*invB;
      }
      b2[0]-=lr*db2*invB;
    }
    std::cout<<"[CPU-omp] Epoch "<<ep<<"\n";
  }
}


// CUDA kernels

__global__ void k_affine1_forward(float* Z1,const float* X,const float* W1,const float* b1,int B,int D,int H){
  int i = blockIdx.y*blockDim.y + threadIdx.y;
  int j = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<B && j<H){
    float sum = b1[j];
    for(int r=0;r<D;++r) sum += X[i*D+r]*W1[r*H+j];
    Z1[i*H+j] = sum;
  }
}
__global__ void k_relu_inplace(float* A,int n){int idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx<n) A[idx]=A[idx]>0?A[idx]:0;}
__global__ void k_out_forward(float* yhat,const float* H1,const float* W2,const float* b2,int B,int H){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<B){float s=b2[0];for(int j=0;j<H;++j)s+=H1[i*H+j]*W2[j];yhat[i]=s;}}
__global__ void k_diff(float* dy,const float* yhat,const float* y,int B){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<B) dy[i]=yhat[i]-y[i];}
__global__ void k_accum_dW2(float* dW2,const float* H1,const float* dy,int B,int H){int h=blockIdx.x*blockDim.x+threadIdx.x;if(h<H){float acc=0;for(int i=0;i<B;++i) acc+=H1[i*H+h]*dy[i];dW2[h]=acc;}}
__global__ void k_accum_db2(float* db2,const float* dy,int B){float acc=0;for(int i=0;i<B;++i)acc+=dy[i];db2[0]=acc;}
__global__ void k_backprop_hidden(float* dh,const float* H1,const float* W2,const float* dy,int B,int H){int i=blockIdx.y*blockDim.y+threadIdx.y;int h=blockIdx.x*blockDim.x+threadIdx.x;if(i<B&&h<H){float mask=H1[i*H+h]>0?1.f:0.f;dh[i*H+h]=mask*W2[h]*dy[i];}}
__global__ void k_accum_dW1(float* dW1,const float* X,const float* dh,int B,int D,int H){int h=blockIdx.x*blockDim.x+threadIdx.x;int r=blockIdx.y*blockDim.y+threadIdx.y;if(h<H&&r<D){float acc=0;for(int i=0;i<B;++i) acc+=X[i*D+r]*dh[i*H+h];dW1[r*H+h]=acc;}}
__global__ void k_accum_db1(float* db1,const float* dh,int B,int H){int h=blockIdx.x*blockDim.x+threadIdx.x;if(h<H){float acc=0;for(int i=0;i<B;++i) acc+=dh[i*H+h];db1[h]=acc;}}
__global__ void k_sgd_update(float* P,const float* G,float lr,float invB,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)P[i]-=lr*G[i]*invB;}
__global__ void k_sgd_update_scalar(float* b,const float* gb,float lr,float invB){if(threadIdx.x==0&&blockIdx.x==0) b[0]-=lr*gb[0]*invB;}


// Simple CPU forward with given params (for validation)


static void forward_cpu(const std::vector<float>& X,const std::vector<float>& W1,const std::vector<float>& b1,
                        const std::vector<float>& W2,const std::vector<float>& b2,int N,int D,int H,
                        std::vector<float>& yhat){
  yhat.assign(N,0.f);
  for(int i=0;i<N;++i){
    std::vector<float> h(H);
    for(int j=0;j<H;++j){
      float s=b1[j];
      for(int r=0;r<D;++r) s+=X[i*D+r]*W1[r*H+j];
      h[j]=s>0?s:0.f;
    }
    float o=b2[0]; for(int j=0;j<H;++j) o+=h[j]*W2[j];
    yhat[i]=o;
  }
}


// CUDA training (mini-batch SGD)

static void train_cuda(const std::vector<float>& X,const std::vector<float>& y,
                       int N,int D,int H,int epochs,int batch,float lr,
                       double &cuda_time,
                       std::vector<float>& W1_out,std::vector<float>& b1_out,
                       std::vector<float>& W2_out,std::vector<float>& b2_out,
                       const std::vector<float>* Xval=nullptr,const std::vector<float>* yval=nullptr,int Nval=0){
  // Device buffers
  float *dX,*dY,*dy,*yhat,*Z1,*H1,*dh;
  float *dW1,*db1,*dW2,*db2,*W1,*b1,*W2,*b2;
  CHECK_CUDA(cudaMalloc(&dX, (size_t)N*D*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dY, (size_t)batch*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dy, (size_t)batch*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&yhat, (size_t)batch*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&Z1, (size_t)batch*H*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&H1, (size_t)batch*H*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dh, (size_t)batch*H*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dW1, (size_t)D*H*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&db1, (size_t)H*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dW2, (size_t)H*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&db2, sizeof(float)));
  CHECK_CUDA(cudaMalloc(&W1, (size_t)D*H*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&b1, (size_t)H*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&W2, (size_t)H*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&b2, sizeof(float)));

  // init
  std::mt19937 rng(123); std::normal_distribution<float> nd(0.f,0.1f);
  std::vector<float> hW1(D*H), hb1(H,0.f), hW2(H), hb2(1,0.f);
  for(auto &v:hW1) v=nd(rng); for(auto &v:hW2) v=nd(rng);
  CHECK_CUDA(cudaMemcpy(W1,hW1.data(),(size_t)D*H*sizeof(float),cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(b1,hb1.data(),(size_t)H*sizeof(float),cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(W2,hW2.data(),(size_t)H*sizeof(float),cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(b2,hb2.data(),sizeof(float),cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dX,X.data(),(size_t)N*D*sizeof(float),cudaMemcpyHostToDevice));

  auto t0=std::chrono::high_resolution_clock::now();

  for(int ep=0; ep<epochs; ++ep){
    for(int s=0; s<N; s+=batch){
      int B=std::min(batch,N-s);
      CHECK_CUDA(cudaMemcpy(dY, y.data()+s, (size_t)B*sizeof(float), cudaMemcpyHostToDevice));

      dim3 blockH(16,16); dim3 gridH((H+15)/16,(B+15)/16);
      k_affine1_forward<<<gridH,blockH>>>(Z1,dX+s*D,W1,b1,B,D,H);
      k_relu_inplace<<<(B*H+255)/256,256>>>(Z1,B*H);
      CHECK_CUDA(cudaMemcpy(H1,Z1,(size_t)B*H*sizeof(float),cudaMemcpyDeviceToDevice));
      k_out_forward<<<(B+255)/256,256>>>(yhat,H1,W2,b2,B,H);
      k_diff<<<(B+255)/256,256>>>(dy,yhat,dY,B);
      k_accum_dW2<<<(H+255)/256,256>>>(dW2,H1,dy,B,H);
      k_accum_db2<<<1,1>>>(db2,dy,B);
      k_backprop_hidden<<<gridH,blockH>>>(dh,H1,W2,dy,B,H);
      k_accum_dW1<<<gridH,blockH>>>(dW1,dX+s*D,dh,B,D,H);
      k_accum_db1<<<(H+255)/256,256>>>(db1,dh,B,H);
      float invB=1.f/B;
      k_sgd_update<<<(D*H+255)/256,256>>>(W1,dW1,lr,invB,D*H);
      k_sgd_update<<<(H+255)/256,256>>>(b1,db1,lr,invB,H);
      k_sgd_update<<<(H+255)/256,256>>>(W2,dW2,lr,invB,H);
      k_sgd_update_scalar<<<1,1>>>(b2,db2,lr,invB);
    }
  }

  auto t1=std::chrono::high_resolution_clock::now();
  cuda_time=std::chrono::duration<double>(t1-t0).count();

  // copy back
  W1_out.resize((size_t)D*H); b1_out.resize(H); W2_out.resize(H); b2_out.resize(1);
  CHECK_CUDA(cudaMemcpy(W1_out.data(),W1,(size_t)D*H*sizeof(float),cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(b1_out.data(),b1,(size_t)H*sizeof(float),cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(W2_out.data(),W2,(size_t)H*sizeof(float),cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(b2_out.data(),b2,sizeof(float),cudaMemcpyDeviceToHost));

  // free
  cudaFree(dX); cudaFree(dY); cudaFree(dy); cudaFree(yhat);
  cudaFree(Z1); cudaFree(H1); cudaFree(dh);
  cudaFree(dW1); cudaFree(db1); cudaFree(dW2); cudaFree(db2);
  cudaFree(W1); cudaFree(b1); cudaFree(W2); cudaFree(b2);

  // optional val metrics printed once here (use host forward)
  if(Xval && yval && Nval>0){
    std::vector<float> yhat_val; double rmse=0,r2=0;
    forward_cpu(*Xval,W1_out,b1_out,W2_out,b2_out,Nval,D,H,yhat_val);
    rmse_r2(*yval,yhat_val,rmse,r2);
    std::cout<<"[VAL] RMSE="<<rmse<<" R2="<<r2<<"\n";
  }
}

// ------------------------------
// Task-parallel sweep (H x lr)

struct SweepJob { int H; float lr; };

static void run_sweep(const Options& base,
                      const std::vector<float>& Xtr,const std::vector<float>& ytr,
                      const std::vector<float>& Xval,const std::vector<float>& yval,
                      int Ntr,int Nval,int D){
  std::vector<SweepJob> jobs;
  for(int H : base.sweep_H)
    for(float lr : base.sweep_lr)
      jobs.push_back({H,lr});
  if(jobs.empty()){ std::cout<<"[SWEEP] no jobs\n"; return; }

  int max_par = base.sweep_par>0 ? base.sweep_par : (int)std::max(1u,std::thread::hardware_concurrency());
  std::mutex io_mtx;

  std::atomic<size_t> next(0);
  auto worker = [&](){
    while(true){
      size_t k = next.fetch_add(1);
      if(k>=jobs.size()) break;
      auto job = jobs[k];
      double t=0.0; std::vector<float> W1,b1,W2,b2;
      train_cuda(Xtr,ytr,Ntr,D,job.H,base.epochs,base.batch,job.lr,
                 t,W1,b1,W2,b2,&Xval,&yval,Nval);
      std::vector<float> yhat_val; double rmse=0,r2=0;
      forward_cpu(Xval,W1,b1,W2,b2,Nval,D,job.H,yhat_val);
      rmse_r2(yval,yhat_val,rmse,r2);
      std::lock_guard<std::mutex> lk(io_mtx);
      std::cout<<"[SWEEP] H="<<job.H<<" lr="<<job.lr<<" time="<<t<<"s RMSE="<<rmse<<" R2="<<r2<<"\n";
    }
  };

  std::vector<std::thread> pool;
  for(int i=0;i<max_par;i++) pool.emplace_back(worker);
  for(auto& th: pool) th.join();
}


// Main Function

int main(int argc,char**argv){
  Options opt = parse(argc,argv);
  std::cout<<"Config: H="<<opt.H
           <<" batch="<<opt.batch<<" epochs="<<opt.epochs<<" lr="<<opt.lr
#ifdef _OPENMP
           <<" omp="<<(opt.cpu_omp?1:0)
#else
           <<" omp=0"
#endif
           <<"\n";

  // Load data
  std::vector<float> M; int rows=0,cols=0;
  if(!read_csv_matrix(opt.data,M,rows,cols)){
    std::cerr<<"Failed to load dataset: "<<opt.data<<"\n";
    return 1;
  }
  if(cols<2){ std::cerr<<"Dataset needs >=2 columns (features + target)\n"; return 1; }

  opt.N = rows; opt.D = cols - 1;
  std::vector<float> X((size_t)opt.N*opt.D), y(opt.N);
  for(int i=0;i<opt.N;++i){
    for(int j=0;j<opt.D;++j) X[i*opt.D+j] = opt.target_last ? M[i*cols + j] : M[i*cols + (j+1)];
    y[i] = opt.target_last ? M[i*cols + (cols-1)] : M[i*cols + 0];
  }

  // Preprocess
  std::vector<float> xmean,xstd; float ymean=0,ystd=1;
  standardize_features(X,opt.N,opt.D,xmean,xstd);
  standardize_target(y,ymean,ystd);

  // Split
  int Nval = int(std::round(opt.N * opt.val_split));
  int Ntr  = opt.N - Nval;
  std::vector<int> perm(opt.N); std::iota(perm.begin(),perm.end(),0);
  std::mt19937 sp_rng(opt.seed^0xBADC0DEu); std::shuffle(perm.begin(),perm.end(),sp_rng);
  std::vector<float> Xtr((size_t)Ntr*opt.D), ytr(Ntr), Xval((size_t)Nval*opt.D), yval(Nval);
  for(int i=0;i<Ntr;++i){ int id=perm[i]; std::copy_n(&X[id*opt.D],opt.D,&Xtr[i*opt.D]); ytr[i]=y[id]; }
  for(int i=0;i<Nval;++i){ int id=perm[Ntr+i]; std::copy_n(&X[id*opt.D],opt.D,&Xval[i*opt.D]); yval[i]=y[id]; }

  // CPU baseline
  if(opt.run_cpu){
#ifdef _OPENMP
    auto t0 = std::chrono::high_resolution_clock::now();
    if(opt.cpu_omp) train_cpu_omp(Xtr,ytr,Ntr,opt.D,opt.H,opt.epochs,opt.batch,opt.lr);
    else            train_cpu_serial(Xtr,ytr,Ntr,opt.D,opt.H,opt.epochs,opt.batch,opt.lr);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout<<"CPU time: "<<std::chrono::duration<double>(t1-t0).count()<<" s\n";
#else
    auto t0 = std::chrono::high_resolution_clock::now();
    train_cpu_serial(Xtr,ytr,Ntr,opt.D,opt.H,opt.epochs,opt.batch,opt.lr);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout<<"CPU time: "<<std::chrono::duration<double>(t1-t0).count()<<" s\n";
#endif
  }

  // GPU
  double cuda_time=0.0; std::vector<float>W1f,b1f,W2f,b2f;
  train_cuda(Xtr,ytr,Ntr,opt.D,opt.H,opt.epochs,opt.batch,opt.lr,
             cuda_time,W1f,b1f,W2f,b2f,&Xval,&yval,Nval);
  std::cout<<"CUDA time: "<<cuda_time<<" s\n";

  // Final validation
  std::vector<float> yhat_val; double rmse=0,r2=0;
  forward_cpu(Xval,W1f,b1f,W2f,b2f,Nval,opt.D,opt.H,yhat_val);
  rmse_r2(yval,yhat_val,rmse,r2);
  std::cout<<"[VAL] RMSE="<<rmse<<" R2="<<r2<<"\n";

  // Optional task-parallel sweep
  if(!opt.sweep_H.empty() && !opt.sweep_lr.empty()){
    run_sweep(opt, Xtr,ytr, Xval,yval, Ntr,Nval, opt.D);
  }

  // Minimal epoch log (CSV)
  FILE* log=fopen("metrics_log.csv","w");
  if(log){
    fprintf(log,"note,rmse,r2\n");
    fprintf(log,"final,%.6f,%.6f\n",rmse,r2);
    fclose(log);
    std::cout<<"Saved metrics_log.csv\n";
  }

  return 0;
}
