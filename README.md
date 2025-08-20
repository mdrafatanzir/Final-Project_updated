# ANN CUDA Training on Dione

This project implements a simple Artificial Neural Network (ANN) in CUDA and CPU for regression on **NIR_Data**.

---

## 🚀 Steps to Run on Dione

1. **Login to Dione**
```bash
ssh mrafath@dione.utu.fi
```

2. **Load CUDA module**
```bash
module list
module load cuda
which nvcc
```

3. **Compile the code**
```bash
nvcc -O3 -arch=sm_70 -std=c++11 -o ann_cuda ann_cuda.cu
```

4. **Submit job with SLURM**
```bash
srun -p gpu -n 1 -t 10:00 --mem=2G -e err.txt -o out.txt \
    ./ann_cuda --data NIR_Data.dat --epochs 10 --batch 128 --lr 0.01
```

---

## 📂 Output Files

### `out.txt`
Example run output:
```text
Config: H=64 batch=128 epochs=10 lr=0.01
[CPU-serial] Epoch 1
[CPU-serial] Epoch 2
[CPU-serial] Epoch 3
[CPU-serial] Epoch 4
[CPU-serial] Epoch 5
[CPU-serial] Epoch 6
[CPU-serial] Epoch 7
[CPU-serial] Epoch 8
[CPU-serial] Epoch 9
[CPU-serial] Epoch 10
CPU time: 1.02863 s
[VAL] RMSE=0.322454 R2=0.901106
CUDA time: 0.0050917 s
[VAL] RMSE=0.322454 R2=0.901106
Saved metrics_log.csv
```

### `metrics_log.csv`
Validation metrics logged per epoch (or final run):
```csv
note,rmse,r2
final,0.322454,0.901106
```

---

## 📝 Notes
- Default dataset is `NIR_Data.csv` (last column is the target).  
- CPU baseline and CUDA version both run; performance is compared.  
- `metrics_log.csv` can be used for plotting RMSE and R² vs epochs.  
