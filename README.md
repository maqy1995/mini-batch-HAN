# mini-batch training HAN with DGL
This is an attempt to implement mini-batch training with HAN, which used `RandomWalkSample` to get sub-graphs.
Original code from [dgl HAN](github.com/dmlc/dgl/tree/master/examples/pytorch/han)

Dependency:
> dgl 0.6.1  
  ogb 1.3.0
----------------

# mini-batch training on ACMRaw(DGL dataset):
```bash
python train_sample.py --dataset ACMRaw --num_neighbors 20 --batch_size 32 --hidden_units 8
```

It can match the full graph training result: [dgl HAN](github.com/dmlc/dgl/tree/master/examples/pytorch/han)
>EarlyStopping counter: 10 out of 10  
Epoch 23 | Val loss 0.5857 | Val Accuracy 0.9177 | Val Micro f1 0.9177 | Val Macro f1 0.9176  
Test loss 0.3041 | Test Accuracy 0.9176 | Test Micro f1 0.9176 | Test Macro f1 0.9174

----------------
# mini-batch training on ognb-mag:
```bash
python train_sample.py --dataset ognb-mag --num_neighbors 100 --batch_size 1024 --hidden_units 512
```
the result is bad (T T):
>Epoch 43 | loss: 3.4052 | train_acc: 0.2359 | train_micro_f1: 0.2359 | train_macro_f1: 0.0625  
EarlyStopping counter: 10 out of 10  
Val loss 3.4083 | Val Accuracy 0.2482 | Val Micro f1 0.2482 | Val Macro f1 0.0388  
Test loss 3.2047 | Test Accuracy 0.2691 | Test Micro f1 0.2691 | Test Macro f1 0.0383  

----------------
# full training on `PAP` based meta-graph(need more than 110G memory):
```bash
python train_PAPMAG_full.py
```
result not good:
>Epoch 522 | loss: 3.3740 | train_acc: 0.2356 | train_micro_f1: 0.2356 | train_macro_f1: 0.0411  
EarlyStopping counter: 100 out of 100  
Val loss 3.1961 | Val Accuracy 0.2638 | Val Micro f1 0.2638 | Val Macro f1 0.0382  
Test loss 3.1105 | Test Accuracy 0.2914 | Test Micro f1 0.2914 | Test Macro f1 0.0425  
