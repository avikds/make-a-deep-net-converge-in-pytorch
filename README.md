# Make a Deep Net Converge in PyTorch

Chapter 11 of Hands-On Machine Learning as a debugging session: start from a deep sigmoid network that barely learns, measure its vanishing gradients layer by layer, then fix it one change at a time - He initialization, ReLU, batch normalization, dropout, a better optimizer, a one-cycle learning-rate schedule and gradient clipping - measuring each against the last on real Fashion-MNIST data. Finish with transfer learning: reuse the lower layers for a new task with a frozen backbone, then unfreeze and fine-tune, and save the result.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** fashion_loaders
- [x] **2.** DeepNet
- [x] **3.** train_epochs
- [x] **4.** gradient_norms
- [x] **5.** apply_he_init
- [x] **6.** compare_configs
- [x] **7.** dropout_effect
- [x] **8.** make_optimizer
- [x] **9.** compare_optimizers
- [x] **10.** one_cycle
- [x] **11.** clipping_effect
- [x] **12.** split_by_class
- [x] **13.** transfer_head
- [x] **14.** transfer_experiment
- [x] **15.** save_deepnet

## Results

```
gradient norm, first layer / last layer: sigmoid+default 2.4e-07   relu+He+BN 15.69
  10x sigmoid, default init        val acc 0.096
  10x relu, He init                val acc 0.481
  10x relu, He init, batchnorm     val acc 0.712
dropout 0.5 on a 3-layer net: train acc 0.821 -> 0.639, val acc 0.754 -> 0.743
optimizers at lr 0.01, 2 epochs: sgd 0.728  momentum 0.792  nesterov 0.792  adam 0.788  adamw 0.802
one-cycle (lr 0.0040 -> 0.10 -> 0.000021): val acc 0.812
gradient clipping at 1.0: total norm 0.42 -> 0.42

sandal vs sneaker: from scratch 0.927   frozen transfer 0.854   fine-tuned 0.854
saved and reloaded: 0.906 accuracy on the first validation batch, eval mode = True
```
