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

---

Built on Deep-ML.
