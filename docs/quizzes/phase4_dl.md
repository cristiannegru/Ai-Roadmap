# Quiz — Phase 4: Deep Learning

Self-test for `docs/deep-learning-roadmap.md` and notebook `04`.
Code: `src/ai_roadmap/torch_utils.py`.

## Q1 — Why do CNNs beat MLPs on images with far fewer parameters?
<details><summary>Answer</summary>

Weight sharing + locality: a convolution reuses the same filter across positions (translation
invariance), while an MLP learns separate weights per pixel. Our CNN (~25k params) beats the
MLP baseline on quadrant digits.
</details>

## Q2 — What does `AdaptiveAvgPool2d((4, 4))` buy the `SimpleCNN`?
<details><summary>Answer</summary>

Size agnosticism: whatever the input resolution (16×16 or 28×28 verified), features collapse to
a fixed 4×4 grid, so the classifier head always sees the same dimension.
</details>

## Q3 — Training loop in one sentence per step (see `train_classifier`).
<details><summary>Answer</summary>

Move batch to device → zero grads → forward pass → CrossEntropy loss → backprop → Adam step →
accumulate loss/accuracy. Repeat per epoch; evaluate on val after each.
</details>

## Q4 — Why CrossEntropyLoss + Adam for classification (not MSE + SGD)?
<details><summary>Answer</summary>

Cross-entropy on logits gives strong gradients for confident-wrong predictions (MSE saturates).
Adam adapts per-parameter learning rates, converging faster and more robustly than fixed-step SGD.
</details>

## Q5 — What happens if you forget `model.eval()` / `torch.no_grad()` at evaluation?
<details><summary>Answer</summary>

Dropout/batchnorm stay in training mode (noisy, wrong stats) and gradients accumulate —
wasted memory and non-deterministic accuracy. `evaluate_accuracy` handles both.
</details>

## Q6 — `get_device("auto")` order and why?
<details><summary>Answer</summary>

CUDA → MPS → CPU: fastest available accelerator first, always a working fallback. Checkpoints
store `state_dict` only, and `load_model`/`evaluate_accuracy` move weights to the target device
(the Chunk-3 MPS-bug fix).
</details>

## Q7 — Why synthetic quadrant digits instead of MNIST?
<details><summary>Answer</summary>

Zero downloads, deterministic (`seed`), CPU-trainable to >80% in seconds. The quadrant + class-bar
pattern is linearly separable with spatial structure — ideal for demonstrating the CNN advantage fast.
</details>

## Q8 — Train loss falls but val accuracy stalls. Diagnosis + two fixes?
<details><summary>Answer</summary>

Overfitting. Fixes: more data/augmentation, dropout, weight decay, early stopping on val accuracy,
or a smaller model. The loss/accuracy curve plot (notebook 04 §4) makes it visible.
</details>

## Q9 — ReLU vs sigmoid in hidden layers (interview Q4)?
<details><summary>Answer</summary>

Sigmoid saturates (|x| large → derivative ≈ 0), causing vanishing gradients in deep stacks.
ReLU's derivative is 1 for positives, keeping gradients alive — plus residuals and batchnorm help.
</details>

## Q10 — What does `count_parameters` count, and why check it?
<details><summary>Answer</summary>

Trainable parameters only (`requires_grad`). It sizes the model (memory, speed, overfitting risk)
and verifies architecture edits actually changed capacity.
</details>
