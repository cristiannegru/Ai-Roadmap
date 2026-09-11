# Deep Learning

Neural networks stack linear layers with nonlinear activations. ReLU keeps
gradients alive while sigmoid saturates in deep stacks. Backpropagation
applies the chain rule from loss to early layers, and the Adam optimizer
adapts each learning rate.

Convolutional neural networks scan images with learned filters, ReLU, and
max pooling. A tiny CNN with adaptive pooling classifies quadrant digits on
CPU in seconds. Train with cross entropy loss, track validation accuracy,
and save state dict checkpoints for deployment.

Graphics cards accelerate matrix multiplies; Apple Silicon uses MPS.
