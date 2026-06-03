# 🧠 Deep Learning Roadmap

Deep Learning powers modern AI systems, from image recognition to large language models. This roadmap covers Phase 4: building neural networks, training architectures, and mastering deep learning frameworks (PyTorch and TensorFlow).

---

## 🗺️ Deep Learning Journey
```
[Artificial Neural Nets] ──> [Architectures: CNN & RNN] ──> [Transformers & Attention]
           │                              │                              │
           ├─ Activation & Loss           ├─ Image Conv & Pooling        ├─ Self-Attention
           └─ Backpropagation & Adam      └─ Sequence Modeling (LSTMs)   └─ GPT/BERT Foundations
```

---

## 📌 Phase 4: Deep Learning Foundations

### 1. Neural Networks & Math
Learn the fundamentals of Perceptrons, Multi-Layer Perceptrons (MLPs), Forward Pass, Loss Functions, Backpropagation, and Optimizers (SGD, RMSprop, Adam).
- 📺 **Neural Networks by 3Blue1Brown**: [Watch Playlist](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- 📺 **Neural Networks: Zero to Hero by Andrej Karpathy**: [Watch Playlist](https://youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUbFy1A)
- 📺 **CampusX Deep Learning Course**: [Watch Playlist](https://youtube.com/playlist?list=PLKnIA16_RmvVqyU6FRG_yL7n5wN1WdYk6)

### 2. Deep Learning Frameworks
Learn how to build neural networks using PyTorch (highly recommended for research/industry) or TensorFlow (used widely in enterprise).
- 📺 **PyTorch for Deep Learning Bootcamp (freeCodeCamp)**: [Watch Video](https://youtu.be/V_xro1bcAuA)
- 📺 **TensorFlow Developer Course (freeCodeCamp)**: [Watch Video](https://youtu.be/tpCFfeUEGs8)

---

## 📌 Deep Learning Architectures

### 1. Convolutional Neural Networks (CNNs)
Best for grid-like data like image pixels, video frames, and spectrographs.
- **Topics**: Convolution layer, Max Pooling, Padding, Strides, ResNet, VGG.
- 📺 **CNNs Explained (StatQuest)**: [Watch Video](https://youtu.be/HGwqe6yp1Ec)
- 📺 **Build a CNN with PyTorch**: [Watch Video](https://youtu.be/Jy4wM1xNeq0)

### 2. Recurrent Neural Networks & LSTMs
Designed for sequential data like text, time-series, and voice transcripts.
- **Topics**: Vanishing Gradient problem, LSTMs, GRUs, Seq2Seq models.
- 📺 **RNNs and LSTMs Explained (StatQuest)**: [Watch Video](https://youtu.be/AsNTP8Kwu80)
- 📺 **LSTMs from Scratch**: [Watch Video](https://youtu.be/YCzL96nL7j0)

### 3. Transformers (The Backbone of Modern GenAI)
Replacing LSTMs in NLP using Attention Mechanisms to process sequences in parallel.
- **Topics**: Self-Attention, Multi-Head Attention, Positional Encoding, Encoder-Decoder architecture.
- 📺 **Transformers Explained (3Blue1Brown)**: [Watch Video](https://youtu.be/zxQyUd1gUXI)
- 📺 **Let's Build GPT from Scratch by Andrej Karpathy**: [Watch Video](https://youtu.be/kCc8FmEb1nY)

---

## 💻 Code Sample: Simple PyTorch Neural Network
Here is how to create a basic Multi-Layer Perceptron (MLP) in PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define Model Architecture
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

model = MLP()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(model)
```

---

## 🛠️ Deep Learning Milestones & Projects
1. **MNIST Digit Classifier** (Your "Hello World" of Deep Learning)
   - 📺 **Tutorial**: [Watch Video](https://youtu.be/OMDnWYtWNC4)
2. **Image Classification using Transfer Learning (ResNet50)**
   - 📺 **Tutorial**: [Watch Video](https://youtu.be/r5S50gMJDvU)
3. **Shakespeare Text Generator (RNN/LSTM)**
   - 📺 **Tutorial**: [Watch Video](https://youtu.be/kCc8FmEb1nY)
