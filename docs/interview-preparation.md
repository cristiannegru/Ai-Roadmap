# 🎯 AI Engineer Interview Preparation Guide

Prepare for your technical AI, Machine Learning, and Generative AI interviews. This guide compiles the most common interview questions across six technical domains with complete, high-quality, conceptual answers.

---

## 🗺️ Interview Domains
```
[Machine Learning] ──> [Deep Learning] ──> [LLMs & GenAI] ──> [RAG & Search] ──> [AI System Design]
```

- 📺 **Machine Learning Interview Questions (CampusX)**: [Watch Video](https://www.youtube.com/watch?v=4jRBRDbJemM)
- 📺 **AI System Design Interview Prep**: [Watch Video](https://www.youtube.com/watch?v=kBXYFaZ0EN0)

---

## 📌 Section 1: Machine Learning Questions

### Q1: Explain the Bias-Variance Tradeoff.
**Answer**: 
- **Bias** is the error introduced by approximating a real-world problem with a simplified model (underfitting). For example, fitting a linear regression to quadratic data.
- **Variance** is the model's sensitivity to small fluctuations in the training set (overfitting). High variance models fit training noise and perform poorly on unseen data.
- **Tradeoff**: As model complexity increases, bias decreases, but variance increases. The goal is to find the optimal complexity where total error is minimized.

### Q2: What is the difference between L1 (Lasso) and L2 (Ridge) Regularization?
**Answer**:
- Both add a penalty term to the loss function to prevent overfitting.
- **L1 (Lasso)** adds the absolute value of coefficients: $\text{Loss} + \lambda \sum |w_i|$. It drives less important feature weights to exactly zero, performing feature selection.
- **L2 (Ridge)** adds the squared value of coefficients: $\text{Loss} + \lambda \sum w_i^2$. It shrinks weights close to zero but never exactly zero, keeping all features.

---

## 📌 Section 2: Deep Learning Questions

### Q3: Why do we use Activation Functions? Can we train a network without them?
**Answer**:
- Activation functions introduce **non-linearity** into the network.
- Without non-linear activation functions, a multi-layer neural network is mathematically equivalent to a single-layer linear model, no matter how many layers you add. It would fail to learn complex patterns (like curves or circles).

### Q4: Explain the Vanishing Gradient Problem and how to mitigate it.
**Answer**:
- During backpropagation, gradients are multiplied chain-rule style from output to input. For deep networks using functions like Sigmoid or Tanh (where derivatives are $< 0.25$), gradients shrink exponentially as they propagate backward, causing early layers to update very slowly or stop learning completely.
- **Mitigation**:
  1. Use activation functions like **ReLU** (derivative is 1 for positive values).
  2. Implement **Residual Connections** (ResNet), allowing gradients to bypass layers.
  3. Apply **Batch Normalization**.

---

## 📌 Section 3: Large Language Models (LLM) Questions

### Q5: How does the Self-Attention mechanism work in Transformers?
**Answer**:
- Self-attention calculates how much focus each word in a sequence should place on other words in the same sequence.
- For each token, the model computes three vectors: **Query ($Q$)**, **Key ($K$)**, and **Value ($V$)**.
- Attention scores are calculated as: 
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
  where $d_k$ is the dimensionality of the key vectors.

### Q6: What is LLM Quantization?
**Answer**:
- Quantization is the process of mapping continuous high-precision floating-point numbers (like FP16 or FP32) to lower-precision integers (like INT8 or INT4).
- This significantly reduces model size, saving GPU VRAM and speeding up inference, allowing models to run on consumer hardware (e.g., via Ollama).

---

## 📌 Section 4: Retrieval-Augmented Generation (RAG) Questions

### Q7: What is the difference between Sparse Search and Dense Search?
**Answer**:
- **Sparse Search** (e.g., BM25, TF-IDF) matches exact keyword frequencies. It is fast and works well for specific terms, IDs, or product codes.
- **Dense Search** (e.g., Vector embeddings) represents the semantic meaning of text in a multi-dimensional space. It can find matches with different wording but similar meanings.
- **Best Practice**: Use **Hybrid Search** combining both, followed by a **Reranker** model to re-score the top matches.

---

## 📌 Section 5: AI System Design Questions

### Q8: Design a system for real-time document search using RAG for 10 million users.
**Answer**:
- **Ingestion Pipeline**: Use Kafka/RabbitMQ to queue document uploads. Chunk document content, create embeddings asynchronously using containerized workers, and store them in a distributed vector index (e.g., Pinecone or Qdrant cluster).
- **Caching Layer**: Cache common queries in Redis to bypass vector search and LLM calls.
- **Load Balancing**: Serve the application behind Nginx. Scale FastAPI container endpoints on Kubernetes (EKS). Use a hosted LLM gateway to handle rate-limiting and model switching.

---

## 📌 Section 6: Coding Questions

### Question: Implement Cosine Similarity in Python using NumPy.
**Answer**:
```python
import numpy as np

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
        
    return dot_product / (norm_v1 * norm_v2)

# Test
vec_a = np.array([1, 2, 3])
vec_b = np.array([2, 3, 4])
similarity = cosine_similarity(vec_a, vec_b)
print(f"Cosine Similarity: {similarity:.4f}")
# Output: Cosine Similarity: 0.9926
```
