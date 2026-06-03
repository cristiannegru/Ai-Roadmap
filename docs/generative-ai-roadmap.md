# 🚀 Generative AI & LLM Roadmap

Generative AI and Large Language Models (LLMs) have transformed how developers build intelligent software. This roadmap covers Phase 5C: prompting strategies, API integration, open-source models, orchestration frameworks, and fine-tuning.

---

## 🗺️ Generative AI Journey
```
[Prompt Engineering] ──> [LLM APIs & Local Models] ──> [LangChain & LlamaIndex] ──> [Fine-Tuning]
          │                           │                            │                    │
          ├─ Few-Shot & CoT           ├─ OpenAI & Claude APIs      ├─ Chains & Agents   ├─ LoRA / QLoRA
          └─ ReAct Framework          └─ Ollama Local Setup        └─ Document Loaders  └─ PEFT / DPO
```

---

## 📌 Phase 5C: Generative AI Specialization

### 1. Prompt Engineering
Mastering the art of writing effective instructions to guide LLM outputs.
- **Topics**: System Prompts, Zero-Shot/Few-Shot Prompting, Chain of Thought (CoT), ReAct Framework, Self-Consistency, Tree of Thoughts.
- 📺 **Prompt Engineering Course for Developers (Andrew Ng)**: [Watch Course Intro](https://youtu.be/mEsleV16qdo)
- 📺 **Prompt Engineering Masterclass**: [Watch Video](https://youtu.be/g3Hw_v2J0pE)

### 2. Large Language Models (LLMs) & Local Setup
Understanding model architectures and how to run LLMs locally or consume them via APIs.
- **Topics**: Quantization, Token limits, Context window, Temperature, Top-p/Top-k sampling, Ollama, HuggingFace Transformers.
- 📺 **Intro to LLMs by Andrej Karpathy**: [Watch Video](https://youtu.be/zjkBMFhNj_g)
- 📺 **Ollama Crash Course (Local LLMs)**: [Watch Video](https://youtu.be/Wjrdr0NU4Sk)

### 3. Orchestration Frameworks: LangChain & LlamaIndex
Libraries that help you connect LLMs to external data sources, memory, and tools.
- **Topics**: Chains, Prompts, Memory, Agents, Output Parsers, Document Loaders, Query Pipelines.
- 📺 **LangChain Crash Course**: [Watch Video](https://youtu.be/lG7Uxts9SXs)
- 📺 **LlamaIndex Tutorial for Beginners**: [Watch Video](https://youtu.be/5753-S2aUQA)

### 4. Fine-Tuning LLMs
Teaching open-source models new tasks or formatting preferences.
- **Topics**: Supervised Fine-Tuning (SFT), Parameter-Efficient Fine-Tuning (PEFT), LoRA (Low-Rank Adaptation), QLoRA, RLHF, DPO (Direct Preference Optimization).
- 📺 **Fine-Tuning LLMs Explained (LoRA/QLoRA)**: [Watch Video](https://youtu.be/eC6Hd1hF568)
- 📺 **Fine-Tuning LLM on Custom Dataset (PyTorch & Unsloth)**: [Watch Video](https://youtu.be/y7JleZ790yI)

---

## 💻 Code Sample: Local LLM Inference with Ollama & LangChain
```python
from langchain_community.llms import Ollama

# Connect to a local Llama 3 instance
llm = Ollama(model="llama3")

# Run simple text generation
prompt = "Explain the difference between supervised learning and deep learning in one paragraph."
response = llm.invoke(prompt)

print(response)
```

---

## 🛠️ Generative AI Milestones & Projects
1. **AI Resume Analyzer (OpenAI API + Streamlit)**
   - 📺 **Project Tutorial**: [Watch Video](https://youtu.be/1mE1U1-9y1E)
2. **Local Customer Support Chatbot (Ollama + LangChain + Streamlit)**
   - 📺 **Project Tutorial**: [Watch Video](https://youtu.be/d0o8959t6t8)
3. **Custom Fine-Tuning of Llama 3 (Unsloth + HuggingFace)**
   - 📺 **Project Tutorial**: [Watch Video](https://youtu.be/y7JleZ790yI)
