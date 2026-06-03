# 💬 Natural Language Processing (NLP) Roadmap

Natural Language Processing (NLP) enables machine intelligence to analyze, interpret, and generate human language. This roadmap bridges traditional linguistics and text vectorization with state-of-the-art transformer pipelines.

---

## 🗺️ NLP Learning Flow
```
[Text Preprocessing] ──> [Vectorization & Embeddings] ──> [Transformers & BERT]
         │                            │                           │
         ├─ Tokenization              ├─ TF-IDF & Bag-of-Words    ├─ HuggingFace Transformers
         └─ Lemmatization & Stopwords └─ Word2Vec & GloVe         └─ Custom Sequence Tagging
```

---

## 📌 Phase 5B: NLP Specialization

### 1. Text Preprocessing & Cleaning
Learn how to prepare raw text for mathematical representation.
- **Topics**: Tokenization, Lowercasing, Stopword Removal, Stemming, Lemmatization, Regular Expressions (Regex).
- 📺 **NLP Course for Beginners (freeCodeCamp)**: [Watch Video](https://youtu.be/xvqWRacEE0o)
- 📺 **CampusX NLP Full Playlist**: [Watch Playlist](https://youtube.com/playlist?list=PLKnIA16_RmvZo7fp5kkIth60-mch1KbIX)

### 2. Traditional NLP & Embeddings
Representing words as numbers so machine learning models can process them.
- **Topics**: Bag of Words (BoW), TF-IDF (Term Frequency-Inverse Document Frequency), Word2Vec (Skip-gram/CBOW), GloVe embeddings.
- 📺 **TF-IDF Explained**: [Watch Video](https://youtu.be/z2V-T1_C-x0)
- 📺 **Word2Vec Explained (StatQuest)**: [Watch Video](https://youtu.be/viZrOn_ADWA)

### 3. Transformer NLP Models & HuggingFace
Transition from LSTMs to Attention-based architectures.
- **Topics**: Encoder models (BERT, RoBERTa), Decoder models (GPT, Llama), Sequence Classification, Named Entity Recognition (NER), HuggingFace Transformers library.
- 📺 **HuggingFace NLP Course**: [Watch Video](https://youtu.be/Xzp857aee2E)
- 📺 **BERT Model Explained**: [Watch Video](https://youtu.be/xI0HHN5XKDo)

---

## 💻 Code Sample: Text Classification with HuggingFace
```python
from transformers import pipeline

# Load pre-trained sentiment analysis pipeline
classifier = pipeline("sentiment-analysis")

# Perform inference
result = classifier("This AI Roadmap is incredibly detailed and well-structured!")
print(result)
# Output: [{'label': 'POSITIVE', 'score': 0.9998}]
```

---

## 🛠️ NLP Milestones & Projects
1. **SMS Spam Classifier (TF-IDF + Naive Bayes)**
   - 📺 **Project Tutorial**: [Watch Video](https://youtu.be/y4gKOdZ-e4E)
2. **Custom Named Entity Recognition (NER) for Resumes using SpaCy**
   - 📺 **Project Tutorial**: [Watch Video](https://youtu.be/WEn34Wf_210)
3. **Sentiment Analysis Dashboard (HuggingFace + Streamlit)**
   - 📺 **Project Tutorial**: [Watch Video](https://youtu.be/0Ska4-N1G98)
