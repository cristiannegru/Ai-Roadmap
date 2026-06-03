# 🗂️ Retrieval-Augmented Generation (RAG) Roadmap

Retrieval-Augmented Generation (RAG) is the industry standard architectural pattern to connect Large Language Models with private, external datasets. This roadmap guides you from a basic document-search bot to an advanced, production-grade RAG pipeline.

---

## 🗺️ RAG Pipeline Architecture
```
[User Query] ──> [Query Translation / Rewriting] ──> [Vector Database Search (Pinecone/Chroma)]
                       │                                         │
                       ▼                                         ▼
[LLM Synthesizer] <── [Re-Ranking (Cross-Encoders)] <── [Retrieved Text Chunks]
```

---

## 📌 Phase 5D: RAG Specialization

### 1. Naive RAG (The Basics)
Learn the core components of reading a file, chunking it, creating vectors, and querying a vector store.
- **Topics**: Chunking Strategies (Fixed size, Token-based, Semantic chunking), Embedding Models (OpenAI, HuggingFace, Cohere), Vector Indexing, Cosine Similarity.
- 📺 **RAG from Scratch by LangChain**: [Watch Playlist](https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x)
- 📺 **Vector Database Basics (Pinecone Tutorial)**: [Watch Video](https://www.youtube.com/watch?v=5fp6e5nhJRk)

### 2. Advanced RAG (Optimizing Retrieval)
Simple semantic search fails when queries are vague or documents are complex. Master these optimization techniques.
- **Query Translation**: Query Rewriting, Multi-Query expansion, Sub-Query decomposition.
- **Routing**: Query routing based on semantic intent or metadata.
- **Post-Retrieval (Reranking)**: Scoring retrieved documents with Cross-Encoder models to select the most relevant chunks.
- 📺 **Advanced RAG Concepts & Architecture**: [Watch Video](https://www.youtube.com/watch?v=A7EpJzaqtNc)
- 📺 **Advanced RAG Tutorial (Query Translation & Reranking)**: [Watch Video](https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x)

### 3. RAG Evaluation (RAGAS & TruLens)
How do you know if your RAG pipeline is working? You must test the quality of retrieval and generation.
- **Metrics**: Faithfulness, Answer Relevance, Context Recall, Context Precision.
- 📺 **Evaluating RAG with RAGAS**: [Watch Video](https://www.youtube.com/watch?v=5fp6e5nhJRk)

---

## 💻 Code Sample: Simple RAG Pipeline with LlamaIndex
Here is how to load a PDF and query it using LlamaIndex:

```python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

# Load documents from a local directory
documents = SimpleDirectoryReader("./data").load_data()

# Build the vector index (automatically chunks & creates embeddings)
index = VectorStoreIndex.from_documents(documents)

# Create the query engine
query_engine = index.as_query_engine()

# Run a query
response = query_engine.query("What are the key milestones in Phase 5 of the AI Roadmap?")
print(response)
```

---

## 🛠️ RAG Milestones & Projects
1. **PDF Question-Answering Assistant (ChromaDB + OpenAI + LangChain)**
   - 📺 **Project Tutorial**: [Watch Video](https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x)
2. **Enterprise RAG Dashboard with Hybrid Search & Cohere Rerank**
   - 📺 **Project Tutorial**: [Watch Video](https://www.youtube.com/watch?v=M6vZ6b75p9k)
3. **Evaluating RAG Performance on 100 Documents (Ragas Framework)**
   - 📺 **Project Tutorial**: [Watch Video](https://www.youtube.com/watch?v=5fp6e5nhJRk)
