# RAG Systems

Retrieval augmented generation grounds answers in your documents. The
ingestion pipeline chunks long texts with overlap, embeds each chunk into
vectors, and stores them in a vector database such as Chroma or Qdrant.

At query time, embed the question and run cosine similarity search for the
top k passages. Hybrid search combines sparse BM25 keyword matching with
dense semantic vectors, then a reranker model re-scores candidates. This
reranking step puts the most relevant passage first.

Chunk size controls the tradeoff: small chunks are precise, large chunks
keep context. Always return citations with chunk identifiers so users can
verify every claim against the source passage.
