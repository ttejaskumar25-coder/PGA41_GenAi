# Vector Databases & Retrieval-Augmented Generation (RAG)

Summary:
RAG combines vector search over embeddings with LLMs to ground responses in external documents. Vector DBs store embeddings for efficient similarity search.

Core concepts:
- Embeddings: mapping text to vectors (OpenAI, Hugging Face, SentenceTransformers)
- Vector DBs: Milvus, Pinecone, Weaviate, Chroma
- Indexing and similarity search (ANN algorithms)
- RAG pipelines: retrieve → condense → prompt → generate

Hands-on:
- Build a simple RAG pipeline: ingest docs, compute embeddings, store in a vector DB, implement retrieval and LLM augmentation

Backlinks:
- Source roadmap: https://roadmap.sh/ai-engineer

Suggested project: document-based Q&A with RAG and evaluation of faithfulness.