# Hugging Face Documentation RAG using ChromaDB

This example demonstrates how to build a Retrieval Augmented Generation (RAG) system using SmoLAgents and LangChain to answer questions about Hugging Face documentation. The system combines vector search, document retrieval, and large language models to provide accurate responses based on official documentation.

## Features

- Downloads and processes Hugging Face documentation from the `m-ric/huggingface_doc` dataset
- Uses LangChain for document processing and text splitting
- Implements semantic search using ChromaDB and sentence transformers
- Provides a custom RetrieverTool for SmoLAgents to perform documentation searches
- Supports multiple LLM backends (Groq, Anthropic, OpenAI, Hugging Face)
- Integration with RagaAI-Catalyst for tracing and monitoring
- Efficient document deduplication and chunking
- Optimized for performance with M1 Macs

## Components

- **Document Processing**:
  - RecursiveCharacterTextSplitter with HuggingFace tokenizer
  - Chunk size: 200 tokens with 20 token overlap
  - Smart text splitting using multiple separators
  - Automatic deduplication of content

- **Embeddings**:
  - Model: `sentence-transformers/all-MiniLM-L6-v2`
  - Vector store: ChromaDB with persistence
  - Efficient similarity search implementation

- **RetrieverTool**:
  - Custom SmoLAgents tool for semantic search
  - Top-k retrieval (k=3)
  - Formatted document presentation
  - Type-safe query handling

## Requirements

- Python 3.x
- RagaAI-Catalyst credentials
- LLM API access (one of):
  - Groq API key
  - Anthropic API key
  - OpenAI API key
  - Hugging Face API key
- Dependencies:
  - datasets
  - langchain
  - chromadb
  - transformers
  - sentence-transformers
  - tqdm

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables in `.env`:
```env
CATALYST_ACCESS_KEY=your_access_key
CATALYST_SECRET_KEY=your_secret_key
CATALYST_BASE_URL=your_base_url
PROJECT_NAME=your_project_name
DATASET_NAME=your_dataset_name
GROQ_API_KEY=your_groq_api_key  # or other LLM provider key
```

3. Run the example:
```bash
python rag_using_chromadb.py
```

## How it Works

1. **Document Processing**:
   - Loads Hugging Face documentation dataset
   - Splits documents into semantic chunks
   - Removes duplicate content
   - Preserves source metadata

2. **Vector Store**:
   - Generates embeddings using sentence-transformers
   - Stores vectors in persistent ChromaDB
   - Enables efficient similarity search

3. **RetrieverTool**:
   - Implements semantic search functionality
   - Returns top 3 most relevant documentation snippets
   - Formats results for easy consumption

4. **Agent Interaction**:
   - Uses CodeAgent with LiteLLMModel
   - Performs contextual retrieval
   - Generates accurate responses

## Performance Notes

- Document embedding takes approximately 5 minutes on M1 Pro MacBook
- ChromaDB persistence enables reuse of computed embeddings
- Efficient memory usage through chunking and deduplication
- Optimized for both accuracy and response time

## Integration with RagaAI-Catalyst

The system integrates with RagaAI-Catalyst for:
- Tracing agent interactions
- Monitoring performance
- Tracking usage patterns
- Quality assessment

## Note

This example demonstrates the power of combining:
- Vector search (ChromaDB)
- Semantic understanding (sentence-transformers)
- Large language models (via LiteLLM)
- Agent-based interaction (SmoLAgents)

All orchestrated through RagaAI-Catalyst for a production-ready documentation QA system.
