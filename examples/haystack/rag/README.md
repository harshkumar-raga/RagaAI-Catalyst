# Haystack RAG Example with RagaAI Catalyst

This example demonstrates how to implement a Retrieval-Augmented Generation (RAG) pipeline using Haystack and RagaAI Catalyst for tracing and monitoring.

## Overview

The example builds a RAG system that:
1. Downloads a text file about Leonardo da Vinci
2. Processes and indexes the document using OpenAI embeddings
3. Implements a query pipeline to answer questions about the text
4. Integrates RagaAI Catalyst for tracing and monitoring

## Prerequisites

- Python 3.x
- OpenAI API key
- RagaAI Catalyst credentials

## Environment Variables

Create a `.env` file with the following variables:

```
CATALYST_ACCESS_KEY=your_access_key
CATALYST_SECRET_KEY=your_secret_key
CATALYST_BASE_URL=your_base_url
PROJECT_NAME=your_project_name
DATASET_NAME=your_dataset_name
OPENAI_API_KEY=your_openai_api_key
```

## Components

### Indexing Pipeline
- TextFileToDocument: Converts text file to document format
- DocumentCleaner: Cleans the document
- DocumentSplitter: Splits document into chunks
- OpenAIDocumentEmbedder: Creates embeddings using OpenAI
- DocumentWriter: Writes to the document store

### RAG Pipeline
- OpenAITextEmbedder: Embeds query text
- InMemoryEmbeddingRetriever: Retrieves relevant documents
- ChatPromptBuilder: Builds prompts for the LLM
- OpenAIChatGenerator: Generates answers using OpenAI

## Usage

1. Set up your environment variables
2. Run the script:
```bash
python rag.py
```

The script will:
- Download a text file about Leonardo da Vinci
- Index and embed the content
- Answer a sample question about Leonardo's age at death

## Monitoring

The implementation includes RagaAI Catalyst integration for tracing and monitoring your RAG pipeline. Access the Catalyst dashboard to view metrics and traces.