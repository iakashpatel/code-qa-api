# RAG-based Code Query API

This project provides a **Retrieval-Augmented Generation (RAG)** API for querying a codebase. It uses:

- **OpenAI embeddings** and **ChatCompletion** (GPT models),
- A **FAISS** index for vector similarity search,
- **FastAPI** for the RESTful API.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Notes & Troubleshooting](#notes--troubleshooting)

---

## Prerequisites

1. **Python 3.9+** (Recommended)
2. **Virtual environment** (e.g., `venv` or **conda**)
3. **FAISS** (for vector indexing):
   - **Conda (recommended)**:
     ```bash
     conda install -c pytorch faiss-cpu
     ```
   - **Pip** (if you find precompiled wheels):
     ```bash
     pip install faiss-cpu
     ```
   - Or **build from source** ([FAISS GitHub](https://github.com/facebookresearch/faiss))
4. **OpenAI Python Library** and dependencies
5. A valid **OpenAI API key** (for embeddings and GPT-based text generation).

---

## Installation

1. **Clone** this repository (or copy the code):

   ```bash
   git clone https://github.com/yourusername/rag-code-query.git
   cd rag-code-query
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows
   ```

3. Install required packages:
   ```python
   pip install -r requirements.txt
   ```

## Environment Variables

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

## Project Structure

```css
/
├── app.py # Main FastAPI application code
├── repository_processor.py
├── requirements.txt
├── README.md
└── ...
```

## Usage

1. Start the Server

Run the FastAPI application (assuming your main file is app.py):

```bash
uvicorn app:app --reload
```

This starts the server at http://127.0.0.1:8000.

2. Index a Repository

Before querying a repository, you need to index it. The indexing process:

- Chunks the code using chunk_repository.
- Generates OpenAI embeddings for each chunk.
- Builds a FAISS index (repository_index.faiss) and a metadata file (metadata.json).

You can automatically create or refresh the index by calling the /index endpoint or letting the code auto-initialize if the index/metadata files are missing.

example:

```curl
curl -X POST "http://127.0.0.1:8000/index
```

3. Query the Repository

Once indexed, you can send queries to the /query endpoint. The endpoint:

1. Retrieves the most relevant chunks from FAISS.
2. Optionally summarizes if necessary.
3. Uses GPT-4 to generate a final answer.

Example (replace placeholders accordingly)

```curl
curl --location 'http://127.0.0.1:8000/query' \
--header 'Content-Type: application/json' \
--data '{
    "question": "What does class Grip do?"
}'
```

### Health Check:

```curl
curl http://127.0.0.1:8000/health
```

## Notes & Troubleshooting

1. FAISS Installation

If pip install faiss-cpu doe

```bash
conda install -c pytorch faiss-cpu
```

or build from source.

2. OpenAI Rate Limits

If you encounter rate limits or 429 errors, you may need to throttle requests or upgrade your OpenAI plan.

3. Large Repositories

- Increase the chunk size or the number of retrieved chunks (k) carefully.
- Summarize or do multi-step retrieval if you exceed GPT’s context window.

4. Token Counting

If you’re hitting token limits, install and use tiktoken to check prompt lengths.

Summarize large chunks before sending them to GPT.
