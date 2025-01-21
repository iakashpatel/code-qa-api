from fastapi import FastAPI, HTTPException, Query, Request
import faiss
import json
import openai
import numpy as np
import os
from typing import List, Dict, Optional
import tiktoken

# Environment setup for OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Constants for file paths
ROOT_PATH = os.getcwd()
INDEX_FILE = f"{ROOT_PATH}/indexes/repository_index.faiss"
METADATA_FILE = f"{ROOT_PATH}/metadata/metadata.json"
REPO_PATH = f"{ROOT_PATH}/data/grip-no-tests"

# FastAPI App
app = FastAPI()


class OpenAIEmbeddingGenerator:
    def __init__(self, model_name: str = "text-embedding-ada-002"):
        self.model_name = model_name

    def generate_embedding(self, text: str) -> np.ndarray:
        response = openai.Embedding.create(model=self.model_name, input=[text])
        embedding = response["data"][0]["embedding"]
        return np.array(embedding, dtype="float32")


class OpenAIAnswerGenerator:
    def __init__(self, model_name: str = "gpt-4-0613"):
        self.model_name = model_name

    def generate_answer(self, context: str, question: str) -> str:
        prompt = f"""
        You are a code assistant. Use the provided context to answer the user's question. 
        Return a concise explanation and include relevant code snippets if applicable.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert assistant for software development.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0,
        )
        return response["choices"][0]["message"]["content"].strip()


# Initialize
embedding_generator = OpenAIEmbeddingGenerator()
answer_generator = OpenAIAnswerGenerator()


def chunk_repository(
    repo_path: str,
    ignored_dirs: Optional[List[str]] = None,
    valid_extensions: Optional[List[str]] = None,
    chunk_size: int = 500,
    follow_symlinks: bool = False,
) -> List[Dict]:
    """
    Traverse the repository (recursively) and extract code/documentation chunks.

    :param repo_path: The root path of the repository to scan.
    :param ignored_dirs: A list of directory names to ignore (e.g., ['.git', '__pycache__']).
    :param valid_extensions: A list of file extensions that you want to chunk (e.g., ['.py', '.md', '.txt']).
    :param chunk_size: Number of characters to include in each chunk.
    :param follow_symlinks: Whether to follow symbolic links.
    :return: A list of dictionaries, each containing text, file path, and start index.
    """

    if ignored_dirs is None:
        # Common directories to ignore in code repos
        ignored_dirs = {".git", "__pycache__", "venv", "node_modules", ".idea"}
    else:
        ignored_dirs = set(ignored_dirs)

    if valid_extensions is None:
        # Default to Python, Markdown, and plain text files
        valid_extensions = [".py", ".md", ".txt", ".html", ".yml", ".yaml", ".css"]

    chunks = []
    visited_dirs = set()

    # os.walk by default doesn't follow symbolic links, but let's be explicit
    for root, dirs, files in os.walk(repo_path, followlinks=follow_symlinks):
        # 1. Avoid repeated or unwanted directories
        #    This modifies dirs in-place, so os.walk won't recurse into them
        dirs[:] = [d for d in dirs if d not in ignored_dirs]

        # 2. Prevent infinite loops due to symlinks referencing already-visited directories
        abs_root = os.path.abspath(root)
        if abs_root in visited_dirs:
            continue
        visited_dirs.add(abs_root)

        # 3. Process files with valid extensions
        for file in files:
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in valid_extensions:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except (UnicodeDecodeError, PermissionError):
                    # Skip files that can't be opened or read properly
                    continue

                # Split content into manageable chunks
                for i in range(0, len(content), chunk_size):
                    chunk_text = content[i : i + chunk_size]
                    chunks.append(
                        {
                            "text": chunk_text,
                            "file": file_path,
                            "start_idx": i,
                        }
                    )

    return chunks


def initialize_index(repo_path: str):
    """
    Create the FAISS index and metadata if they don't exist.
    This is a placeholder for your actual indexing logic.
    """
    if not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE):
        print("Index or metadata file missing. Creating new ones...")

        chunks = chunk_repository(repo_path)
        embeddings = []
        meta = []

        for chunk in chunks:
            emb = embedding_generator.generate_embedding(chunk["text"])
            embeddings.append(emb)
            meta.append(chunk)

        dim = len(embeddings[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings, dtype="float32"))
        faiss.write_index(index, INDEX_FILE)

        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)
        print("Index and metadata created.")
    else:
        print("Index and metadata files found.")


def exceeds_token_limit(
    text: str, model_name: str = "gpt-4-0613", threshold_ratio: float = 0.9
) -> bool:
    """
    Checks if the text likely exceeds the model's token limit (using a threshold).
    - threshold_ratio: how close to the limit we allow before summarizing (e.g., 0.9 means 90% of the max tokens).
    """
    # Different GPT-4 variants have different max token lengths (8k, 16k, 32k).
    # We'll assume 8k tokens for illustration.
    # Adjust this to match your actual model's capacity.
    model_token_limit = 8192

    # Use tiktoken to count tokens:
    encoding = tiktoken.encoding_for_model(model_name)
    token_count = len(encoding.encode(text))

    # Compare token count to threshold
    if token_count > (model_token_limit * threshold_ratio):
        return True
    return False


def summarize_long_context(chunks: list, model_name: str = "gpt-4-0613") -> str:
    """
    Summarizes multiple chunks into a more concise context to fit within token limits.
    This is a simple 'map' approach where each chunk is summarized individually
    then concatenated. For a large codebase, you might do a 'map-reduce' approach.
    """
    summarized_contexts = []
    for chunk in chunks:
        # Summarize each chunk
        summary_prompt = f"Summarize the following code chunk:\n\n{chunk}"
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an expert code summarizer."},
                {"role": "user", "content": summary_prompt},
            ],
            max_tokens=200,
            temperature=0,
        )
        summary = response["choices"][0]["message"]["content"].strip()
        summarized_contexts.append(summary)

    # Combine all summaries into one string
    final_summary = "\n\n".join(summarized_contexts)
    return final_summary


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "ok"}


@app.post("/query")
async def query_api(request: Request):
    """
    Query the indexed repository and generate an answer using the RAG pipeline.
    """
    try:
        # Initialize index if files are missing
        initialize_index(REPO_PATH)
        body = await request.json()
        question = body.get("question")
        if not question:
            return {"error": "question is required."}
        # Load FAISS index and metadata
        index = faiss.read_index(INDEX_FILE)
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Generate query embedding
        query_embedding = embedding_generator.generate_embedding(question).reshape(
            1, -1
        )

        # Retrieve more chunks
        k = 30
        distances, indices = index.search(query_embedding, k)

        # Build context
        raw_contexts = []
        for idx in indices[0]:
            if idx < 0:
                continue
            chunk = metadata[idx]
            snippet = (
                f"File: {chunk['file']} (Lines: {chunk['start_idx']}+)\n"
                f"{chunk['text']}\n"
            )
            raw_contexts.append(snippet)

        # If no relevant info, return early
        if not raw_contexts:
            return {"query": question, "answer": "No relevant information found."}

        combined_context = "\n\n".join(raw_contexts)

        # Check token usage
        if exceeds_token_limit(combined_context):
            combined_context = summarize_long_context(raw_contexts)

        # Generate answer
        answer = answer_generator.generate_answer(combined_context, question)
        return {"query": question, "context": combined_context, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index")
async def reindex_repository():
    """
    Re-index the repository using OpenAI embeddings.
    """
    try:
        # Remove existing index/metadata if you want a fresh reindex
        if os.path.exists(INDEX_FILE):
            os.remove(INDEX_FILE)
        if os.path.exists(METADATA_FILE):
            os.remove(METADATA_FILE)

        # Re-initialize index
        initialize_index(REPO_PATH)
        return {"status": "success", "message": "Repository re-indexed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
