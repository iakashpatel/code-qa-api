from fastapi import FastAPI, HTTPException, Query
import faiss
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
from repository_processor import (
    EmbeddingGenerator,
    chunk_repository,
    create_index,
)

# FastAPI App
app = FastAPI()

# Load FAISS Index and Metadata
INDEX_FILE = "repository_index.faiss"
METADATA_FILE = "metadata.json"


class AnswerGenerator:
    def __init__(self, model_name: str = "gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate_answer(self, context: str, question: str) -> str:
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        )
        outputs = self.model.generate(
            inputs["input_ids"], max_length=150, num_return_sequences=1
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# Load Index and Metadata
print("Loading FAISS index and metadata...")
index = faiss.read_index(INDEX_FILE)
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

embedding_generator = EmbeddingGenerator()
answer_generator = AnswerGenerator(model_name="gpt-3.5-turbo")  # Example model name


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/query")
async def query_api(
    question: str = Query(
        ..., description="The natural language question to query the repository"
    )
):
    """
    Query the indexed repository and generate an answer using the RAG pipeline.
    """
    try:
        # Step 1: Generate query embedding
        query_embedding = (
            embedding_generator.generate_embedding(question)
            .astype("float32")
            .reshape(1, -1)
        )

        # Step 2: Retrieve relevant chunks
        k = 5  # Number of results to return
        distances, indices = index.search(query_embedding, k)

        # Step 3: Combine retrieved chunks into a single context
        context = ""
        for idx in indices[0]:
            if idx < 0:
                continue
            chunk = metadata[idx]
            context += f"File: {chunk['file']} (Lines: {chunk['start_idx']}+)\n{chunk['text']}\n\n"

        if not context:
            return {"query": question, "answer": "No relevant information found."}

        # Step 4: Generate an answer
        answer = answer_generator.generate_answer(context, question)

        return {"query": question, "context": context, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index")
async def reindex_repository(
    repo_path: str = Query(..., description="Path to the repository to re-index")
):
    """
    Re-index the repository.
    """
    try:
        print("Chunking the repository...")
        chunks = chunk_repository(repo_path)

        print("Creating a new FAISS index...")
        new_index, new_metadata = create_index(chunks, embedding_generator)

        # Save the new index and metadata
        faiss.write_index(new_index, INDEX_FILE)
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(new_metadata, f, indent=4)

        return {"status": "success", "message": "Repository re-indexed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
