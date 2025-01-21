import os
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
import faiss


# Load NLP Model for Embedding Generation
class EmbeddingGenerator:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate_embedding(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        outputs = self.model.encoder(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()


# Chunking the repository
def chunk_repository(repo_path: str) -> List[Dict]:
    """
    Traverse the repository and extract chunks of code or documentation.
    """
    chunks = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(
                (".py", ".md", ".txt", ".html", ".txt", ".yml", ".yaml")
            ):  # Adjust for relevant extensions
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Split content into manageable chunks
                chunk_size = 500  # Adjust for meaningful granularity
                for i in range(0, len(content), chunk_size):
                    chunks.append(
                        {
                            "text": content[i : i + chunk_size],
                            "file": file_path,
                            "start_idx": i,
                        }
                    )
    return chunks


# Indexing the chunks
def create_index(
    chunks: List[Dict], embedding_generator: EmbeddingGenerator
) -> Tuple[faiss.IndexFlatL2, List[Dict]]:
    """
    Create a vector index from the extracted chunks.
    """
    embeddings = []
    metadata = []

    for chunk in chunks:
        embedding = embedding_generator.generate_embedding(chunk["text"])
        embeddings.append(embedding)
        metadata.append(chunk)

    # Convert embeddings to FAISS-compatible format
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    return index, metadata
