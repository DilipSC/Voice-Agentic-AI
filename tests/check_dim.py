from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

embedding = model.encode(["hello world"], normalize_embeddings=True)[0]

print("Embedding length:", len(embedding))