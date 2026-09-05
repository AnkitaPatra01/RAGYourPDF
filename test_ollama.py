import ollama

response = ollama.embed(
    model="nomic-embed-text",
    input=["Hello, this is a test document"]
)

embeddings = response["embeddings"]

print("Number of embeddings:", len(embeddings))
print("Embedding dimensions:", len(embeddings[0]))