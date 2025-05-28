from pymilvus import MilvusClient, model

# https://milvus.io/docs/quickstart.md

client = MilvusClient(
    uri="https://in03-c49fdbccd9c0d36.serverless.gcp-us-west1.cloud.zilliz.com",
    user="db_c49fdbccd9c0d36",
    password="Ky4~V}hRaFk%VxR?",
)

if client.has_collection(collection_name="demo_collection"):
    client.drop_collection(collection_name="demo_collection")
client.create_collection(
    collection_name="demo_collection",
    dimension=768,
)

embedding_fn = model.DefaultEmbeddingFunction()
# Text strings to search from.
docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]

vectors = embedding_fn.encode_documents(docs)

print("Dim:", embedding_fn.dim, vectors[0].shape)

data = [
    {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"}
    for i in range(len(vectors))
]

print(f"Data has {len(data)} entities,each with fileds: {data[0].keys()}")
print(f"Vector dim: {len(data[0]["vector"])}")

# insert data
res = client.insert(collection_name="demo_collection", data=data)
print(res)

# semantic search
query_vectors = embedding_fn.encode_queries(["Who is Alan Turing?"])
res = client.search(
    collection_name="demo_collection",
    data=query_vectors,
    limit=2,
    output_fields=["text", "subject"],
)
for r in res[0]:
    print(r)

# vector search with metadata filtering
docs = [
    "Machine learning has been used for drug design.",
    "Computational synthesis with AI algorithms predicts molecular properties.",
    "DDR1 is involved in cancers and fibrosis.",
]
vectors = embedding_fn.encode_documents(docs)
data = [
    {"id": 3 + i, "vector": vectors[i], "text": docs[i], "subject": "biology"}
    for i in range(len(vectors))
]
client.insert(collection_name="demo_collection", data=data)
res = client.search(
    collection_name="demo_collection",
    data=embedding_fn.encode_queries(["tell me AI related information"]),
    filter="subject == 'biology'",
    limit=2,
    output_fields=["text", "subject"],
)
print("\nsearch with metadata")
for r in res[0]:
    print(r)


def main():
    print("Hello from learn-milvus!")


if __name__ == "__main__":
    main()
