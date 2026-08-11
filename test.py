import chromadb
client = chromadb.PersistentClient(path="./chroma_db")  # or your absolute path
print(client._system.settings.persist_directory)