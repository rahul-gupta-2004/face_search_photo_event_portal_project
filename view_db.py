import chromadb

def view_database_contents():
    print("START: Reading Local Vector Database...")
    
    try:
        chroma_client = chromadb.PersistentClient(path = "./chroma_db")
        collection = chroma_client.get_collection(name = "event_photos")
        
        # Fetch everything stored in the collection
        results = collection.get(include = ["metadatas", "embeddings"])
        total_records = len(results["ids"])
        
        print(f"SUCCESS: Connected. Found {total_records} face entry records.\n")
        print("------------------------------")
        
        for idx in range(total_records):
            face_id = results["ids"][idx]
            metadata = results["metadatas"][idx]
            file_path = metadata.get("file_path", "Unknown Path")
            
            # Unpack vector dimensions safely
            vector_dims = len(results["embeddings"][idx]) if results["embeddings"] else 0
            
            print(f"ID: {face_id} | Image: {file_path} | Vector Dimensions: {vector_dims}")
            
        print("------------------------------")
        print("FINISHED: Database display complete.")
        
    except Exception as e:
        print(f"ERROR: Failed to read database records: {str(e)}")

if __name__ == "__main__":
    view_database_contents()