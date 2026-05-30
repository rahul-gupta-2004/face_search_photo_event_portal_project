import chromadb

def clear_database_collection():
    print("START: Attempting Database Reset...")
    
    try:
        chroma_client = chromadb.PersistentClient(path = "./chroma_db")
        
        # Check existing collections
        existing_collections = [c.name for c in chroma_client.list_collections()]
        
        if "event_photos" in existing_collections:
            # Drop the collection completely
            chroma_client.delete_collection(name = "event_photos")
            print("SUCCESS: 'event_photos' collection has been completely deleted.")
            print("INFO: Run 'python3 ingest.py' now to create a fresh, clean index.")
        else:
            print("WARNING: Collection 'event_photos' does not exist. Nothing to clear.")
            
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to clear database collection: {str(e)}")

if __name__ == "__main__":
    clear_database_collection()