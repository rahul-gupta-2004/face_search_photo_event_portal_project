import os
import sys

# Silences TensorFlow CPU/GPU warning logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Redirect stderr temporarily if C++ binaries bypass log levels during initialization
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

import cv2
import chromadb
from deepface import DeepFace

# Restore stderr right after imports are finished
sys.stderr = stderr

def run_organizer_ingestion():
    print("START: Initializing Local Vector Database")

    # Establish a persistent connection to the local database folder
    try:
        chroma_client = chromadb.PersistentClient(path = "./chroma_db")
        collection = chroma_client.get_or_create_collection(name = "event_photos")
        print("SUCCESS: Connected to ChromaDB")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to connect to ChromaDB: {str(e)}")
        return
    
    # Load OpenCV Haar Cascade Detector
    cascade_path = "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        print(f"CRITICAL ERROR: Missing {cascade_path} in workspace root directory")
        return
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Pre-Load ArcFace model weights into system memory
    try:
        print("Loading ArcFace model architecture")
        DeepFace.build_model("ArcFace")
        print("SUCCESS: ArcFace model is ready")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to build ArcFace model: {str(e)}")
        return
    
    image_folder = "images"
    if not os.path.exists(image_folder):
        print(f"ERROR: Folder '{image_folder}' does not exist")
        return
    
    # Filter out files and sort them sequentially (image_1.jpg to image_25.jpg)
    # Pass the extensions inside a tuple ()
    try:
        all_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        all_files.sort()
    except Exception as e:
        print(f"ERROR: Failed to read images directory: {str(e)}")
        return
    
    face_counter = 0
    print(f"INFO: Found {len(all_files)} files inside '{image_folder}' dataset folder. Scanning faces...")

    for filename in all_files:
        image_path = os.path.join(image_folder, filename)

        # Read file image into raw BGR pixels
        img = cv2.imread(image_path)
        if img is None:
            print(f"WARNING: Skipping unreadable image file: {filename}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect individual bounding face box regions
        detected_faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5)
        
        # Edge Case 1: If 0 faces found in image, skip entirely
        if len(detected_faces) == 0:
            print(f"SKIP: Image {filename} has no human faces detected")
            continue
            
        print(f"PROCESSING: Image {filename} - Detected {len(detected_faces)} faces")

        # Edge Case 2: Loop over every isolated face inside the same frame image
        for idx, (x, y, w, h) in enumerate(detected_faces):
            face_counter += 1

            # Crop exact pixel boundaries of the isolated face
            cropped_face = img[y:y+h, x:x+w]

            try:
                # Generate vectors natively on cropped slices
                embedding_objs = DeepFace.represent(
                    img_path = cropped_face,
                    model_name = "ArcFace",
                    detector_backend = "skip"
                )

                vector = embedding_objs[0]["embedding"]
                unique_face_id = f"face_{face_counter:03d}"

                # Push elements safely into the explicit database vectors row footprint
                collection.add(
                    ids = [unique_face_id],
                    embeddings = [vector],
                    metadatas = [{"file_path": image_path}]
                )  
            
            except Exception as e:
                print(f"WARNING: Could not compute vector for {filename} index {idx}: {str(e)}")

            
    print(f"\nFINISHED: Ingestion complete. Registered {face_counter} total faces inside ChromaDB")

if __name__ == "__main__":
    run_organizer_ingestion()