__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import sys

# Quiet TensorFlow warnings before deepface loads
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

import cv2
import chromadb
import streamlit as st
from deepface import DeepFace

# Restore stderr streams safely
sys.stderr = stderr

st.set_page_config(page_title="Face Search App", page_icon=":mag:", layout="wide")

st.title("Live Event Photo Delivery Portal")
st.write("Select an attendee profile to query the database using pure vector similarity embeddings with dynamic cluster filtering.")
st.write("---")

# Verify ChromaDB directory dependency
if not os.path.exists("./chroma_db"):
    st.error("ERROR: Database directory missing! Please run 'python3 ingest.py' first.")
    st.stop()

@st.cache_resource
def initialize_backend():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="event_photos")
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    DeepFace.build_model("ArcFace")
    return collection, face_cascade

collection, face_cascade = initialize_backend()

# SIDEBAR: Clean user mapping using your exact filenames directly
st.sidebar.header("User Registration Portal")
selected_user = st.sidebar.selectbox(
    "Choose User to Retrieve Photos:",
    ["-- Select Attendee --", "Chris Tucker", "Jackie Chan"]
)

if selected_user != "-- Select Attendee --":
    st.sidebar.write(f"Querying profile records for: **{selected_user}**")
    
    # Direct path mapping without messy conditional block clusters
    filename_map = {"Chris Tucker": "chris_tucker.png", "Jackie Chan": "jackie_chan.png"}
    sample_path = f"test_images/{filename_map[selected_user]}"
    
    if not os.path.exists(sample_path):
        st.sidebar.error(f"Missing registration asset: '{sample_path}'")
        st.stop()

    # Load and display active registration avatar in sidebar
    opencv_img = cv2.imread(sample_path)
    if opencv_img is None:
        st.error(f"ERROR: Failed to decode {sample_path}")
        st.stop()
        
    st.sidebar.image(cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB), caption="Active Reference File", width=180)
    
    with st.spinner("Extracting face embeddings and searching vector index..."):
        try:
            # Face localization pipeline
            gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
            detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            
            if len(detected_faces) == 0:
                st.warning("No face structure isolated from registration file.")
                st.stop()
                
            (x, y, w, h) = detected_faces[0]
            cropped_face = opencv_img[y:y+h, x:x+w]

            # Generate vector representations via ArcFace
            user_embedding_objs = DeepFace.represent(
                img_path=cropped_face, model_name="ArcFace", detector_backend="skip"
            )
            target_vector = user_embedding_objs[0]["embedding"]

            # Query ChromaDB Vector space for matches
            query_results = collection.query(query_embeddings=[target_vector], n_results=20)
            matched_photo_paths = []

            if query_results and 'metadatas' in query_results and len(query_results['metadatas'][0]) > 0:
                raw_metadatas = query_results['metadatas'][0]
                raw_distances = query_results['distances'][0]
                
                # Dynamic cutoff based on the maximum distance gap between consecutive matches
                cutoff_index = len(raw_distances)
                max_gap = 0.0
                
                for i in range(len(raw_distances) - 1):
                    gap = raw_distances[i+1] - raw_distances[i]
                    # A spike greater than 5.0 indicates a clear transition to a different person
                    if gap > 5.0 and gap > max_gap:
                        max_gap = gap
                        cutoff_index = i + 1
                        break
                
                # Filter results using the dynamic cutoff index
                for idx in range(cutoff_index):
                    path = raw_metadatas[idx]["file_path"]
                    if os.path.exists(path) and path not in matched_photo_paths:
                        matched_photo_paths.append(path)
            
            # SORT SYSTEM: Sort paths in ascending order naturally by file numeric ID string lengths
            matched_photo_paths.sort(key=lambda x: [int(s) if s.isdigit() else s.lower() for s in os.path.basename(x).replace('.', '_').split('_')])

            # RENDER LAYER: Display matches in a tight, clean 3-column table grid layout
            st.subheader(f"True Vector Matched Photos found ({len(matched_photo_paths)} images)")
            
            if len(matched_photo_paths) > 0:
                # Creates structural row arrays containing exactly 3 columns each
                for i in range(0, len(matched_photo_paths), 3):
                    row_paths = matched_photo_paths[i:i+3]
                    cols = st.columns(3)
                    
                    for index, path in enumerate(row_paths):
                        filename = os.path.basename(path)
                        display_img = cv2.imread(path)
                        
                        if display_img is not None:
                            with cols[index]:
                                # Renders cleanly with fixed, clear image sizing parameters
                                st.image(
                                    cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), 
                                    caption=filename, 
                                    width=240
                                )
                        else:
                            with cols[index]:
                                st.error(f"Read error: {filename}")
            else:
                st.warning("No matching vectors found in database index matching this user profile within the confidence threshold.")
                
        except Exception as err:
            st.error(f"Execution Error running vector query engine: {str(err)}")
else:
    st.info("Please select an attendee profile from the sidebar dropdown menu to simulate matching.")