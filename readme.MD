# Photo Event Portal Project

A Streamlit application for a live event photo delivery portal. It uses pure vector similarity embeddings with ChromaDB and DeepFace to search for attendee photos.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirments.txt
   ```

2. Ingest photos into the database:
   ```bash
   python ingest.py
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Features

- Uses ChromaDB for vector storage.
- DeepFace (ArcFace model) for generating face embeddings.
- OpenCV for face localization.
- Dynamic cluster filtering to isolate matching photos.
