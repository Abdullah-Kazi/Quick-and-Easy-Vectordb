import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
import uuid
import PyPDF2
import io
import json
import base64
import os

# ... (previous imports and model loading code remains the same)

# Function to save DataFrame to a file
def save_dataframe(df, filename='processed_data.pkl'):
    df.to_pickle(filename)

# Function to load DataFrame from a file
def load_dataframe(filename='processed_data.pkl'):
    if os.path.exists(filename):
        return pd.read_pickle(filename)
    return None

# ... (other functions like read_file, chunk_text, create_embedding, etc. remain the same)

# Streamlit app
def main():
    st.title("Persistent Multi-format Document Search App")

    # Load existing data if available
    df = load_dataframe()
    if df is not None:
        st.success("Existing processed data loaded!")
        st.markdown(get_file_download_link(df, 'csv'), unsafe_allow_html=True)
        st.markdown(get_file_download_link(df, 'tsv'), unsafe_allow_html=True)
        
        # Option to clear existing data
        if st.button("Clear existing data and upload new files"):
            df = None
            if os.path.exists('processed_data.pkl'):
                os.remove('processed_data.pkl')
            st.experimental_rerun()
    
    # File uploader for multiple files
    uploaded_files = st.file_uploader("Choose files", type=["txt", "json", "pdf", "tsv"], accept_multiple_files=True)

    if uploaded_files:
        all_chunks = []
        all_embeddings = []
        
        progress_bar = st.progress(0)
        for i, uploaded_file in enumerate(uploaded_files):
            content = read_file(uploaded_file)
            if content:
                chunks = chunk_text(content)
                embeddings = [create_embedding(chunk) for chunk in chunks]
                
                all_chunks.extend(chunks)
                all_embeddings.extend(embeddings)
            
            progress_bar.progress((i + 1) / len(uploaded_files))

        if all_chunks:
            df = pd.DataFrame({
                'chunk_id': [str(uuid.uuid4()) for _ in range(len(all_chunks))],
                'doc_id': [file.name for file in uploaded_files for _ in range(len(chunk_text(read_file(file))))],
                'document_file': [file.name for file in uploaded_files for _ in range(len(chunk_text(read_file(file))))],
                'chunk_text': all_chunks,
                'vector_embedding': all_embeddings
            })

            st.success("Files processed successfully!")
            st.markdown(get_file_download_link(df, 'csv'), unsafe_allow_html=True)
            st.markdown(get_file_download_link(df, 'tsv'), unsafe_allow_html=True)

            # Save the processed data
            save_dataframe(df)

        else:
            st.warning("No valid content found in uploaded files.")
            return

    # Search functionality
    if df is not None:
        st.header("Search Documents")
        query = st.text_input("Enter your search query")

        if query:
            chunks = df['chunk_text'].tolist()
            embeddings = np.array(df['vector_embedding'].tolist())

            results = search_embeddings(query, embeddings, chunks)

            for chunk, score in results:
                st.write(f"Similarity: {score:.4f}")
                st.write(chunk)
                st.write("---")

if __name__ == "__main__":
    main()
