import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
import os
import uuid
import PyPDF2
import io
import base64
import json

# Initialize DistilBERT model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Function to read different file formats
def read_file(file):
    if file.name.endswith('.txt'):
        return file.read().decode('utf-8')
    elif file.name.endswith('.json'):
        json_data = json.load(file)
        # Assuming the JSON contains a 'text' field. Adjust as necessary.
        return json_data.get('text', str(json_data))
    elif file.name.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(file)
        return ' '.join([page.extract_text() for page in pdf_reader.pages])
    else:
        st.error("Unsupported file format. Please upload a .txt, .json, or .pdf file.")
        return None

# Function to chunk text
def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Function to create embeddings
def create_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Function to search embeddings
def search_embeddings(query, embeddings, chunks):
    query_embedding = create_embedding(query)
    similarities = cosine_similarity([query_embedding], embeddings).flatten()
    top_indices = similarities.argsort()[-5:][::-1]
    return [(chunks[i], similarities[i]) for i in top_indices]

# Function to create a download link for the CSV file
def get_csv_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="document_chunks.csv">Download CSV File</a>'
    return href

# Streamlit app
def main():
    st.title("Multi-format Document Search App")

    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=["txt", "json", "pdf"])

    if uploaded_file is not None:
        # Read file content
        content = read_file(uploaded_file)

        if content:
            # Chunk the text
            chunks = chunk_text(content)

            # Create embeddings
            embeddings = [create_embedding(chunk) for chunk in chunks]

            # Save to CSV
            df = pd.DataFrame({
                'chunk_id': [str(uuid.uuid4()) for _ in range(len(chunks))],
                'doc_id': uploaded_file.name,
                'document_file': uploaded_file.name,
                'chunk_text': chunks,
                'vector_embedding': embeddings
            })
            df.to_csv('document_chunks.csv', index=False)

            st.success("File processed and saved!")

            # Add download button for CSV
            st.markdown(get_csv_download_link(df), unsafe_allow_html=True)

    # Search functionality
    st.header("Search Documents")
    query = st.text_input("Enter your search query")

    if query:
        if os.path.exists('document_chunks.csv'):
            df = pd.read_csv('document_chunks.csv')
            chunks = df['chunk_text'].tolist()
            embeddings = np.array(df['vector_embedding'].tolist())

            results = search_embeddings(query, embeddings, chunks)

            for chunk, score in results:
                st.write(f"Similarity: {score:.4f}")
                st.write(chunk)
                st.write("---")
        else:
            st.warning("No documents have been processed yet.")

if __name__ == "__main__":
    main()
