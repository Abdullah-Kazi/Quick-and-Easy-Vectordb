import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
import os
import uuid
import csv
import json
import PyPDF2
import io
import base64

# Initialize DistilBERT tokenizer and model
@st.cache(allow_output_mutation=True)
def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    return tokenizer, model

tokenizer, model = load_model()

# Function to chunk text
def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Function to create embedding
def create_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# Function to process documents
def process_document(file):
    filename = file.name
    content = file.getvalue()
    
    if filename.endswith('.txt'):
        content = content.decode('utf-8')
    elif filename.endswith('.pdf'):
        content = extract_text_from_pdf(io.BytesIO(content))
    elif filename.endswith('.json'):
        json_data = json.loads(content.decode('utf-8'))
        if isinstance(json_data, dict):
            content = ' '.join(str(value) for value in json_data.values() if isinstance(value, (str, int, float)))
        elif isinstance(json_data, list):
            content = ' '.join(str(item) for item in json_data if isinstance(item, (str, int, float)))
    
    chunks = chunk_text(content)
    documents = []
    for chunk in chunks:
        documents.append({
            'chunk_id': str(uuid.uuid4()),
            'doc_id': filename,
            'document_file': filename,
            'chunk_text': chunk,
            'vector_embedding': create_embedding(chunk)
        })
    return documents

# Function to save data to CSV
def save_to_csv(documents, filename='document_embeddings.csv'):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['chunk_id', 'doc_id', 'document_file', 'chunk_text', 'vector_embedding'])
        for doc in documents:
            writer.writerow([
                doc['chunk_id'],
                doc['doc_id'],
                doc['document_file'],
                doc['chunk_text'],
                ','.join(map(str, doc['vector_embedding']))
            ])

# Function to load data from CSV
def load_from_csv(filename='document_embeddings.csv'):
    if not os.path.exists(filename):
        return []
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            data.append({
                'chunk_id': row[0],
                'doc_id': row[1],
                'document_file': row[2],
                'chunk_text': row[3],
                'vector_embedding': np.array(list(map(float, row[4].split(','))))
            })
    return data

# Function to search
def search(query, data):
    query_embedding = create_embedding(query)
    similarities = cosine_similarity([query_embedding], [doc['vector_embedding'] for doc in data])[0]
    top_indices = similarities.argsort()[::-1][:5]  # Get top 5 results
    results = [data[i] for i in top_indices]
    return results

# Function to create a download link for the CSV file
def get_csv_download_link(filename):
    with open(filename, 'r') as f:
        csv = f.read()
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

# Streamlit app
def main():
    st.title("Multi-format Document Search System with DistilBERT")

    # File uploader
    uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True, type=['txt', 'pdf', 'json'])

    if uploaded_files:
        # Process uploaded documents
        all_documents = load_from_csv()  # Load existing documents
        new_documents = []
        for file in uploaded_files:
            documents = process_document(file)
            new_documents.extend(documents)
        
        all_documents.extend(new_documents)  # Add new documents to existing ones
        
        # Save to CSV
        save_to_csv(all_documents)

        st.success(f"Processed {len(new_documents)} new document chunks. Total chunks: {len(all_documents)}")

    # Download link for CSV
    if os.path.exists('document_embeddings.csv'):
        st.markdown(get_csv_download_link('document_embeddings.csv'), unsafe_allow_html=True)

    # Search functionality
    st.header("Search")
    query = st.text_input("Enter your search query")

    if query:
        # Load data and embeddings from CSV
        data = load_from_csv()

        # Perform search
        results = search(query, data)

        # Display results
        for result in results:
            st.subheader(f"Document: {result['document_file']}")
            st.write(f"Chunk ID: {result['chunk_id']}")
            st.write(f"Text: {result['chunk_text'][:200]}...")  # Display first 200 characters
            st.write("---")

if __name__ == "__main__":
    main()
