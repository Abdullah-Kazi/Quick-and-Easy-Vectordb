import os
import csv
import streamlit as st
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity

# Initialize DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

def chunk_text(text, chunk_size=1000, overlap=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def process_documents(directory):
    data = []
    chunk_id = 0
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                text = file.read()
            chunks = chunk_text(text)
            for chunk in chunks:
                embedding = get_embedding(chunk)
                data.append({
                    'chunk_id': chunk_id,
                    'doc_id': filename,
                    'document_file': filepath,
                    'chunk_text': chunk,
                    'vector_embedding': embedding
                })
                chunk_id += 1
    return data

def save_to_csv(data, csv_filename):
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['chunk_id', 'doc_id', 'document_file', 'chunk_text', 'vector_embedding']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            row['vector_embedding'] = ','.join(map(str, row['vector_embedding']))
            writer.writerow(row)

def load_from_csv(csv_filename):
    data = []
    with open(csv_filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row['vector_embedding'] = np.array(list(map(float, row['vector_embedding'].split(','))))
            data.append(row)
    return data

def search(query, data, top_k=5):
    query_embedding = get_embedding(query)
    similarities = [cosine_similarity([query_embedding], [item['vector_embedding']])[0][0] for item in data]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [data[i] for i in top_indices]

# Streamlit UI
st.title("Document Embedding and Search System (DistilBERT)")

# File uploader for documents
uploaded_files = st.file_uploader("Upload text documents", accept_multiple_files=True, type=['txt'])

if uploaded_files:
    # Process uploaded documents
    data = []
    chunk_id = 0
    for uploaded_file in uploaded_files:
        text = uploaded_file.read().decode()
        chunks = chunk_text(text)
        for chunk in chunks:
            embedding = get_embedding(chunk)
            data.append({
                'chunk_id': chunk_id,
                'doc_id': uploaded_file.name,
                'document_file': uploaded_file.name,
                'chunk_text': chunk,
                'vector_embedding': embedding
            })
            chunk_id += 1

    # Save to CSV
    csv_filename = "embeddings.csv"
    save_to_csv(data, csv_filename)
    st.success(f"Embeddings saved to {csv_filename}")

    # Download CSV button
    with open(csv_filename, "rb") as file:
        st.download_button(
            label="Download CSV",
            data=file,
            file_name=csv_filename,
            mime="text/csv"
        )

    # Search functionality
    search_query = st.text_input("Enter your search query:")
    if search_query:
        results = search(search_query, data)
        for result in results:
            st.write(f"Chunk ID: {result['chunk_id']}")
            st.write(f"Document: {result['doc_id']}")
            st.write(f"Text: {result['chunk_text'][:200]}...")
            
            # Download button for the original document
            with open(result['document_file'], "rb") as file:
                st.download_button(
                    label=f"Download {result['doc_id']}",
                    data=file,
                    file_name=result['doc_id'],
                    mime="text/plain"
                )
            st.write("---")
