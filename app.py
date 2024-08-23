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

# Initialize DistilBERT model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    return tokenizer, model

tokenizer, model = load_model()

# Function to read different file formats
def read_file(file):
    if file.name.endswith('.txt'):
        return file.read().decode('utf-8')
    elif file.name.endswith('.json'):
        json_data = json.load(file)
        return json_data.get('text', str(json_data))
    elif file.name.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(file)
        return ' '.join([page.extract_text() for page in pdf_reader.pages])
    elif file.name.endswith('.tsv'):
        df = pd.read_csv(file, sep='\t')
        return ' '.join(df.astype(str).values.flatten())
    else:
        st.error(f"Unsupported file format: {file.name}. Please upload a .txt, .json, .pdf, or .tsv file.")
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
@st.cache_data
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

# Function to create a download link for the CSV/TSV file
def get_file_download_link(df, file_format):
    if file_format == 'csv':
        data = df.to_csv(index=False)
        filename = "document_chunks.csv"
        mime = "text/csv"
    else:  # TSV
        data = df.to_csv(index=False, sep='\t')
        filename = "document_chunks.tsv"
        mime = "text/tab-separated-values"
    
    b64 = base64.b64encode(data.encode()).decode()
    href = f'<a href="data:file/{mime};base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

# Streamlit app
def main():
    st.title("Multi-format Document Search App")

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
            
            # Provide options to download as CSV or TSV
            st.markdown(get_file_download_link(df, 'csv'), unsafe_allow_html=True)
            st.markdown(get_file_download_link(df, 'tsv'), unsafe_allow_html=True)

            # Store the dataframe in session state for searching
            st.session_state['search_df'] = df
        else:
            st.warning("No valid content found in uploaded files.")
            return

    # Search functionality
    st.header("Search Documents")
    query = st.text_input("Enter your search query")

    if query and 'search_df' in st.session_state:
        df = st.session_state['search_df']
        chunks = df['chunk_text'].tolist()
        embeddings = np.array(df['vector_embedding'].tolist())

        results = search_embeddings(query, embeddings, chunks)

        for chunk, score in results:
            st.write(f"Similarity: {score:.4f}")
            st.write(chunk)
            st.write("---")

if __name__ == "__main__":
    main()
