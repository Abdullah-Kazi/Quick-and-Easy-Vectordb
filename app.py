import streamlit as st
import pandas as pd
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
import torch

# Page configuration
st.set_page_config(page_title="Document Search", layout="wide")

# Initialize DistilBERT model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

def search_similar(query, df, top_k=5):
    query_embedding = get_embedding(query)
    df['similarity'] = df['vector_embedding'].apply(lambda x: np.dot(x, query_embedding))
    return df.sort_values('similarity', ascending=False).head(top_k)

def load_data(file):
    df = pd.read_csv(file)
    df['vector_embedding'] = df['vector_embedding'].apply(lambda x: np.fromstring(x, sep=','))
    return df

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Search", "Add New Data", "Upload File", "Instructions"])

# Load or initialize the dataframe
if 'df' not in st.session_state:
    predefined_file = "vector_final_db.csv"
    st.session_state.df = load_data(predefined_file)

# Main area
if page == "Search":
    st.title("Document Search")

    query = st.text_input("Enter your search query:")

    if st.button("Search"):
        if query:
            results = search_similar(query, st.session_state.df)
            st.subheader("Search Results")
            for _, row in results.iterrows():
                with st.expander(f"Result - Similarity: {row['similarity']:.4f}"):
                    st.markdown(f"**Document ID:** {row['document_id']}")
                    st.markdown(f"**Chunk ID:** {row['chunk_id']}")
                    st.markdown("**Text:**")
                    st.write(row['chunk_text'])
        else:
            st.warning("Please enter a search query.")

elif page == "Add New Data":
    st.title("Add New Data")
    
    new_chunk_id = st.number_input("Chunk ID", min_value=0, step=1)
    new_document_id = st.text_input("Document ID")
    new_chunk_text = st.text_area("Chunk Text")
    
    if st.button("Submit New Data"):
        new_embedding = get_embedding(new_chunk_text)
        new_row = pd.DataFrame({
            'chunk_id': [new_chunk_id],
            'document_id': [new_document_id],
            'chunk_text': [new_chunk_text],
            'vector_embedding': [new_embedding]
        })
        st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
        st.success("New data added successfully!")

elif page == "Upload File":
    st.title("Upload CSV File")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        st.session_state.df = load_data(uploaded_file)
        st.success("File uploaded successfully!")

elif page == "Instructions":
    st.title("CSV File Format Instructions")
    st.write("""
    If you're uploading your own CSV file, please ensure it follows this format:
    - The file should be a CSV (Comma-Separated Values) file.
    - It should contain the following columns:
      1. chunk_id (integer): A unique identifier for each text chunk
      2. document_id (string): An identifier for the source document
      3. chunk_text (string): The actual text content of the chunk
      4. vector_embedding (string): The embedding vector as a comma-separated string of numbers

    Example:
    ```
    chunk_id,document_id,chunk_text,vector_embedding
    1,doc1,"This is a sample text.",0.1,0.2,0.3,0.4,...
    2,doc1,"This is another sample.",0.2,0.3,0.4,0.5,...
    ```
    """)

# Display some statistics about the loaded data
st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Statistics")
st.sidebar.write(f"Total documents: {st.session_state.df['document_id'].nunique()}")
st.sidebar.write(f"Total chunks: {len(st.session_state.df)}")
