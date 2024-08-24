import streamlit as st
import pandas as pd
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
import torch

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
    """Generate embedding for a single text using DistilBERT."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

def search_similar(query, df, top_k=5):
    """Search for similar chunks."""
    query_embedding = get_embedding(query)
    df['similarity'] = df['vector_embedding'].apply(lambda x: np.dot(x, query_embedding))
    return df.sort_values('similarity', ascending=False).head(top_k)

def load_data(file):
    df = pd.read_csv(file)
    df['vector_embedding'] = df['vector_embedding'].apply(lambda x: np.fromstring(x, sep=','))
    return df

st.title("Document Search Application")

# File upload instructions
st.header("File Upload Instructions")
st.write("""
If you're uploading your own CSV file, please ensure it follows this format:
- The file should be a CSV (Comma-Separated Values) file.
- It should contain the following columns:
  1. chunk_id (integer): A unique identifier for each text chunk
  2. document_id (string): An identifier for the source document
  3. document_file (string): The file path or name of the source document
  4. chunk_text (string): The actual text content of the chunk
  5. vector_embedding (string): The embedding vector as a comma-separated string of numbers
- The vector_embedding should be a string representation of the numerical values, separated by commas.
  For example: "0.1,0.2,0.3,0.4,..."

Example:
```
chunk_id,document_id,document_file,chunk_text,vector_embedding
1,doc1,/path/to/doc1.pdf,"This is a sample text.",0.1,0.2,0.3,0.4,...
2,doc1,/path/to/doc1.pdf,"This is another sample.",0.2,0.3,0.4,0.5,...
```
""")

# File upload or use predefined file
file_option = st.radio("Choose a file option:", ("Upload a CSV file", "Use predefined CSV file"))

if file_option == "Upload a CSV file":
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.success("File uploaded successfully!")
else:
    predefined_file = "vector_final_db.csv"  # Replace with your predefined file path
    df = load_data(predefined_file)
    st.success(f"Using predefined file: {predefined_file}")

# Manual data entry
st.header("Add New Data")
with st.form("new_data_form"):
    new_chunk_id = st.number_input("Chunk ID", min_value=0, step=1)
    new_document_id = st.text_input("Document ID")
    new_document_file = st.text_input("Document File")
    new_chunk_text = st.text_area("Chunk Text")
    
    submitted = st.form_submit_button("Add New Data")
    if submitted:
        new_embedding = get_embedding(new_chunk_text)
        new_row = pd.DataFrame({
            'chunk_id': [new_chunk_id],
            'document_id': [new_document_id],
            'document_file': [new_document_file],
            'chunk_text': [new_chunk_text],
            'vector_embedding': [new_embedding]
        })
        df = pd.concat([df, new_row], ignore_index=True)
        st.success("New data added successfully!")

# Search functionality
st.header("Search")
query = st.text_input("Enter your search query:")

if st.button("Search"):
    if query:
        results = search_similar(query, df)
        st.subheader("Search Results")
        for _, row in results.iterrows():
            st.write(f"Document: {row['document_file']}")
            st.write(f"Chunk ID: {row['chunk_id']}")
            st.write(f"Similarity: {row['similarity']:.4f}")
            st.write(f"Text: {row['chunk_text']}")
            st.write("---")
    else:
        st.warning("Please enter a search query.")

# Display some statistics about the loaded data
if 'df' in locals():
    st.sidebar.header("Dataset Statistics")
    st.sidebar.write(f"Total documents: {df['document_id'].nunique()}")
    st.sidebar.write(f"Total chunks: {len(df)}")
