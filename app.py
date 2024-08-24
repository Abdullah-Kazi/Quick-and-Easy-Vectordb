# Import necessary libraries
import streamlit as st  # For creating the web app interface
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
from transformers import DistilBertTokenizer, DistilBertModel  # For text embedding
import torch  # For machine learning operations
import os  # For file and directory operations

# Configure the Streamlit page
st.set_page_config(page_title="Document Search", layout="wide")

# Initialize the DistilBERT model and tokenizer
@st.cache_resource  # This decorator helps cache the model to improve performance
def load_model():
    # Load the pre-trained tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    # Set up the device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move the model to the appropriate device
    return tokenizer, model, device

# Load the model and tokenizer
tokenizer, model, device = load_model()

# Function to generate text embeddings
def get_embedding(text):
    # Tokenize the input text and prepare it for the model
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to the appropriate device
    
    # Generate the embedding
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(**inputs)
    
    # Return the mean of the last hidden state as the embedding
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

# Function to search for similar text chunks
def search_similar(query, df, top_k=5):
    # Generate embedding for the query
    query_embedding = get_embedding(query)
    
    # Calculate similarity between query and all chunks
    df['similarity'] = df['vector_embedding'].apply(lambda x: np.dot(x, query_embedding))
    
    # Return top k most similar results
    return df.sort_values('similarity', ascending=False).head(top_k)

# Function to load data from a CSV file
def load_data(file):
    # Read the CSV file
    df = pd.read_csv(file)
    
    # Convert the string representation of embeddings back to numpy arrays
    df['vector_embedding'] = df['vector_embedding'].apply(lambda x: np.fromstring(x, sep=','))
    
    return df

# Set up the sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Instructions", "Upload File", "Search", "Add New Data"])

# Load or initialize the dataframe
if 'df' not in st.session_state:
    predefined_file = "vector_final_db.csv"
    if os.path.exists(predefined_file):
        # Load the predefined file if it exists
        st.session_state.df = load_data(predefined_file)
    else:
        # Create an empty dataframe if the file doesn't exist
        st.warning(f"Predefined file '{predefined_file}' not found. Please upload a CSV file.")
        st.session_state.df = pd.DataFrame(columns=['chunk_id', 'document_id', 'chunk_text', 'vector_embedding'])

# Main area of the app
if page == "Search":
    st.title("Document Search")

    if st.session_state.df.empty:
        st.warning("No data available. Please upload a CSV file or add new data.")
    else:
        # Create an input box for the search query
        query = st.text_input("Enter your search query:")

        if st.button("Search"):
            if query:
                # Perform the search and display results
                results = search_similar(query, st.session_state.df)
                st.subheader("Search Results")
                for _, row in results.iterrows():
                    with st.expander(f"Result - Similarity: {row['similarity']:.4f}"):
                        st.markdown(f"**Chunk ID:** {row['chunk_id']}")
                        st.markdown("**Text:**")
                        st.write(row['chunk_text'])
            else:
                st.warning("Please enter a search query.")

elif page == "Add New Data":
    st.title("Add New Data")
    
    # Create input fields for new data
    new_chunk_id = st.number_input("Chunk ID", min_value=0, step=1)
    new_chunk_text = st.text_area("Chunk Text")
    
    if st.button("Submit New Data"):
        # Generate embedding for the new text
        new_embedding = get_embedding(new_chunk_text)
        
        # Create a new row of data
        new_row = pd.DataFrame({
            'chunk_id': [new_chunk_id],
            'document_id': ['user_added'],
            'chunk_text': [new_chunk_text],
            'vector_embedding': [new_embedding]
        })
        
        # Add the new row to the existing dataframe
        st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
        st.success("New data added successfully!")

elif page == "Upload File":
    st.title("Upload CSV File")
    
    # Display instructions for CSV file format
    st.write("""
    ### CSV File Format Instructions
    
    If you're uploading your own CSV file, please ensure it follows this format:
    - The file should be a CSV (Comma-Separated Values) file.
    - It should contain the following columns:
      1. chunk_id (integer): A unique identifier for each text chunk
      2. document_id (string): An identifier for the source document (used internally)
      3. chunk_text (string): The actual text content of the chunk
      4. vector_embedding (string): The embedding vector as a comma-separated string of numbers

    Example:
    ```
    chunk_id,document_id,chunk_text,vector_embedding
    1,doc1,"This is a sample text.",0.1,0.2,0.3,0.4,...
    2,doc1,"This is another sample.",0.2,0.3,0.4,0.5,...
    ```
    """)
    
    # Create a file uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            # Try to load the uploaded file
            st.session_state.df = load_data(uploaded_file)
            st.success("File uploaded successfully!")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

elif page == "Instructions":
    st.title("How to Use the Document Search App")
    
    # Display instructions for using the app
    st.write("""
    Welcome to the Document Search Application! Here's how to use each feature:

    1. **Search**:
       - Navigate to the 'Search' page.
       - Enter your query in the text box.
       - Click the 'Search' button to find similar text chunks.
       - Results will be displayed with similarity scores and content.

    2. **Add New Data**:
       - Go to the 'Add New Data' page.
       - Enter a unique Chunk ID and the text content.
       - Click 'Submit New Data' to add it to the database.

    3. **Upload File**:
       - Visit the 'Upload File' page.
       - Prepare your CSV file according to the format instructions provided on that page.
       - Use the file uploader to select and upload your CSV file.

    4. **Navigation**:
       - Use the sidebar to switch between different pages of the app.

    5. **Dataset Statistics**:
       - View the current number of chunks in the dataset in the sidebar.

    Remember, you need to either upload a file or add data manually before you can perform searches.
    """)

# Display statistics about the loaded data in the sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Statistics")
if not st.session_state.df.empty:
    st.sidebar.write(f"Total chunks: {len(st.session_state.df)}")
else:
    st.sidebar.write("No data loaded")
