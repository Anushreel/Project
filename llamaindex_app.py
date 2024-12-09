import streamlit as st
import os
import pdfplumber
from llama_index.core import Settings, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

# Set up environment variables for Gemini API key
os.environ['GEMINI_API_KEY'] = os.getenv("GEMINI_API_KEY")

# Set up Gemini LLM and embed model
llm = Gemini(api_key=os.environ["GEMINI_API_KEY"])
embed_model = GeminiEmbedding(model_name="models/embedding-001", api_key=os.environ["GEMINI_API_KEY"])

# Function to extract text and tables from PDF
def extract_pdf_content(pdf_path):
    text_content = ""
    tables = []

    with pdfplumber.open(pdf_path) as pdf:
        # Extract text from all pages
        for page in pdf.pages:
            text_content += page.extract_text() + "\n"  # Extract text
        
        # Extract tables from each page
        for page in pdf.pages:
            page_tables = page.extract_tables()
            for table in page_tables:
                tables.append(table)
    
    return text_content, tables

# Convert the extracted tables into a string format suitable for LlamaIndex
def format_tables_as_text(tables):
    table_text = ""
    for table in tables:
        for row in table:
            # Handle None values in the table row by replacing with empty string
            row = [str(cell) if cell is not None else "" for cell in row]
            table_text += "\t".join(row) + "\n"  # Tab-separated row
        table_text += "\n---\n"  # Separator between tables (optional)
    return table_text

# Streamlit UI
st.title("PDF Query Engine using Gemini")

# Upload PDF file
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded PDF to a temporary location
    pdf_path = f"./{uploaded_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Extract content and tables from the uploaded PDF
    text_content, tables = extract_pdf_content(pdf_path)

    # Format tables into text
    table_text = format_tables_as_text(tables)

    # Combine text and table data into a single document
    combined_text = text_content + "\n\n" + "Tables extracted from the PDF:\n" + table_text

    # Initialize Chroma Client and VectorStore
    load_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = load_client.get_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Create a Settings object and configure it with Gemini LLM and embeddings
    Settings.llm = llm
    Settings.embed_model = embed_model

    # Create the index from the vector store
    index = VectorStoreIndex.from_vector_store(vector_store)

    # Display the first 500 characters of the extracted text (for preview)
    st.subheader("Extracted Text Preview:")
    st.text(combined_text[:500])  # Display the first 500 characters of the extracted text

    # User input for query
    query = st.text_input("Enter your query")

    if query:
        # Query the index and get the response
        test_query_engine = index.as_query_engine()
        response = test_query_engine.query(query)

        # Display the response
        st.subheader("Response:")
        
        # Assuming the correct response structure, extract and display the text
        st.write(response)  # This could be adjusted based on response structure
