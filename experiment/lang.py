import os
import streamlit as st
import pdfplumber
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import Chroma
from langchain.schema import Document

# Streamlit UI
st.title("PDF Query Engine with Multiple Models")

# Upload PDF file
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Multi-Model Selection
model_choices = st.multiselect("Choose Models", ["Gemini", "Groq"])

# Prompt input
prompt_input = st.text_input("Enter your query")

if uploaded_file is not None and prompt_input and model_choices:
    # Save the uploaded PDF to a temporary location
    pdf_path = f"./{uploaded_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load the PDF using pdfplumber for table and text extraction
    with pdfplumber.open(pdf_path) as pdf:
        all_text = ""
        for page in pdf.pages:
            # Extract text
            text = page.extract_text()
            all_text += text + "\n"

            # Extract tables
            tables = page.extract_tables()
            for table in tables:
                filtered_table = [row for row in table if None not in row]
                table_text = "\n".join(["\t".join(row) for row in filtered_table])
                all_text += "\n" + table_text + "\n"

    # Create text chunks
    text_splitter = CharacterTextSplitter(
        separator=". ",
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(all_text)

    # Convert chunks into Document objects
    documents = [Document(page_content=chunk, metadata={}) for chunk in chunks]

    # Initialize models and embeddings
    responses = {}
    for model_choice in model_choices:
        if model_choice == "Gemini":
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                api_key=os.environ["GEMINI_API_KEY"]
            )
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=os.environ["GEMINI_API_KEY"]
            )
        elif model_choice == "Groq":
            llm = ChatGroq(
                model="mixtral-8x7b-32768",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                api_key=os.environ["GROQ_API_KEY"]
            )
            # embeddings = GroqEmbeddings(...)  # Uncomment if Groq embeddings are available
            embeddings = None  # Placeholder for Groq embeddings

        # Generate embeddings and store them in Chroma
        if embeddings:
            vectordb = Chroma.from_documents(documents, embeddings)
            retriever = vectordb.as_retriever(search_kwargs={"k": 5})

            # Create retrieval chain
            template = """
            You are a helpful AI assistant.
            Answer based on the context provided. 
            context: {context}
            input: {input}
            answer:
            """
            prompt = PromptTemplate.from_template(template)
            combine_docs_chain = create_stuff_documents_chain(llm, prompt)
            retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

            # Query the chain
            response = retrieval_chain.invoke({"input": prompt_input})
            responses[model_choice] = response["answer"]
        else:
            responses[model_choice] = "Embeddings not available for this model."

    # Display the responses
    st.subheader("Responses from Models:")
    for model, response in responses.items():
        st.markdown(f"### {model} Response")
        st.write(response)
