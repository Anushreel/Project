import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# Function to load models based on user selection (Groq or Gemini)
def load_model(model_selection):
    if model_selection == "Gemini":
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=os.getenv("GEMINI_API_KEY")
        )
    else:
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model="mixtral-8x7b-32768",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=os.getenv("GROQ_API_KEY")
        )
    return llm

# Streamlit UI
st.title("Generative Model Prompt Generator")

# Dropdown for model selection
model_selection = st.selectbox("Select Model", ["Gemini", "Groq"])

# Text area to input life query or text
input_text = st.text_area("Enter your query or text here:")

# Button to generate prompt
if st.button("Generate Prompt"):
    if input_text:
        try:
            # Load the selected model
            llm = load_model(model_selection)

            # Load and split document
            loader = PyPDFLoader("data/paper.pdf")
            text_splitter = CharacterTextSplitter(
                separator=".",
                chunk_size=500,
                chunk_overlap=50,
                length_function=len,
            )
            pages = loader.load_and_split(text_splitter)

            # Convert the chunks into documents
            documents = [Document(page_content=chunk.page_content) for chunk in pages]

            # Generate embeddings and store in Chroma
            embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.environ["GEMINI_API_KEY"])

            vectordb = Chroma.from_documents(documents,embedding_model)

            # Configure the retriever
            retriever = vectordb.as_retriever(search_kwargs={"k": 5})

            # Create the retrieval chain
            template = """
            You are a helpful AI assistant.
            Answer based on the context provided. 
            context: {context}
            input: {input}
            answer:
            """
            prompt = PromptTemplate.from_template(template)

            # Call the retriever and generate response
            results = retriever.get_relevant_documents(input_text)
            response = llm.predict(prompt.format(context=results, input=input_text))

            # Display the response
            st.write("Response:")
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.error("Please enter some text or a query to generate a prompt.")


import os
import streamlit as st
import pdfplumber
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import Chroma
from langchain.schema import Document

# Streamlit UI
st.title("PDF Query Engine using Gemini or Groq")

# Upload PDF file
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Select Model
model_choice = st.selectbox("Choose a Model", ["Gemini", "Groq"])

# Prompt input
prompt_input = st.text_input("Enter your query")

if uploaded_file is not None and prompt_input:
    # Save the uploaded PDF to a temporary location
    pdf_path = f"./{uploaded_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load the PDF using pdfplumber for table and text extraction
    with pdfplumber.open(pdf_path) as pdf:
        all_text = ""
        pages = []

        # Loop through each page in the PDF
        for page in pdf.pages:
            # Extract text
            text = page.extract_text()
            all_text += text + "\n"

            # Extract tables from each page
            tables = page.extract_tables()
            for table in tables:
                # Filter out rows containing None
                filtered_table = [row for row in table if None not in row]
                table_text = "\n".join(["\t".join(row) for row in filtered_table])
                all_text += "\n" + table_text + "\n"

            # Add page content to the list
            pages.append(text + "\n" + table_text)

    # Create text chunks
    text_splitter = CharacterTextSplitter(
        separator=".",
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_text(all_text)

    # Convert chunks into Document objects
    documents = [Document(page_content=chunk, metadata={}) for chunk in chunks]

    # Initialize models based on user selection
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
        # embeddings = GroqEmbeddings(  # Uncomment and replace with correct embeddings if available
        #     model="groq-embedding-1",  # Replace with the correct embedding model
        #     api_key=os.environ["GROQ_API_KEY"]
        # )

    # Generate embeddings and store them in Chroma
    vectordb = Chroma.from_documents(documents, embeddings)

    # Configure Chroma as a retriever with top_k=5
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    # Create the retrieval chain
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
    if prompt_input:
        response = retrieval_chain.invoke({"input": prompt_input})

        # Display the response
        st.subheader("Response:")
        st.write(response["answer"])  # Display only the LLM's response



