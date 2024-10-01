import streamlit as st
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
import tempfile
import os

# Initialize embeddings
embeddings = HuggingFaceEmbeddings()

# Initialize LLM with adjusted parameters
llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",  # Using a smaller model
    model_kwargs={
        "temperature": 0.7,
        "max_length": 512,
        "top_p": 0.95
    }
)

# Create a prompt template
prompt_template = PromptTemplate(
    input_variables=["question", "context"],
    template="""
    You are a helpful AI assistant. Using the following context, provide a detailed and informative answer to the question. If the context doesn't contain relevant information, use your general knowledge to give a comprehensive response.

    Context: {context}

    Question: {question}

    Detailed Answer:"""
)

# Initialize vector store
vector_store = None

def process_file(file):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as temp_file:
        temp_file.write(file.getvalue())
        temp_file_path = temp_file.name

    # Load the document
    loader = UnstructuredFileLoader(temp_file_path)
    documents = loader.load()
    
    # Remove the temporary file
    os.unlink(temp_file_path)
    
    return documents

def main():
    st.set_page_config(page_title="Document Chatbot")
    st.header("Document Chatbot with Custom Knowledge Base")

    # File uploader for multiple files
    uploaded_files = st.file_uploader("Choose files", type=["txt", "pdf"], accept_multiple_files=True)

    if uploaded_files:
        all_documents = []
        for file in uploaded_files:
            documents = process_file(file)
            all_documents.extend(documents)
        
        # Split the documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(all_documents)
        
        # Create the vector store
        global vector_store
        vector_store = FAISS.from_documents(texts, embeddings)
        
        st.success(f"{len(uploaded_files)} file(s) processed successfully!")

    # Initialize conversation memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Initialize the conversation chain
    if 'vector_store' in globals() and vector_store is not None:
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt_template}
        )

        # Chat interface
        user_question = st.text_input("Ask a question about your documents:")
        if user_question:
            response = conversation_chain({"question": user_question})
            st.write("Answer:", response['answer'])

if __name__ == "__main__":
    main()