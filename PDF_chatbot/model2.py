import streamlit as st
import os  # For environment variable access
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA

# Set DB_FAISS_PATH from environment variable (replace with actual variable name)
# DB_FAISS_PATH = os.getenv("MEDICAL_BOT_FAISS_PATH")
DB_FAISS_PATH = 'vectorstore/db_faiss'


# Function to load HuggingFace Embeddings (cached)
@st.cache_resource
def load_embeddings():
    # return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    
    return HuggingFaceEmbeddings(model_name="TheBloke/Llama-2-7B-Chat-GGML", model_kwargs={'device': 'cpu'})


# Function to load LLM (Llama) model (cached)
# @st.cache_resource
def load_llm():
    return CTransformers(model="llama-2-7b-chat.ggmlv3.q8_0.bin", model_type="llama", max_new_tokens=512, temperature=0.5)

# Function to define the custom prompt template (cached)
# @st.cache_resource
def set_custom_prompt():
    custom_prompt_template = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
    return PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

# Function to build the RetrievalQA Chain (cached)
# @st.cache_resource
def build_qa_chain(embeddings, llm):
    if not DB_FAISS_PATH:
        st.error("Please set the 'MEDICAL_BOT_FAISS_PATH' environment variable to point to your FAISS vector store.")
        return None  # Indicate error

    try:
        # db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        db = FAISS.load_local(DB_FAISS_PATH, embeddings)
        qa_prompt = set_custom_prompt()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=db.as_retriever(search_kwargs={'k': 2}),
                                               return_source_documents=True, chain_type_kwargs={'prompt': qa_prompt})
        return qa_chain
    except Exception as e:
        st.error(f"Error building QA Chain: {e}")
        return None

# Function to process user query
def process_query(query, qa):
    if not qa:
        return "Error: Could not build QA chain."

    response = qa({'query': query})
    answer = response["result"]
    sources = response.get("source_documents", [])
    if sources:
        answer += f"\nSources: {sources}"
    else:
        answer += "\nNo sources found"
    return answer

def main():
    """Main function for the Streamlit app"""

    st.title("Medical Bot")

    # Load model and prompt template (cached)
    embeddings = load_embeddings()
    llm = load_llm()
    qa_prompt = set_custom_prompt()

    # Build QA Chain (cached)
    qa = build_qa_chain(embeddings, llm)

    # User input field
    user_query = st.text_input("Ask your medical question here:")

    # Process query if user enters text
    if user_query:
        answer = process_query(query=user_query, qa=qa)
        st.write(answer)

if __name__ == '__main__':
    main()
