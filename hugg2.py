import os
import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
load_dotenv()
# Get the Hugging Face Hub API token from environment variable
api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# Function to set up the model and chain
def setup_model(api_token):
    model_id = 'gpt2-medium'
    conv_model = HuggingFaceHub(huggingfacehub_api_token=api_token,
                                     repo_id=model_id,
                                     model_kwargs={'temperature': 0.8, "max_new_tokens": 200})
    template = """You are a helpful AI assistant that makes stories by completing the query provided by the user
    {query}
    """
    prompt = PromptTemplate(template=template, input_variables=['query'])
    conv_chain = LLMChain(llm=conv_model, prompt=prompt, verbose=True)
    return conv_chain

st.title("AI Storytelling Assistant")
st.write("Ask the AI to complete your story by providing a query.")


if api_token:
    conv_chain = setup_model(api_token)

    # Input for the user query
    user_query = st.text_input("Enter the beginning of your story:")
    
    if st.button("Complete Story"):
        if user_query:
            # Run the model to complete the story
            result = conv_chain.run(user_query)
            st.write("## Completed Story:")
            st.write(result)
        else:
            st.write("Please enter a query to get a story completion.")
else:
    st.write("API Token not found. Please set the HUGGINGFACEHUB_API_TOKEN environment variable.")
