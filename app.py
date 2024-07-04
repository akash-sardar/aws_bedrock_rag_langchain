import json
import os
import sys
import boto3
import numpy as np
import streamlit as st

# We will use TITAN Embeddings model to generate Embedding

# Get embeddings
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock

# Data Ingestion
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector Embedding and Vector Store
from langchain.vectorstores import FAISS

# LLM Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
#-------------------------------------------------------------------------------#

#----------- create clients for aws --------------------------------------------#
# Bedrock Clients
bedrock = boto3.client(service_name = "bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id = "amazon.titan-embed-text-v1", client = bedrock)
#-------------------------------------------------------------------------------#

#----------- data ingestion layer   --------------------------------------------#
def data_ingestion_layer():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    docs = text_splitter.split_documents(documents)
    return docs
#-------------------------------------------------------------------------------#

#----------- Vector Store and Embeddings   --------------------------------------------#
def get_vector_store(docs):
    vector_store = FAISS.from_documents(docs, bedrock_embeddings)
    vector_store.save_local("data\\faiss_index")
#-------------------------------------------------------------------------------#

#----------- Create LLMs   --------------------------------------------#
def get_claude_llm():
    # create the Anthropic Model
    llm = Bedrock(model_id = "ai21.j2-mid-v1", 
                  client = bedrock,
                  model_kwargs = {'maxTokens': 512}
                  )
    return llm

def get_llama_llm():
    # create the Anthropic Model
    llm = Bedrock(model_id = "meta.llama2-70b-chat-v1", 
                  client = bedrock,
                  model_kwargs = {'max_gen_len': 512}
                  )
    return llm
#-------------------------------------------------------------------------------#
#----------- Prompt Template--------------------------------------------#
prompt_template = '''
Human: Use the following pieces of context to provide a concise answer to the question in 150 words. If the answer is unclear, just say that answer is unknown, do not generate answer.
<context>
{context}
</context>
Question: {question}
Assistant:
'''
prompt = PromptTemplate(
    template = prompt_template, input_variables=["context", "question"]
)
#-------------------------------------------------------------------------------#

#----------- get answer from llm --------------------------------------------#
def get_response_llm(llm, vector_store, query):
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = vector_store.as_retriever(
            search_type = "similarity",
            search_kwargs = {"k":3}
        ),
        return_source_documents = True,
        chain_type_kwargs = {"prompt": prompt}
    )
    answer = qa({"query" : query})
    return answer["result"]
#-------------------------------------------------------------------------------#

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using AWS Bedrock")

    user_question = st.text_input("Ask a question from color of magic")

    with st.sidebar:
        st.title("Update or Create vector Store:")
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion_layer()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Claude Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("data\\faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_claude_llm()
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")

    if st.button("Llama Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("data\\faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_llama_llm()
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")   

if __name__ == "__main__":
    main()
                   

