from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.llms import Ollama
import streamlit as st

## Loading Documents
loader = PyPDFLoader("data/IJEPA.pdf")
docs = loader.load()

## Performing Transform on Loaded Documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
documents = text_splitter.split_documents(docs)

## Creating Embeddings from Splitted Documents and Loading in to FAISS Vector Database
db = FAISS.from_documents(documents[:30],OllamaEmbeddings(model="llama3"))

## Design Chat Prompt Template
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context.
Think step by step before providing a detailed answer.
I will tip you $1000 if the user finds the answer helpful.
<context>
{context}
</context>
Question:{input}""")

## Load Ollama llama3 Model
llm = Ollama(model="llama3")

## Create Stuff Documents Chain
document_chain = create_stuff_documents_chain(llm,prompt)
retriever = db.as_retriever()

## Retriver chain
from langchain.chains import create_retrieval_chain
retrievar_chain = create_retrieval_chain(retriever,document_chain)

prompt = st.text_input("Do Search here")

if prompt:
    respose = retrievar_chain.invoke({"input":prompt})
    st.write(respose['answer'])
