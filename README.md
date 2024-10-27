# PDF Question & Answer Application

A Streamlit application that enables users to ask questions about PDF documents using LangChain and Ollama LLMs. The application leverages PDF text extraction, vector embeddings, and retrieval-based question answering to provide accurate responses based on document content.

## Features

- PDF document loading and processing
- Text chunking with overlap for better context preservation
- Vector embeddings using Ollama
- Fast similarity search using FAISS vector database
- Interactive Q&A interface using Streamlit
- Retrieval-augmented generation for accurate answers

## Prerequisites

- Python 3.8+
- Ollama installed and running locally

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install required dependencies:
```bash
pip install langchain-community langchain streamlit faiss-cpu
```

3. Make sure Ollama is installed and running with the llama3 model:
```bash
ollama pull llama3
```

## Project Structure

```
.
├── data/
│   └── IJEPA.pdf    # Your PDF document
├── app.py           # Main application code
└── README.md        # This file
```

## Usage

1. Place your PDF file in the `data/` directory

2. Run the Streamlit application:
```bash
streamlit run app.py
```

3. Enter your questions in the search box to get answers based on the PDF content

## Code Explanation

### Document Loading and Processing
```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFLoader("data/IJEPA.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)
```

### Vector Store Creation
```python
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

db = FAISS.from_documents(documents[:30], OllamaEmbeddings(model="llama3"))
```

### Question-Answering Chain
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.llms import Ollama

prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context.
Think step by step before providing a detailed answer.
I will tip you $1000 if the user finds the answer helpful.
<context>
{context}
</context>
Question:{input}""")

llm = Ollama(model="llama3")
document_chain = create_stuff_documents_chain(llm, prompt)
```

### Retrieval Chain and User Interface
```python
from langchain.chains import create_retrieval_chain

retriever = db.as_retriever()
retrievar_chain = create_retrieval_chain(retriever, document_chain)

prompt = st.text_input("Do Search here")
if prompt:
    respose = retrievar_chain.invoke({"input":prompt})
    st.write(respose['answer'])
```

## Configuration

The application can be customized by modifying these parameters:

- `chunk_size`: Size of text chunks (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 200)
- `documents[:30]`: Number of documents to process (adjust as needed)

## Limitations

- Currently processes only the first 30 documents from the PDF
- Requires Ollama to be running locally
- Performance depends on the size of the PDF and available system resources

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
