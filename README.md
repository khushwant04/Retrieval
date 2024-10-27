# PDF Question-Answering App with LangChain, FAISS, and Streamlit

This project provides a question-answering application that loads a PDF document, processes and stores its text content in a vector database, and allows users to query the document via a Streamlit interface. It leverages LangChain's capabilities for document loading, text splitting, and vector storage and retrieval, making it efficient and effective for natural language querying of PDF content.

## Features

- **PDF Document Loading**: Loads and parses PDF files using LangChain's `PyPDFLoader`.
- **Text Splitting**: Splits document text into manageable chunks for processing using `RecursiveCharacterTextSplitter`.
- **Embeddings Creation**: Embeds text chunks using the `OllamaEmbeddings` model (`llama3`).
- **FAISS Vector Storage**: Stores embeddings in FAISS for efficient similarity-based retrieval.
- **Streamlit Interface**: Provides a user-friendly interface to enter queries and view answers based on the PDF content.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/pdf-question-answering-app.git
   cd pdf-question-answering-app
