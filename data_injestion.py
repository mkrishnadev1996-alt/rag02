import fitz
# from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
# from http_client import client, async_client


load_dotenv()

# extract text from pdf file. Uses PyMuPDF / fitz
def get_text(file)->str:
    '''
    Extract text from PDF file stream
    Args:
        file: Uploaded PDF file
    Returns:
        str: Extracted text from PDF
    '''
    doc = fitz.open(stream=file.read(), filetype="pdf")
    texts = []
    for page in doc:
        text = page.get_text("text")
        # If text is present , append to list
        # to avoid empty pages
        if text:
            texts.append(text)
    return "\n".join(texts)

# Get Hugging Face embeddings - Local inference (faster, no API calls)
def get_embeddings()->HuggingFaceEmbeddings:
    '''
    Get Hugging Face embeddings for the text chunks using local model
    Returns:
        HuggingFaceEmbeddings: Embeddings object
    '''
    return HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'batch_size': 32}
    )

# API-based embeddings (commented out - slower due to network calls)
# def get_embeddings()->HuggingFaceEndpointEmbeddings:
#     return HuggingFaceEndpointEmbeddings(
#         repo_id='sentence-transformers/all-MiniLM-L6-v2',
#         task="feature-extraction",
#         huggingfacehub_api_token=os.getenv('HF_TOKEN'),
#         client=client,
#         async_client=async_client
#     )

# Get text splitter for the document
def get_text_splitter():
    '''
    Get text splitter for the document
    Returns:
        RecursiveCharacterTextSplitter: Text splitter object
'''
    return RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "."],
        chunk_size=800,
        chunk_overlap=150
    )

# Create vector database from text chunks
# def create_vecor_db(chunks,embeddings):
#     '''
#     Create vector database from text chunks
#     Args:
#         chunks (list): List of text chunks
#         embeddings: Embeddings object
#     Returns:
#         FAISS: Vector database object
#     '''
#     return FAISS.from_texts(texts=chunks, embedding=embeddings)

# Create vector database with progress tracking
def create_vecor_db_with_progress(chunks, embeddings, progress_bar):
    '''
    Create vector database from text chunks with progress bar
    Args:
        chunks (list): List of text chunks
        embeddings: Embeddings object
        progress_bar: Streamlit progress bar object
    Returns:
        FAISS: Vector database object
    '''
    batch_size = 32
    total_chunks = len(chunks)
    
    # Initialize with first batch
    first_batch = chunks[:batch_size]
    vector_db = FAISS.from_texts(texts=first_batch, embedding=embeddings)
    progress_bar.progress(min(batch_size / total_chunks, 1.0))
    
    # Process remaining chunks in batches
    for i in range(batch_size, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        vector_db.add_texts(batch)
        progress_bar.progress(min((i + batch_size) / total_chunks, 1.0))
    
    progress_bar.progress(1.0)
    return vector_db
