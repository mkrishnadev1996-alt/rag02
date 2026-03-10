import streamlit as st
import httpx
from operator import itemgetter
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from chat_history_manager import TokenLimitedChatHistory
from guardrails import validate_input, validate_output
from data_injestion import get_text_splitter,get_embeddings,get_text,create_vecor_db_with_progress
from prompt import prompt

# Load env variables
load_dotenv()


# Streamlit UI 
st.set_page_config(page_title="RAG AI Chatbot", layout="wide")
st.title("📚 RAG AI Chatbot with Citations")

# Upload pdf file
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "store" not in st.session_state:
    st.session_state.store = {}
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# Initialize LLM
llm = ChatOpenAI(
    model=os.getenv('HF_MODEL'),
    http_client=httpx.Client(verify=False),
    base_url=os.getenv('HF_URL'),
    api_key=os.getenv('HF_TOKEN')
)

# Get chat history from session
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    '''
    Get chat history for a session
    Args:
        session_id (str): The session ID
    Returns:
        BaseChatMessageHistory: The chat history for the session
'''
    # If session is not present in store, get chat history with token limit of 2000
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = TokenLimitedChatHistory(max_tokens=2000)
    return st.session_state.store[session_id]

# Process the uploaded PDF file
# Extract text - > Chunk -> Embedding -> Vector store DB
if st.button("Process Document") :
    if uploaded_file:
        try:
            with st.spinner("Extracting text...", show_time=True):
                text = get_text(uploaded_file)
            
            with st.spinner("Chunking document...", show_time=True):
                splitter = get_text_splitter()
                chunks = splitter.split_text(text)
            
            st.write(f"Creating embeddings for {len(chunks)} chunks...")
            progress_bar = st.progress(0)
            embeddings = get_embeddings()
            st.session_state.vector_db = create_vecor_db_with_progress(chunks, embeddings, progress_bar)
            
            st.success("Document processed successfully!")
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
    else:
        st.warning("Please upload a PDF file first.")

# After creation of vector store and storing it in session
# Create retriever and chain and call llm
if st.session_state.vector_db:
    retriever = st.session_state.vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # Simple formatting of list retrieved from vector db for similarity search
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Creating the chain with retriever, prompt, llm and output parser
    chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
            "history": itemgetter("history")
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # Creating chain with the chat history (max 2000 token)
    chain_with_history = RunnableWithMessageHistory(
        runnable=chain,
        get_session_history=get_session_history,
        input_messages_key="question",
        history_messages_key="history"
    )

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            # Displaye the sources used for the response
            with st.expander("📄 View Sources"):
                for i, src in enumerate(msg["sources"], 1):
                    st.text_area(f"Source {i}", src, height=100, disabled=True, key=f"history_src_{st.session_state.messages.index(msg)}_{i}")

    # After getting question from the user
    # Validate input -> Invoke chain with history -> 
    # Validate output -> Display response and sources
    if question := st.chat_input("Ask about the document"):
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        with st.spinner("Thinking...", show_time=True):
            try:
                # validate user input with guardrails. 
                # ValueError is raised if validation is fail
                validate_input(question)
                
                docs = retriever.invoke(question)
                sources = [doc.page_content for doc in docs]
                
                # invoke the RAG chain with chat history 
                response = chain_with_history.invoke(
                    {"question": question},
                    config={"configurable": {"session_id": "chat1"}}
                )
                # validate llm output with guardrails. 
                # ValueError is raised if validation is fail
                final_response = validate_output(response)
                
                # Add response and sources to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": final_response,
                    "sources": sources
                })

                # Display the response and sources
                st.chat_message("assistant").write(final_response)
                with st.expander("📄 View Sources"):
                    for i, src in enumerate(sources, 1):
                        st.text_area(f"Source {i}", src, height=100, disabled=True, key=f"current_src_{i}")

            # Guardrails validation exception    
            except ValueError as e:
                error_msg = f"⚠️ Guardrail violation: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.chat_message("assistant").write(error_msg)

            # Rest all exception
            except Exception as e:
                error_msg = f"⚠️ Exception occurred: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.chat_message("assistant").write(error_msg)
