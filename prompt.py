from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Create prompt from the template
# with system message, chat history, user question
prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a helpful assistant answering questions from a document.

Use ONLY the provided context. Do not make speculative claims or financial predictions.

If the context does not contain the answer say:
"I could not find this information in the document."

Context:
{context}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])