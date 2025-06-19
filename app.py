import streamlit as st
import os

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Hardcoded API key
os.environ['OPENAI_API_KEY'] = "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"  

# Initialize embeddings and LLM
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4")

# Streamlit UI
st.title("Netmark Business Services Handbook Q&A Bot")
session_id = st.text_input("Session ID", value="default_session")

# Initialize session state
if 'store' not in st.session_state:
    st.session_state.store = {}

# PDF path - this needs to be relative or in the same directory as your script
pdf_path = "Handbook1.pdf"  # Make sure this file is in your GitHub repo

# Check if file exists
if not os.path.exists(pdf_path):
    st.error(f"PDF file not found at: {pdf_path}")
    st.stop()

# Load PDF
try:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # Split and embed
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    
    # Define prompts
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question "
                   "which might reference context in the chat history, "
                   "formulate a standalone question which can be understood "
                   "without the chat history. Do NOT answer the question."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the following context to answer the question. "
                   "If the answer is unknown, say so. Keep it concise.\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    # Build chain
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=contextualize_q_prompt
    )

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    # User query input
    user_input = st.text_input("Ask a question:")
    if user_input:
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        st.write("**Assistant:**", response["answer"])
        
        # Display chat history
        st.write("**Chat History:**")
        session_history = get_session_history(session_id)
        for message in session_history.messages:
            st.write(f"{message.type}: {message.content}")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")