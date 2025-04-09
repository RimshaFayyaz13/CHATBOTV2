

# import langchain


import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


# In[6]:

os.environ["LANGCHAIN_TRACKING_V2"] = "true"
langchain_api_key = os.getenv("LANGCHAIN_API_KEY_KEY")
os.environ["LANGCHAIN_PROJECT"] = "langchain-project"

# In[7]:
import os

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
from langchain_core.output_parsers import StrOutputParser



from langchain_core.prompts import ChatPromptTemplate


# In[22]:


from langchain_community.document_loaders import PythonLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from typing import List
from langchain_core.documents import Document

langchain_text_splitters = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

docx_loader = Docx2txtLoader(r"C:\Users\Yasir\Desktop\chatbotV2\app\KB\KB.docx")
documents = docx_loader.load()

splits = langchain_text_splitters.split_documents(documents)
print(f"Split the document into {len(splits)} chunks")


from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
import os

def load_documents(folder_path: str) -> List[Document]:
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif filename.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        else:
            print(f"Unsupported file type: {filename}")
            continue
        documents.extend(loader.load())
    return documents

folder_path = r"C:\Users\Yasir\Desktop\chatbotV2\app\KB"
documents = load_documents(folder_path)
print(f"Loaded {len(documents)} documents from the folder.")


from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
document_embeddings = embeddings.embed_documents([split.page_content for split in splits])
print(f"Created embeddings for {len(document_embeddings)} document chunks.")


# print (document_embeddings)


from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


embedding_function = embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
document_embeddings = embedding_function.embed_documents([split.page_content for split in splits])
print(document_embeddings[0][:5])


# In[32]:


from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Define embedding function
embedding_function = OpenAIEmbeddings()

# Your list of document splits
splits = langchain_text_splitters.split_documents(documents)

# Define persistent directory
persist_directory = "./chroma_db"

# Create and persist the vectorstore
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=embedding_function,
    persist_directory=persist_directory
)

print(f"Vector store created and persisted to '{persist_directory}'")

# In[33]:


query = "What makes TechNova Solutions stand out from other tech consultancies?"
search_results = vectorstore.similarity_search(query, k=2)
print(f"\nTop 2 most relevant chunks for the query: '{query}'\n")
for i, result in enumerate(search_results, 1):
    print(f"Result {i}:")
    print(f"Source: {result.metadata.get('source', 'Unknown')}")
    print(f"Content: {result.page_content}")
    print()




retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
retriever_results = retriever.invoke("What makes TechNova Solutions stand out from other tech consultancies?")
# print(retriever_results)


from langchain_core.prompts import ChatPromptTemplate
template = """Answer the question based only on the following context:
{context}
Question: {question}
Answer: """

prompt = ChatPromptTemplate.from_template(template)

from langchain.schema.runnable import RunnablePassthrough

rag_chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt )
rag_chain.invoke("What makes TechNova Solutions stand out from other tech consultancies?")

def docs2str(docs):
    return "\n\n".join(doc.page_content for doc in docs)



rag_chain = (
    {"context": retriever | docs2str, "question": RunnablePassthrough()} | prompt
)
rag_chain.invoke("What makes TechNova Solutions stand out from other tech consultancies?")

rag_chain = (
    {"context": retriever | docs2str, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
question = "What makes TechNova Solutions stand out from other tech consultancies?"
response = rag_chain.invoke(question)
print(question)
print(response)


from langchain_core.messages import HumanMessage, AIMessage
chat_history = []
chat_history.extend([
    HumanMessage(content=question),
    AIMessage(content=response)
])

chat_history

from langchain_core.prompts import MessagesPlaceholder
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question"
    "Which might reference context in the chat history"
    "formulate a standalone question which can be understood"
    "without the chat history, Do not answer the question"
    "Just reformulate it if needed and otherwise return it as it"
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

contextualize_chain = contextualize_q_prompt | llm | StrOutputParser()
contextualize_chain.invoke({"input": "How has TechNova evolved since its founding in 2010?", "chat_history": chat_history})


# In[43]:


from langchain.chains import create_history_aware_retriever
history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_q_prompt
)
history_aware_retriever.invoke({"input": "How has TechNova evolved since its founding in 2010?", "chat_history": chat_history})

retriever.invoke("How has TechNova evolved since its founding in 2010?")

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use the following context to answer the user's question."),
    ("system", "Context: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

rag_chain.invoke({"input": "How has TechNova evolved since its founding in 2010?", "chat_history": chat_history})

import sqlite3
from datetime import datetime
import uuid

DB_NAME = "rag_app.db"

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def create_application_logs():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS application_logs
    (id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    user_query TEXT,
    gpt_response TEXT,
    model TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.close()

def insert_application_logs(session_id, user_query, gpt_response, model):
    conn = get_db_connection()
    conn.execute('INSERT INTO application_logs (session_id, user_query, gpt_response, model) VALUES (?, ?, ?, ?)',
                 (session_id, user_query, gpt_response, model))
    conn.commit()
    conn.close()

def get_chat_history(session_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT user_query, gpt_response FROM application_logs WHERE session_id = ? ORDER BY created_at', (session_id,))
    messages = []
    for row in cursor.fetchall():
        messages.extend([
            {"role": "human", "content": row['user_query']},
            {"role": "ai", "content": row['gpt_response']}
        ])
    conn.close()
    return messages

# Initialize the database
create_application_logs()

# Example usage for a new user
session_id = str(uuid.uuid4())
question = "How has TechNova evolved since its founding in 2010?"
chat_history = get_chat_history(session_id)
answer = rag_chain.invoke({"input": question, "chat_history": chat_history})['answer']
insert_application_logs(session_id, question, answer, "gpt-3.5-turbo")
print(f"Human: {question}")
print(f"AI: {answer}\n")



# In[49]:


# Example of a follow-up question
question2 = "What has been TechNova’s biggest milestone so far?"
chat_history = get_chat_history(session_id)
answer2 = rag_chain.invoke({"input": question2, "chat_history": chat_history})['answer']
insert_application_logs(session_id, question2, answer2, "gpt-3.5-turbo")
print(f"Human: {question2}")
print(f"AI: {answer2}")


# In[50]:


from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Initialize FastAPI app
app = FastAPI()

# Load the existing Chroma DB
persist_directory = "./chroma_db"
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings())

# Initialize LLM and QA chain
llm = OpenAI()
qa = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())

# Request model
class Question(BaseModel):
    query: str

@app.post("/ask")
def chatbot_function(question: Question):
    """Endpoint to get chatbot responses"""
    response = qa.invoke(question)
    return {"response": response}

# Run the API with: uvicorn main:app --reload