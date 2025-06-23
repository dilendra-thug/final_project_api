from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
import os

# --- Initialize FastAPI ---
app = FastAPI(title="Agriculture Chatbot API")

# --- Hardcoded Config ---
GOOGLE_API_KEY = "AIzaSyBK5gc2fbQAOBP218EAplCHdssNf7C3hm8"
DB_FAISS_PATH = os.path.join(os.path.dirname(__file__), "vectorstore", "db_faiss")

# --- Prompt Template ---
CUSTOM_PROMPT_TEMPLATE = """
You are the best personal tutor.

If the question is about your identity, such as "Who created you?", "Who invented you?", "Who is your owner?", or similar questions related to your origin or developer, answer with:
"I am a chatbot developed by SOEIntel."
Repace  the term gemini by SOEIntel for this types of questions

Use only the information provided in the context to answer all other questions. If the answer is not found in the context, respond with: "The information is not available in the provided context."

Use **paragraph format** for definitions or general explanations.  
Use **bullet points** only when listing steps, advantages, roles, features, or other multi-point responses.

Do not speculate.

If the user explicitly requests the answer in Nepali (e.g., says "explain in Nepali", "in Nepali", "answer in Nepali" or similar), then:
- If the context is in English, translate your answer into Nepali before responding.
- If the context is already in Nepali, answer in Nepali directly.

Otherwise, respond in English by default.

Context:
{context}

User Question:
{question}

Answer:
"""

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# --- Setup LLM, Embeddings, Vector DB, and Memory ---
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY
)

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# Load FAISS vectorstore
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Setup memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Setup QA chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={'k': 7}),
    memory=memory,
    combine_docs_chain_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# --- Request Schema ---
class QueryRequest(BaseModel):
    question: str

# --- POST Endpoint ---
@app.post("/ask")
async def ask_question(query: QueryRequest):
    try:
        response = qa_chain.invoke({"question": query.question})
        return {"answer": response["answer"]}
    except Exception as e:
        return {"error": f"Failed to process query: {str(e)}"}
