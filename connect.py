import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Setup environment
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")
model = "meta-llama/llama-4-scout-17b-16e-instruct"

# Load model
def load_model(model: str):
    model = ChatGroq(
        model=model,
        temperature=0.01,
        max_tokens=512,
        groq_api_key=groq_api_key
    )

    return model

model = load_model(model)

# Connect model with FAISS and Create chain
custom_prompt = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you don't know, dont try to make up an answer. 
Don't provide anything out of the given context

Context: {context}
Question: {question}

Answer directly. No small talk requires.
"""

def create_custom_prompt(custom_prompt):
    prompt = PromptTemplate.from_template(template=custom_prompt)
    return prompt

prompt = create_custom_prompt(custom_prompt)

# Load database
db_faiss_path = "store/db_faiss"

model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

db = FAISS.load_local(db_faiss_path, embeddings=embedding_model, allow_dangerous_deserialization=True)

# Create QA Chain
# chain_type: "map_reduce": xử lý từng tài liệu riêng, rồi tổng hợp kết quả, "refine": xử lý lần lượt, mỗi lần bổ sung câu trả lời, "stuff" nghĩa là lấy tất cả tài liệu tìm được, nhét nguyên vào prompt cho LLM.
# Trong trường hợp này dùng chain_type = "stuff"
qa_chain = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    # return_source_documents=True,
    chain_type_kwargs={'prompt': prompt}
)

# Invoke with a single query
user_query=input("Input your query here: ")
response=qa_chain.invoke({'query': user_query})
print("RESULT: ", response["result"])
# print("SOURCE DOCUMENTS: ", response["source_documents"])