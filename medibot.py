import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")

db_faiss_path = "store/db_faiss"
embedding_model_name='sentence-transformers/all-MiniLM-L6-v2'
model_name = "meta-llama/llama-4-scout-17b-16e-instruct"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    db = FAISS.load_local(db_faiss_path, embedding_model, allow_dangerous_deserialization=True)
    return db


def create_custom_prompt(custom_prompt):
    prompt = PromptTemplate.from_template(template=custom_prompt)
    return prompt

def load_model(model_name):
    model = ChatGroq(
        model=model_name,
        temperature=0.01,
        max_tokens=512,
        groq_api_key=groq_api_key
    )
    return model

model = load_model(model_name)

def main():
    st.title("Retrieval Healthcare Knowledge")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Input your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT = """
            Use the pieces of information provided in the context to answer user's question.
            If you dont know the answer, just say that you don't know, don't try to make up an answer. 
            Don't provide anything out of the given context

            Context: {context}
            Question: {question}

            Answer directly. No small talk require.
        """

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain = RetrievalQA.from_chain_type(
                llm=model,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': create_custom_prompt(CUSTOM_PROMPT)}
            )

            response = qa_chain.invoke({'query': prompt})

            result = response["result"]
            # source_documents = response["source_documents"]
            # result_to_show = result + "\nSource Docs:\n" + str(source_documents)
            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({'role': 'assistant', 'content': result})

        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()