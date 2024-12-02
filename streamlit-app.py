import streamlit as st
from pymilvus import MilvusClient
from utils import get_prompt, get_context
from openai import AzureOpenAI
from utils import generate_embeddings
from dotenv import load_dotenv


load_dotenv()


SYSTEM_PROMPT = """
Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
"""

DB_URI = './milvus.db'
COLLECTION_NAME = 'borme'

client = AzureOpenAI(
    api_version='2024-02-01'
)

milvus_client = MilvusClient(uri=DB_URI)


if "messages" not in st.session_state:
    st.session_state['messages'] = []

for message in st.session_state['messages']:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Qué quieres saber?"):

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        st.session_state['messages'].append(
            {"role": "user", "content": question})

        context = get_context(generate_embeddings(
            question, client), COLLECTION_NAME, milvus_client)
        prompt = get_prompt(question, context)

        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )

        response = st.write_stream(stream)

    st.session_state['messages'].append(
        {"role": "assistant", "content": response})
