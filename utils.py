from pymilvus import MilvusClient
from openai import AzureOpenAI
from typing import List


def get_context(query_vector: list[float], collection_name: str, client: MilvusClient) -> str:
    search_res = client.search(
        collection_name=collection_name,
        data=[query_vector],
        limit=3,
        search_params={'metric_type': 'L2', 'params': {}},
        output_fields=['text']
    )

    return '\n'.join(res['entity']['text'] for res in search_res[0])


def get_prompt(question: str, context: str) -> str:
    return f"""
        Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
        <context>
        {context}
        </context>
        <question>
        {question}
        </question>
    """


def generate_embeddings(text: str, client: AzureOpenAI, model: str = "text-embedding-ada-002") -> List[float]:
    return client.embeddings.create(input=[text], model=model).data[0].embedding
