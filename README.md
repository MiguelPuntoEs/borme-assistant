# BORME Assistant

Commercial Registry information from [The Official Gazette of the Commercial Registry](https://www.boe.es/diario_borme/ayuda.php?lang=en) (English).

[Open data API](https://www.boe.es/datosabiertos/api/api.php) from Official State Gazette Agency (Spanish).

[API for access to the summaries of BORME](https://www.boe.es/datosabiertos/documentos/APIsumarioBORME.pdf) (Spanish).

## Vector database

`main.ipynb`: This notebook explores the information available at the BORME API, and loads today's data into a Milvus vector database through embedding PDF documents corresponding to the "Actos inscritos" section with `text-embedding-ada-002` model.

## Streamlit app

`streamlit-app.py`: RAG-based streamlit web chat application to answer questions from today's Commercial Registry Gazette using GPT-4o model.

## LangGraph example

`langgraph.ipynb`: this LangGraph example shows the capabilities of LLMs deciding on using custom-made tools for specific purposes. In this particular example, the LLM decides whether to call the retrieval function to process BORME information or not, and if so, generating an answer based on the retrieved information from the vector database using GPT-4o model
