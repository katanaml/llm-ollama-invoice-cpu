# Invoice data processing LLM RAG on CPU with Ollama and ChromaDB


**Youtube**: <a href="https://www.youtube.com/watch?v=XuvdgCuydsM" target="_blank">Change Video</a>

___

## Quickstart

### RAG runs offline on local CPU
   
1. Install the requirements: 

```
pip install -r requirements.txt`
```

2. Copy text PDF files to the `data` folder.
3. Run the script, to convert text to vector embeddings and save in Chroma vector storage: 

```
python ingest.py
```

4. Run the script, to process data with LLM RAG and return the answer: 

```
python main.py "What is the invoice number value?"
```
