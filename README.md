# Invoice data processing LLM RAG on CPU with Ollama and ChromaDB


<a href="https://www.youtube.com/watch?v=Higmr8qMoNk" target="_blank">Easy-to-Follow RAG Pipeline Tutorial: Invoice Processing with ChromaDB & LangChain</a>

___

## Quickstart

### RAG runs offline on local CPU
   
1. Install the requirements: 

```
pip install -r requirements.txt`
```

Install Ollama and pull LLM model specified in config.yml

2. Copy text PDF files to the `data` folder.
3. Run the script, to convert text to vector embeddings and save in Chroma vector storage: 

```
python ingest.py
```

4. Run the script, to process data with LLM RAG and return the answer: 

```
python main.py "What is the invoice number value?"
```
