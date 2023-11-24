from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import box
import yaml
import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)


# Import config vars
with open('config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def load_embedding_model(model_name, normalize_embedding=True):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': cfg.DEVICE},  # here we will run the model with CPU only
        encode_kwargs={
            'normalize_embeddings': normalize_embedding  # keep True to compute cosine similarity
        }
    )


def load_retriever(embeddings, store_path, num_results=1):
    vector_store = Chroma(collection_name=cfg.COLLECTION_NAME,
                          persist_directory=store_path,
                          collection_metadata={"hnsw:space": cfg.VECTOR_SPACE},
                          embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": num_results})

    return retriever

def build_rag_pipeline():
    print("Loading embedding model...")
    embeddings = load_embedding_model(model_name=cfg.EMBEDDINGS,
                                      normalize_embedding=cfg.NORMALIZE_EMBEDDINGS)

    print("Loading vector store and retriever...")
    retriever = load_retriever(embeddings, cfg.VECTOR_DB, cfg.NUM_RESULTS)

    return retriever


