from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import box
import yaml
import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)


def load_embedding_model(model_name, normalize_embedding=True, device='cpu'):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs={
            'normalize_embeddings': normalize_embedding
        }
    )


def load_retriever(embeddings, store_path, collection_name, vector_space, num_results=1):
    vector_store = Chroma(collection_name=collection_name,
                          persist_directory=store_path,
                          collection_metadata={"hnsw:space": vector_space},
                          embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": num_results})

    return retriever

def build_rag_pipeline():
    # Import config vars
    with open('config.yml', 'r', encoding='utf8') as ymlfile:
        cfg = box.Box(yaml.safe_load(ymlfile))

    print("Loading embedding model...")
    embeddings = load_embedding_model(model_name=cfg.EMBEDDINGS,
                                      normalize_embedding=cfg.NORMALIZE_EMBEDDINGS,
                                      device=cfg.DEVICE)

    print("Loading vector store and retriever...")
    retriever = load_retriever(embeddings,
                               cfg.VECTOR_DB,
                               cfg.COLLECTION_NAME,
                               cfg.VECTOR_SPACE,
                               cfg.NUM_RESULTS)

    return retriever


