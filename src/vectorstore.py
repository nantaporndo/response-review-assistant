import chromadb
import pandas as pd

from src.config import PROJECT_ROOT, load_config

_ADD_BATCH = 5000

def get_collection(config: dict | None = None, reset : bool=False):
    cfg = (config or load_config())["vectorstore"]
    client = chromadb.PersistentClient(path=str(PROJECT_ROOT/cfg["persist_path"]))
    if reset:
        try:
            client.delete_collection(cfg["collection_name"])

        except Exception:
            pass

    return client.get_or_create_collection(
        cfg["collection_name"],
        metadata={"hnsw:space": cfg["distance"]},
    )

def build_metadatas(df: pd.DataFrame) -> list[dict]:
    return[
        {"rating": int(r),"country":str(c),"review_date":str(d.date())}
        for r,c,d in zip(df["rating",df["country"],df["review_date"]])
    ]

def add_documents(collection,df:pd.DataFrame,embeddings) -> None:
    ids = df["review_id"].tolist()
    docs = df["document"].tolist()
    metas = build_metadatas(df)

    for i in range(0,len(ids),_ADD_BATCH):
        collection.add(
            ids=ids[i : i+_ADD_BATCH],
            embeddings=embeddings[i:i+_ADD_BATCH].tolist(),
            documents=docs[i:i+_ADD_BATCH],
            metadatas=metas[i:i+_ADD_BATCH]
        )
    print(f"vectorstore: {collection.count():,} documents indexed")