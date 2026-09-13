from functools import lru_cache

import numpy as np

from src.config import load_config

@lru_cache(maxsize=2)
def get_model(model_name:str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)

def embed_texts(texts: list[str],config: dict | None = None) -> np.ndarray:
    cfg = (config or load_config())["embedding"]
    model = get_model(cfg["model_name"])
    return model.encode(
        texts,
        batch_size=cfg["batch_size"],
        normalize_embeddings=cfg["normalize"],
        show_progress_bar=len(texts) > 100,
    )

def embed_query(query:str, config:dict | None = None) -> list[float]:
    return embed_texts([query],config)[0].tolist()