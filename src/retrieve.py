from dataclasses import dataclass

from src.config import load_config
from src.embed import embed_query
from src.vectorstore import get_collection

@dataclass
class SearchResult:
    review_id: str
    document: str
    rating: int
    country: str
    review_date:str
    distance: float

def search(
        query:str,
        k: int | None = None,
        where: dict | None = None,
        config: dict | None = None,
        collection=None, 
) -> list[SearchResult]:
    cfg = config or load_config()
    k = k or cfg["retrieval"]["top_k"]
    collection = collection if collection is not None else get_collection(cfg)

    res = collection.query(
        query_embeddings=[embed_query(query,cfg)],
        n_results=k,
        where=where
    )
    return[
        SearchResult(
            review_id=rid,
            document=doc,
            rating=meta["rating"],
            country=meta["country"],
            review_date=meta["review_date"],
            distance=dist,
        )
        for rid,doc,meta,dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]

def pretty_print(results: list[SearchResult], width: int = 200) -> None:
    for r in results:
        print(f"[{r.rating}* {r.country} {r.review_date}] dist={r.distance:.3f} id={r.review_id}")
        print(f"  {r.document[:width]}{'...' if len(r.document) > width else ''}\n")