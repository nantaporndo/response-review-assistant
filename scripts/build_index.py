"""รัน pipeline เต็ม: raw CSV -> clean -> embed -> index
ใช้: uv run python scripts/build_index.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import ingest
from src.config import load_config
from src.embed import embed_texts
from src.vectorstore import add_documents, get_collection


def main():
    cfg = load_config()
    df = ingest.run(cfg)
    embeddings = embed_texts(df["document"].tolist(), cfg)
    collection = get_collection(cfg, reset=True)
    add_documents(collection, df, embeddings)
    print("done — ลอง query ได้เลย: uv run python scripts/query.py \"late delivery refund\"")


if __name__ == "__main__":
    main()