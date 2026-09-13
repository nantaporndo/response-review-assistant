from pathlib import Path
import pandas as pd
from src.config import PROJECT_ROOT, load_config

def make_document(title:str|float, text:str) -> str:
    title = title.strip() if isinstance(title,str) else ""
    text = text.strip()
    core = title.rstrip("...").rstrip(".").strip()
    if core and text.lower().startswith(core.lower()):
        return text
    return f"{title}. {text}".strip(". ").strip() if title else text

def load_raw(path:str | Path) -> pd.DataFrame:
    return pd.read_csv(path,engine="python")

def clean(df: pd.DataFrame, mix_text_length: int = 20) -> pd.DataFrame:
    df = df.dropna(subset=["Review Text"]).copy()
    df["rating"] = df["Rating"].str.extract(r"(\d)").astype(int)
    df["document"] = [
        make_document(t,x) for t,x in zip(df["Review Title"], df["Review Text"])
    ]
    df = df[df["document"].str.len() >= mix_text_length]
    df = df.drop_duplicates(subset="document")
    df["review_date"] = pd.to_datetime(df["Review Date"],errors="coerce",utc=True)
    df["country"] = df["Country"].fillna("unknown")
    out = df[["document","rating","review_date","country"]].reset_index(drop=True)
    out["review_id"]=out.index.astype(str)
    return out

def run(config:dict | None = None) -> pd.DataFrame:
    cfg = config or load_config()
    raw_path = PROJECT_ROOT / cfg["data"]["raw_path"]
    out_path = PROJECT_ROOT / cfg["data"]["processed_path"]

    df = clean(load_raw(raw_path),cfg["data"]["min_text_length"])

    out_path.parent.mkdir(parents=True,exist_ok=True)
    df.to_parquet(out_path,index=False)
    print(f"ingest:{len(df):,} documents -> {out_path}")
    return df