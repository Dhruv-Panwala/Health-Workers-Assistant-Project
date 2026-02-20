import faiss
import numpy as np
import psycopg2
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer

conn_str = "postgresql://postgres:sierra-leone1%409@3.249.169.238:5432/sierra_leone_db"

model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

with psycopg2.connect(conn_str) as conn:
    metric_df = pd.read_sql("SELECT DISTINCT name FROM dataelement", conn)
    org_df = pd.read_sql("SELECT DISTINCT name FROM organisationunit", conn)

metric_names = metric_df["name"].tolist()
org_names = org_df["name"].tolist()

metric_vecs = model.encode(metric_names, normalize_embeddings=True).astype("float32")
org_vecs = model.encode(org_names, normalize_embeddings=True).astype("float32")

dim = metric_vecs.shape[1]

metric_index = faiss.IndexFlatIP(dim)
metric_index.add(metric_vecs)

org_index = faiss.IndexFlatIP(dim)
org_index.add(org_vecs)

faiss.write_index(metric_index, "metric_index.faiss")
faiss.write_index(org_index, "org_index.faiss")

with open("metric_names.pkl", "wb") as f:
    pickle.dump(metric_names, f)

with open("org_names.pkl", "wb") as f:
    pickle.dump(org_names, f)

print("Offline FAISS build complete.")
