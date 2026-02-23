# import faiss
# import numpy as np
# import psycopg2
# import pandas as pd
# import re
# from sentence_transformers import SentenceTransformer, CrossEncoder
# from rapidfuzz import process, fuzz
# import pickle

# # -------------------------
# # Models
# # -------------------------

# embed_model = None
# reranker = None

# metric_names = None
# org_names = None

# metric_index = None
# org_index = None


# # -------------------------
# # Build FAISS Cache
# # -------------------------

# def load_rag_assets():
#     global embed_model, reranker
#     global metric_names, org_names
#     global metric_index, org_index

#     if metric_index is None:

#         metric_index = faiss.read_index("metric_index.faiss")
#         org_index = faiss.read_index("org_index.faiss")

#         with open("metric_names.pkl", "rb") as f:
#             metric_names = pickle.load(f)

#         with open("org_names.pkl", "rb") as f:
#             org_names = pickle.load(f)

#     if embed_model is None:
#         embed_model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

#     if reranker is None:
#         reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# def build_rag_cache(conn_str):

#     global metric_names, org_names
#     global metric_index, org_index

#     with psycopg2.connect(conn_str) as conn:
#         metric_df = pd.read_sql("SELECT DISTINCT name FROM dataelement", conn)
#         org_df = pd.read_sql("SELECT DISTINCT name FROM organisationunit", conn)

#     metric_names = metric_df["name"].tolist()
#     org_names = org_df["name"].tolist()

#     metric_vecs = embed_model.encode(metric_names, normalize_embeddings=True).astype("float32")
#     org_vecs = embed_model.encode(org_names, normalize_embeddings=True).astype("float32")

#     dim = metric_vecs.shape[1]

#     metric_index = faiss.IndexFlatIP(dim)
#     metric_index.add(metric_vecs)

#     org_index = faiss.IndexFlatIP(dim)
#     org_index.add(org_vecs)

#     print("FAISS ready:",
#           len(metric_names), "metrics |",
#           len(org_names), "orgs")


# # -------------------------
# # Step 1: Clean Query
# # -------------------------

# def clean_question(q):

#     q = q.lower()

#     q = re.sub(
#         r"\b(how many|show|display|what|cases|records|patients|related to)\b",
#         "",
#         q
#     )

#     q = re.sub(r"\b\d{4}\b", "", q)
#     q = re.sub(r"\s+", " ", q).strip()

#     return q


# # -------------------------
# # Step 2: Automatic Org Extraction
# # -------------------------

# def extract_org_mentions(query, limit=10):

#     matches = process.extract(
#         query,
#         org_names,
#         scorer=fuzz.partial_ratio,
#         limit=limit
#     )

#     return [m[0] for m in matches if m[1] > 75]


# # -------------------------
# # Step 3: Automatic Metric Keyword Extraction
# # -------------------------

# def extract_metric_mentions(query, limit=10):

#     query = query.lower()

#     # --- Step 1: Tokenize query into words ---
#     tokens = re.findall(r"[a-z0-9]+", query)

#     mentions = set()

#     # --- Step 2: Handle short code tokens like TT3 ---
#     for t in tokens:
#         if len(t) <= 4:   # TT3, OPV, ANC1 etc.
#             for name in metric_names:
#                 if t.upper() in name.upper():
#                     mentions.add(name)

#     # --- Step 3: Normal fuzzy matching for longer text ---
#     matches = process.extract(
#         query,
#         metric_names,
#         scorer=fuzz.partial_ratio,
#         limit=limit
#     )

#     for m, score, _ in matches:
#         if score > 65:   # lower threshold
#             mentions.add(m)

#     return list(mentions)

# # -------------------------
# # FAISS Candidate Retrieval
# # -------------------------

# def faiss_candidates(query, index, names, k=20):

#     vec = embed_model.encode([query], normalize_embeddings=True).astype("float32")
#     scores, idx = index.search(vec, k)

#     return [(names[i], float(scores[0][j])) for j, i in enumerate(idx[0])]


# # -------------------------
# # CrossEncoder Reranking
# # -------------------------

# def rerank(full_query, candidates, top_k=5):

#     if not candidates:
#         return []

#     pairs = [(full_query, text) for text, _ in candidates]
#     scores = reranker.predict(pairs)

#     reranked = list(zip([text for text, _ in candidates], scores))
#     reranked.sort(key=lambda x: x[1], reverse=True)

#     # Return ONLY clean names
#     return [text for text, _ in reranked[:top_k]]


# # -------------------------
# # Final Automatic Multi-Signal Retrieval
# # -------------------------

# def get_rag_context(question):
#     load_rag_assets()
#     query = clean_question(question)

#     org_mentions = extract_org_mentions(query)
#     metric_mentions = extract_metric_mentions(query)

#     metric_pool = []
#     org_pool = []

#     # ---- Metrics Retrieval ----
#     if not metric_mentions:
#         metric_pool += faiss_candidates(query, metric_index, metric_names, k=20)
#     else:
#         for m in metric_mentions:
#             metric_pool += faiss_candidates(m, metric_index, metric_names, k=20)

#     # ---- Org Retrieval ----
#     if not org_mentions:
#         org_pool += faiss_candidates(query, org_index, org_names, k=20)
#     else:
#         for o in org_mentions:
#             org_pool += faiss_candidates(o, org_index, org_names, k=20)

#     # Remove duplicates
#     metric_pool = list({x[0]: x for x in metric_pool}.values())
#     org_pool = list({x[0]: x for x in org_pool}.values())

#     # Rerank
#     metric_best = rerank(question, metric_pool, top_k=10)
#     org_best = rerank(question, org_pool, top_k=10)

#     return metric_best[:5], org_best[:5]

