"""Trích xuất đặc trưng (TF-IDF, BERT) và lưu/nạp đặc trưng ở dạng .npy / .h5."""
from pathlib import Path
from typing import Sequence

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import (
    BERT_BATCH_SIZE,
    BERT_MAX_LENGTH,
    BERT_MODEL_NAME,
    BERT_POOLING,
    DEVICE,
    TFIDF_PARAMS,
)


# ----------------------------------------------------------------------------
# Lưu / nạp
# ----------------------------------------------------------------------------
def save_sparse_h5(path, X) -> None:
    """Lưu ma trận thưa (CSR) vào file .h5 (nén gzip)."""
    import h5py

    X = sp.csr_matrix(X, dtype=np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=X.data, compression="gzip")
        f.create_dataset("indices", data=X.indices, compression="gzip")
        f.create_dataset("indptr", data=X.indptr, compression="gzip")
        f.attrs["shape"] = X.shape


def load_sparse_h5(path) -> sp.csr_matrix:
    import h5py

    with h5py.File(path, "r") as f:
        shape = tuple(f.attrs["shape"])
        return sp.csr_matrix((f["data"][:], f["indices"][:], f["indptr"][:]), shape=shape)


def save_features(X, path) -> Path:
    """Lưu đặc trưng: ma trận thưa -> .h5, mảng dense -> .npy."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if sp.issparse(X):
        save_sparse_h5(path.with_suffix(".h5"), X)
        return path.with_suffix(".h5")
    np.save(path.with_suffix(".npy"), np.asarray(X, dtype=np.float32))
    return path.with_suffix(".npy")


def load_features(path):
    path = Path(path)
    if path.suffix == ".h5":
        return load_sparse_h5(path)
    return np.load(path)


# ----------------------------------------------------------------------------
# Method 1: TF-IDF
# ----------------------------------------------------------------------------
def build_tfidf(train_texts: Sequence[str], test_texts: Sequence[str], **params):
    """Fit TF-IDF trên train, transform cả train và test.

    Trả về (X_train, X_test, vectorizer); X_* là ma trận thưa scipy.
    """
    cfg = {**TFIDF_PARAMS, **params}
    vectorizer = TfidfVectorizer(**cfg)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer


# ----------------------------------------------------------------------------
# Method 2: BERT embeddings
# ----------------------------------------------------------------------------
def load_bert(model_name: str = BERT_MODEL_NAME, device: str = DEVICE):
    """Tải tokenizer và mô hình BERT (chế độ eval)."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    return tokenizer, model


@torch.no_grad()
def extract_bert(
    texts: Sequence[str],
    tokenizer,
    model,
    device: str = DEVICE,
    max_length: int = BERT_MAX_LENGTH,
    batch_size: int = BERT_BATCH_SIZE,
    pooling: str = BERT_POOLING,
    show_progress: bool = True,
) -> np.ndarray:
    """Trích embedding của văn bản bằng BERT.

    pooling="cls": vector của token [CLS]; pooling="mean": trung bình các token (có mask).
    Tokenize theo từng batch nên tiết kiệm bộ nhớ.
    """
    if pooling not in ("cls", "mean"):
        raise ValueError("pooling phải là 'cls' hoặc 'mean'")

    texts = list(texts)
    starts = range(0, len(texts), batch_size)
    if show_progress:
        from tqdm.auto import tqdm

        starts = tqdm(starts, desc=f"BERT ({pooling})")

    model.eval()
    chunks = []
    for start in starts:
        enc = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        hidden = model(**enc).last_hidden_state
        if pooling == "cls":
            emb = hidden[:, 0, :]
        else:
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        chunks.append(emb.cpu().numpy())
    return np.vstack(chunks).astype(np.float32)


def bert_features_cached(
    texts: Sequence[str],
    path,
    model_name: str = BERT_MODEL_NAME,
    device: str = DEVICE,
    force: bool = False,
    **kwargs,
) -> np.ndarray:
    """Nạp embedding từ file .npy nếu có (và đúng số mẫu), ngược lại trích xuất và lưu lại."""
    path = Path(path)
    if path.exists() and not force:
        arr = np.load(path)
        if arr.shape[0] == len(texts):
            print(f"[cache] Nạp {path.name}: {arr.shape}")
            return arr
        print(f"[cache] {path.name} có {arr.shape[0]} mẫu != {len(texts)} -> trích xuất lại")

    tokenizer, model = load_bert(model_name, device)
    arr = extract_bert(texts, tokenizer, model, device=device, **kwargs)
    save_features(arr, path)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return arr
