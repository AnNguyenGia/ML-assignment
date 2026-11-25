"""Cấu hình toàn cục của dự án (đường dẫn, seed, tham số mô hình, các options)."""
from pathlib import Path
import random

import numpy as np
import torch

# ----------------------------------------------------------------------------
# Đường dẫn
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = PROJECT_ROOT / "features"
REPORTS_DIR = PROJECT_ROOT / "outputs"

# ----------------------------------------------------------------------------
# Thiết bị & seed
# ----------------------------------------------------------------------------
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = SEED) -> None:
    """Cố định seed để kết quả tái lập được."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# Dữ liệu (nguồn công khai)
# ----------------------------------------------------------------------------
DATASET_NAME = "SetFit/bbc-news"
DATASET_URL = "https://huggingface.co/datasets/SetFit/bbc-news"
DATASET_ORIGIN_URL = "https://www.kaggle.com/yufengdev/bbc-text-categorization/data"

# ----------------------------------------------------------------------------
# Trích xuất đặc trưng
# ----------------------------------------------------------------------------
TFIDF_PARAMS = dict(
    max_features=20000,
    min_df=2,
    max_df=0.95,
    ngram_range=(1, 2),
    sublinear_tf=True,
)

BERT_MODEL_NAME = "bert-base-uncased"
BERT_MAX_LENGTH = 256
BERT_BATCH_SIZE = 16
BERT_POOLING = "cls"  # "cls" hoặc "mean"

# ----------------------------------------------------------------------------
# Các lựa chọn thực nghiệm (dùng để in ra / ghi báo cáo)
# ----------------------------------------------------------------------------
DATA_PROCESSING_OPTIONS = {
    "METHOD_1": "TF-IDF Vectorizer (Thống kê tần suất từ)",
    "METHOD_2": "BERT Embeddings (Trích xuất CLS token từ mô hình Transformer)",
}

MODEL_OPTIONS = {
    "MODEL_1": "Naive Bayes (Gaussian, tự cài đặt)",
    "MODEL_2": "Decision Tree (Cây quyết định)",
    "MODEL_3": "MLP (Multi-Layer Perceptron - Neural Network)",
}

EVALUATION_METRICS = ["Accuracy", "Precision", "Recall", "F1-Score (Macro & Weighted)"]
