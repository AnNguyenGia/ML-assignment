"""Nạp dữ liệu BBC News và tiền xử lý văn bản."""
import re
from typing import Dict, List, Tuple

import pandas as pd

from .config import DATASET_NAME


def load_bbc(name: str = DATASET_NAME) -> Dict[str, pd.DataFrame]:
    """Tải dataset từ Hugging Face Hub và trả về dict {split: DataFrame}."""
    from datasets import load_dataset

    dataset = load_dataset(name)
    return {split: dataset[split].to_pandas() for split in dataset}


def clean_text(text) -> str:
    """Chuẩn hóa văn bản: lower, bỏ URL/HTML, chỉ giữ chữ cái, gọn khoảng trắng."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)  # URL
    text = re.sub(r"<.*?>", " ", text)  # HTML
    text = re.sub(r"[^a-z\s]", " ", text)  # chỉ giữ chữ cái và khoảng trắng
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_data(
    dfs: Dict[str, pd.DataFrame], remove_cross_split_overlap: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Làm sạch văn bản, loại bài trùng lặp và trả về (train_df, test_df, info).

    - Thêm cột ``clean_text``.
    - Loại bài trùng nhau (theo ``clean_text``) trong từng split.
    - Nếu ``remove_cross_split_overlap``: loại khỏi test các bài đã xuất hiện trong
      train để tránh rò rỉ dữ liệu (data leakage).
    - ``reset_index`` để các mảng đặc trưng luôn khớp thứ tự với DataFrame.
    """
    train_df = dfs["train"].copy()
    test_df = dfs["test"].copy()

    for df in (train_df, test_df):
        df["clean_text"] = df["text"].apply(clean_text)

    info = {
        "train_raw": len(train_df),
        "test_raw": len(test_df),
    }

    train_df = train_df.drop_duplicates(subset="clean_text").reset_index(drop=True)
    test_df = test_df.drop_duplicates(subset="clean_text").reset_index(drop=True)
    info["train_after_dedup"] = len(train_df)
    info["test_after_dedup"] = len(test_df)

    overlap_mask = test_df["clean_text"].isin(set(train_df["clean_text"]))
    info["train_test_overlap"] = int(overlap_mask.sum())
    if remove_cross_split_overlap:
        test_df = test_df[~overlap_mask].reset_index(drop=True)

    info["train_final"] = len(train_df)
    info["test_final"] = len(test_df)
    return train_df, test_df, info


def get_label_names(df: pd.DataFrame) -> List[str]:
    """Tên nhãn theo thứ tự id nhãn (dùng ``label_text`` nếu có)."""
    if "label_text" in df.columns:
        mapping = df.drop_duplicates("label").set_index("label")["label_text"].to_dict()
        return [str(mapping[k]) for k in sorted(mapping)]
    return [str(k) for k in sorted(df["label"].unique())]
