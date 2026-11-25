"""Các hàm EDA. Hàm trả về DataFrame để notebook tự ``display``; hàm vẽ gọi ``plt.show()``."""
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

Dfs = Dict[str, pd.DataFrame]


def _label_col(df: pd.DataFrame) -> str:
    return "label_text" if "label_text" in df.columns else "label"


def add_text_stats(dfs: Dfs) -> Dfs:
    """Thêm cột char_count, word_count, sentence_count vào từng DataFrame (in-place)."""
    for df in dfs.values():
        text = df["text"].astype(str)
        df["char_count"] = text.str.len()
        df["word_count"] = text.str.split().str.len()
        df["sentence_count"] = text.str.count(r"[.!?]")
    return dfs


def check_quality(dfs: Dfs) -> pd.DataFrame:
    """Số dòng, giá trị thiếu và số bài trùng lặp của từng split."""
    rows = []
    for split, df in dfs.items():
        rows.append(
            {
                "split": split,
                "rows": len(df),
                "missing_values": int(df.isnull().sum().sum()),
                "duplicate_texts": int(df.duplicated(subset="text").sum()),
            }
        )
    return pd.DataFrame(rows)


def label_distribution(dfs: Dfs) -> pd.DataFrame:
    """Số lượng và tỉ lệ từng nhãn theo split."""
    parts = []
    for split, df in dfs.items():
        col = _label_col(df)
        counts = df[col].value_counts().sort_index()
        part = pd.DataFrame(
            {
                "split": split,
                "label": counts.index,
                "count": counts.values,
                "proportion": (counts / counts.sum()).round(4).values,
            }
        )
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def plot_label_distribution(dfs: Dfs) -> None:
    fig, axes = plt.subplots(1, len(dfs), figsize=(6 * len(dfs), 4), squeeze=False)
    for ax, (split, df) in zip(axes[0], dfs.items()):
        col = _label_col(df)
        counts = df[col].value_counts().sort_index()
        counts.plot(kind="bar", ax=ax, color="steelblue")
        ax.set_title(f"Class Distribution - {split}")
        ax.set_xlabel("Label")
        ax.set_ylabel("Number of Articles")
        ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.show()


def describe_text_stats(dfs: Dfs) -> pd.DataFrame:
    """Thống kê mô tả của char/word/sentence count theo split."""
    parts = {}
    for split, df in dfs.items():
        parts[split] = df[["char_count", "word_count", "sentence_count"]].describe().round(2)
    return pd.concat(parts, axis=1)


def plot_word_count_hist(dfs: Dfs, bins: int = 50) -> None:
    fig, axes = plt.subplots(1, len(dfs), figsize=(6 * len(dfs), 4), squeeze=False)
    for ax, (split, df) in zip(axes[0], dfs.items()):
        ax.hist(df["word_count"], bins=bins, color="steelblue")
        ax.set_title(f"Article Word Count - {split}")
        ax.set_xlabel("Number of Words")
        ax.set_ylabel("Number of Articles")
    plt.tight_layout()
    plt.show()


def word_count_by_class(dfs: Dfs) -> pd.DataFrame:
    """Thống kê độ dài bài viết (số từ) theo từng nhãn và split."""
    parts = []
    for split, df in dfs.items():
        col = _label_col(df)
        stat = df.groupby(col)["word_count"].agg(["count", "mean", "median", "min", "max", "std"])
        stat.insert(0, "split", split)
        parts.append(stat.round(2))
    return pd.concat(parts)


def plot_word_count_boxplot(dfs: Dfs) -> None:
    fig, axes = plt.subplots(1, len(dfs), figsize=(6 * len(dfs), 4), squeeze=False)
    for ax, (split, df) in zip(axes[0], dfs.items()):
        sns.boxplot(data=df, x=_label_col(df), y="word_count", ax=ax)
        ax.set_title(f"Word Count by Class - {split}")
        ax.set_xlabel("Label")
        ax.set_ylabel("Word Count")
    plt.tight_layout()
    plt.show()


def show_examples(dfs: Dfs, n: int = 1, max_chars: int = 300, split: str = "train") -> None:
    """In ``n`` bài mẫu cho mỗi nhãn của một split."""
    df = dfs[split]
    col = _label_col(df)
    for label in sorted(df[col].unique()):
        print(f"\n--- Label: {label} ---")
        for i, text in enumerate(df[df[col] == label]["text"].head(n), 1):
            print(f"[{i}] {text[:max_chars]} ...")


def short_articles(df: pd.DataFrame, threshold: int = 20, n: int = 10) -> pd.DataFrame:
    return df[df["word_count"] < threshold][["text", "label", "word_count"]].head(n)


def longest_articles(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return df.nlargest(n, "word_count")[["text", "label", "word_count"]]


def summary_table(dfs: Dfs) -> pd.DataFrame:
    """Bảng tổng hợp cho báo cáo."""
    rows = []
    for split, df in dfs.items():
        rows.append(
            {
                "split": split,
                "rows": len(df),
                "columns": len(df.columns),
                "missing_values": int(df.isnull().sum().sum()),
                "duplicates": int(df.duplicated(subset="text").sum()),
                "mean_words": round(df["word_count"].mean(), 1),
                "median_words": df["word_count"].median(),
                "min_words": df["word_count"].min(),
                "max_words": df["word_count"].max(),
                "num_classes": df["label"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def top_words_by_class(df: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """Các từ xuất hiện nhiều nhất của mỗi nhãn (đã bỏ stop words tiếng Anh)."""
    col = _label_col(df)
    text_col = "clean_text" if "clean_text" in df.columns else "text"
    result = {}
    for label in sorted(df[col].unique()):
        cv = CountVectorizer(stop_words="english", max_features=n)
        cv.fit(df[df[col] == label][text_col])
        counts = cv.transform(df[df[col] == label][text_col]).sum(axis=0).A1
        words = sorted(zip(cv.get_feature_names_out(), counts), key=lambda x: -x[1])
        result[label] = [w for w, _ in words]
    return pd.DataFrame(result)
