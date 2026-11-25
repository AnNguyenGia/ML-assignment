# Bài tập lớn Machine Learning — Phân loại chủ đề tin tức BBC News

## Thông tin môn học
- **Môn học:** Học máy
- **Mã môn học:** CO3117
- **Học kỳ / Năm học:** Học kỳ HK261, năm học 2026-2027

## Giảng viên hướng dẫn
- Trương Vĩnh Lân

## Thành viên nhóm
| STT | Họ và tên | MSSV | Email |
|-----|-----------|------|-------|
| 1 | Nguyễn Gia An | 2410021 | an.nguyengia2006@hcmut.edu.vn |
| 2 | Nông Chí Thiện | 2414113 | thien.nongchi566@hcmut.edu.vn |



## Mục tiêu
Xây dựng và so sánh các mô hình học máy cho bài toán **phân loại chủ đề bài báo** (5 lớp: business,
entertainment, politics, sport, tech) trên tập dữ liệu BBC News:

- **Đặc trưng:** TF-IDF (unigram + bigram) và BERT embeddings (`bert-base-uncased`, vector `[CLS]`).
- **Mô hình:** Naive Bayes (Gaussian, tự cài đặt), Decision Tree, MLP (PyTorch).
- **Đánh giá:** Accuracy, Precision, Recall, F1 (macro & weighted), ma trận nhầm lẫn.

## Dữ liệu
- Hugging Face: https://huggingface.co/datasets/SetFit/bbc-news (2.225 bài báo, ~1.2K train / ~1K test)
- Nguồn gốc: https://www.kaggle.com/yufengdev/bbc-text-categorization/data
- Dữ liệu được tải tự động trong notebook bằng thư viện `datasets`; không cần mount Google Drive.

## Hướng dẫn chạy
### Trên Google Colab (khuyến nghị)
1. Mở notebook `notebooks/ML_Assignment.ipynb` trên Colab (Link Colab: `<link>`).
2. Ở cell đầu tiên, điền `REPO_URL` là link GitHub của repository này.
3. Chọn **Runtime → Change runtime type → GPU (T4)** (không bắt buộc nhưng giúp trích xuất BERT nhanh hơn).
4. Chọn **Runtime → Run all**. Notebook tự động: clone repo, cài thư viện (`requirements.txt`),
   tải dữ liệu, trích xuất đặc trưng, huấn luyện, đánh giá và lưu kết quả vào `features/` và `reports/`.

### Chạy trên máy cá nhân
```bash
git clone <link-repo>
cd <repo>
pip install -r requirements.txt
jupyter notebook notebooks/ML_Assignment.ipynb
```

### Yêu cầu thư viện
`numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `torch`, `transformers`,
`datasets`, `h5py`, `tqdm` (xem `requirements.txt`).

## Cấu trúc thư mục
```
.
├── notebooks/
│   └── ML_Assignment.ipynb   # Notebook chính (front-end)
├── modules/
│   ├── config.py             # Cấu hình: seed, đường dẫn, tham số TF-IDF/BERT, options
│   ├── data.py               # Tải dữ liệu, làm sạch văn bản, loại trùng lặp
│   ├── eda.py                # Các hàm phân tích khám phá dữ liệu
│   ├── features.py           # TF-IDF, BERT embeddings, lưu/nạp .npy/.h5
│   ├── models.py             # Naive Bayes (tự cài), MLP (PyTorch)
│   └── evaluate.py           # Metric, bảng so sánh, ma trận nhầm lẫn, biểu đồ
├── features/                 # Đặc trưng đã trích xuất (.npy, .h5)
├── reports/                  # Báo cáo PDF, bảng kết quả, hình ảnh
├── requirements.txt
└── README.md
```

## Đặc trưng đã trích xuất (`features/`)
| File | Nội dung |
|------|----------|
| `tfidf_train.h5`, `tfidf_test.h5` | Ma trận TF-IDF thưa (CSR: `data`, `indices`, `indptr`, thuộc tính `shape`) |
| `bert_train.npy`, `bert_test.npy` | BERT embeddings (`float32`, kích thước `N x 768`) |
| `labels_train.npy`, `labels_test.npy` | Nhãn tương ứng |

Nạp lại bằng `modules.features.load_features(path)`. Link tải (nếu không nộp kèm): `<link>`.

## Liên kết
- Báo cáo PDF: `reports/report.pdf` / `<link>`
- Colab notebook: `<link>`
