# 🌿 Plantie AI — Bác sĩ cây trồng thông minh

Plantie AI là dự án ứng dụng **AI/Deep Learning** để nhận diện bệnh trên lá cây từ hình ảnh, giúp người dùng (nông dân, kỹ thuật viên nông nghiệp, người trồng cây tại nhà) phát hiện sớm vấn đề và tham khảo hướng xử lý nhanh chóng.

---

## 🎯 Mục đích dự án

Dự án giải quyết bài toán:
- Chẩn đoán bệnh cây thủ công thường mất thời gian và phụ thuộc kinh nghiệm.
- Khó xác định chính xác bệnh khi người dùng không có chuyên môn.

Plantie AI hỗ trợ:
- Phân loại bệnh cây từ ảnh lá.
- Trả về độ tin cậy dự đoán.
- Gợi ý giải pháp xử lý theo từng loại bệnh.

---

## ✨ Tính năng chính

- 🧠 Dự đoán bệnh cây bằng mô hình học sâu (`plant_disease_best_model.h5`).
- 🛡️ Lọc ảnh đầu vào bằng MobileNetV2 (ImageNet) để giảm trường hợp ảnh không phải cây trồng.
- 📊 Hiển thị mức độ tin cậy của kết quả.
- 💊 Gợi ý hướng xử lý/điều trị theo từng bệnh.
- 🌐 Giao diện web thân thiện với **Streamlit**.
- 💻 Hỗ trợ chạy dự đoán qua terminal với script CLI.

---

## 🧰 Công nghệ sử dụng

| Thành phần | Công nghệ |
|---|---|
| Ngôn ngữ | Python |
| Machine Learning / Deep Learning | TensorFlow, Keras (MobileNetV2) |
| Xử lý dữ liệu | NumPy, SciPy |
| Xử lý ảnh | Pillow, OpenCV |
| Web UI | Streamlit |

---

## ⚙️ Hướng dẫn cài đặt (Local)

### 1) Clone repository
```bash
git clone https://github.com/LeQuangHieuVKU/Plantie_AI.git
cd Plantie_AI
```

### 2) Tạo môi trường ảo (khuyến nghị)
```bash
python -m venv .venv
```

**Windows**
```bash
.venv\Scripts\activate
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

### 3) Cài đặt dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Cách sử dụng

### Chạy ứng dụng web
```bash
streamlit run web_app.py
```

Sau khi chạy, mở URL Streamlit trong trình duyệt, tải ảnh lá cây lên để nhận:
- Loại bệnh dự đoán
- Độ tin cậy
- Gợi ý xử lý

### Chạy dự đoán bằng CLI
```bash
python predict_app.py
```

Nhập đường dẫn ảnh khi chương trình yêu cầu.

---

## 🗂️ Cấu trúc thư mục chính

```text
Plantie_AI/
├── web_app.py                    # Giao diện web Streamlit
├── predict_app.py                # Dự đoán qua terminal (CLI)
├── train.py                      # Script huấn luyện mô hình
├── requirements.txt              # Danh sách thư viện
├── class_indices.json            # Mapping chỉ số lớp -> tên lớp
├── plant_disease_best_model.h5   # Mô hình đã huấn luyện
├── train.rar                     # Dữ liệu train (nén)
└── valid.rar                     # Dữ liệu validation (nén)
```

---

## 📌 Ghi chú

- Đảm bảo tồn tại `plant_disease_best_model.h5` và `class_indices.json` trước khi chạy dự đoán.
- Kết quả phụ thuộc chất lượng ảnh đầu vào (nên chụp rõ nét, đủ sáng, cận cảnh lá/cây).

