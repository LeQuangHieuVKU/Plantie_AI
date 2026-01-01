# File: web_app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import json
import os

# --- CẤU HÌNH ---
MODEL_PATH = 'plant_disease_best_model.h5'
LABELS_PATH = 'class_indices.json'
CONFIDENCE_THRESHOLD = 0.70

# --- TỪ ĐIỂN GIẢI PHÁP ---
DISEASE_SOLUTIONS = {
    "Apple___Apple_scab": (
        "Táo: Bệnh ghẻ táo",
        "Tỉa lá già. Phun Mancozeb hoặc Chlorothalonil. Giữ lá khô khi tưới."
    ),
    "Apple___Black_rot": (
        "Táo: Thối đen",
        "Cắt bỏ cành bệnh. Thu gom quả rụng. Phun Captan theo lịch."
    ),
    "Apple___Cedar_apple_rust": (
        "Táo: Gỉ sắt",
        "Cắt bỏ lá bệnh. Giảm ẩm cho tán. Phun Mancozeb định kỳ."
    ),
    "Apple___healthy": (
        "Táo: Khỏe mạnh",
        "Duy trì tưới vừa phải. Giữ gốc thoáng."
    ),
    "Blueberry___healthy": (
        "Việt quất: Khỏe mạnh",
        "Giữ đất thoát nước. Bón phân đúng liều."
    ),
    "Cherry_(including_sour)___Powdery_mildew": (
        "Anh đào: Phấn trắng",
        "Phun lưu huỳnh hoặc Copper. Tăng thông gió tán."
    ),
    "Cherry_(including_sour)___healthy": (
        "Anh đào: Khỏe mạnh",
        "Tưới gốc đều. Giữ vườn sạch lá rụng."
    ),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": (
        "Ngô: Đốm lá xám",
        "Thu gom lá bệnh. Phun Strobilurin. Giảm mật độ cây."
    ),
    "Corn_(maize)___Common_rust_": (
        "Ngô: Gỉ sắt",
        "Phun Triazole. Chọn giống kháng. Giữ lá khô."
    ),
    "Corn_(maize)___Northern_Leaf_Blight": (
        "Ngô: Cháy lá Bắc",
        "Dọn tàn dư lá. Phun Mancozeb. Tăng thông gió ruộng."
    ),
    "Corn_(maize)___healthy": (
        "Ngô: Khỏe mạnh",
        "Tưới đều. Tránh ruộng ẩm kéo dài."
    ),
    "Grape___Black_rot": (
        "Nho: Thối đen",
        "Thu gom lá bệnh. Phun Captan. Tăng khoảng cách tán."
    ),
    "Grape___Esca_(Black_Measles)": (
        "Nho: Esca",
        "Cắt bỏ cành bệnh. Giảm tưới ẩm. Không chữa khỏi hoàn toàn nên cần quản lý tốt."
    ),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": (
        "Nho: Cháy lá",
        "Phun Mancozeb. Loại bỏ lá bệnh. Giữ tán thông thoáng."
    ),
    "Grape___healthy": (
        "Nho: Khỏe mạnh",
        "Cắt tỉa hợp lý. Tránh tưới lên lá."
    ),
    "Orange___Haunglongbing_(Citrus_greening)": (
        "Cam: Vàng lá Greening",
        "Cắt nhánh bệnh. Quản lý rầy chổng cánh. Không phục hồi hoàn toàn."
    ),
    "Peach___Bacterial_spot": (
        "Đào: Đốm vi khuẩn",
        "Phun Copper. Tỉa thông thoáng tán."
    ),
    "Peach___healthy": (
        "Đào: Khỏe mạnh",
        "Bón phân đều. Giữ đất thông thoáng."
    ),
    "Pepper,_bell___Bacterial_spot": (
        "Ớt chuông: Đốm vi khuẩn",
        "Phun Copper. Tránh tưới lên lá."
    ),
    "Pepper,_bell___healthy": (
        "Ớt chuông: Khỏe mạnh",
        "Tưới gốc. Giữ khoảng cách cây."
    ),
    "Potato___Early_blight": (
        "Khoai tây: Mốc sớm",
        "Phun Chlorothalonil. Thu gom lá bệnh."
    ),
    "Potato___Late_blight": (
        "Khoai tây: Mốc muộn",
        "Phun Copper. Thu gom tàn dư. Giảm ẩm ruộng."
    ),
    "Potato___healthy": (
        "Khoai tây: Khỏe mạnh",
        "Tưới đều. Tránh làm ướt lá."
    ),
    "Raspberry___healthy": (
        "Mâm xôi: Khỏe mạnh",
        "Giữ luống thoáng. Tưới đúng mức."
    ),
    "Soybean___healthy": (
        "Đậu tương: Khỏe mạnh",
        "Bón phân hợp lý. Giảm cỏ dại."
    ),
    "Squash___Powdery_mildew": (
        "Bí: Phấn trắng",
        "Phun lưu huỳnh. Tăng thông gió."
    ),
    "Strawberry___Leaf_scorch": (
        "Dâu: Cháy lá",
        "Cắt lá cháy. Phun Copper. Giảm ẩm."
    ),
    "Strawberry___healthy": (
        "Dâu: Khỏe mạnh",
        "Giữ đất tơi. Tưới gốc."
    ),
    "Tomato___Bacterial_spot": (
        "Cà chua: Đốm vi khuẩn",
        "Phun Copper. Cắt bỏ lá bệnh. Tránh tưới phun mưa."
    ),
    "Tomato___Early_blight": (
        "Cà chua: Mốc sớm",
        "Phun Chlorothalonil. Tỉa lá gốc."
    ),
    "Tomato___Late_blight": (
        "Cà chua: Mốc muộn",
        "Cắt lá bệnh. Phun Copper. Giảm ẩm tán."
    ),
    "Tomato___Leaf_Mold": (
        "Cà chua: Mốc lá",
        "Tăng thông gió. Phun Copper."
    ),
    "Tomato___Septoria_leaf_spot": (
        "Cà chua: Đốm Septoria",
        "Cắt lá bệnh. Phun Mancozeb."
    ),
    "Tomato___Spider_mites Two-spotted_spider_mite": (
        "Cà chua: Nhện đỏ",
        "Phun dầu khoáng. Tăng ẩm nhẹ cho tán."
    ),
    "Tomato___Target_Spot": (
        "Cà chua: Đốm mục tiêu",
        "Phun Chlorothalonil. Loại bỏ lá bệnh."
    ),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": (
        "Cà chua: Vàng xoăn lá",
        "Quản lý bọ phấn. Loại bỏ cây bệnh."
    ),
    "Tomato___Tomato_mosaic_virus": (
        "Cà chua: Virus khảm",
        "Không có thuốc trị. Loại bỏ cây bệnh. Khử trùng dụng cụ."
    ),
    "Tomato___healthy": (
        "Cà chua: Khỏe mạnh",
        "Tưới gốc. Bón phân đều."
    )
}


# --- TỪ KHÓA LỌC ẢNH ---
PLANT_KEYWORDS = ['plant', 'leaf', 'flower', 'fruit', 'vegetable', 'tree',
                  'grass', 'garden', 'agriculture', 'broccoli', 'cabbage',
                  'cucumber', 'corn', 'potato', 'tomato', 'pepper']


# --- HÀM LOAD MODEL (Dùng cache để không phải load lại mỗi lần f5) ---
@st.cache_resource
def load_system():
    # 1. Load Model Bệnh
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        return None, None, None

    disease_model = load_model(MODEL_PATH)
    with open(LABELS_PATH, 'r', encoding='utf-8') as f:
        labels_map = json.load(f)

    # 2. Load Model "Bảo vệ" (ImageNet)
    gatekeeper = MobileNetV2(weights='imagenet')

    return disease_model, labels_map, gatekeeper


# --- GIAO DIỆN WEB ---
st.set_page_config(page_title="Bác sĩ Cây Trồng AI", page_icon="🌿")

st.title("🌿 Bác Sĩ Cây Trồng AI")
st.write("Tải ảnh lá cây lên để chẩn đoán bệnh và nhận phác đồ điều trị.")

# Load hệ thống
disease_model, labels_map, gatekeeper_model = load_system()

if disease_model is None:
    st.error(f"LỖI: Không tìm thấy file '{MODEL_PATH}'. Hãy chạy train_upgrade.py trước!")
else:
    # Widget tải ảnh
    uploaded_file = st.file_uploader("Chọn ảnh lá cây...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        try:
            # Hiển thị ảnh
            img = Image.open(uploaded_file)
            st.image(img, caption='Ảnh đã tải lên', use_column_width=True)

            # --- BƯỚC 1: TIỀN XỬ LÝ ---
            # Resize ảnh về 224x224 cho đúng chuẩn model
            img = img.resize((224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            with st.spinner('Đang phân tích tế bào thực vật...'):

                # --- BƯỚC 2: KIỂM TRA PHẢI CÂY KHÔNG? ---
                gate_input = preprocess_input(img_array.copy())
                preds = gatekeeper_model.predict(gate_input)
                decoded = decode_predictions(preds, top=5)[0]

                is_plant = False
                detected_text = []
                for _, label, _ in decoded:
                    detected_text.append(label)
                    for kw in PLANT_KEYWORDS:
                        if kw in label.lower():
                            is_plant = True
                            break
                    if is_plant: break

                if not is_plant:
                    st.error("⛔ CẢNH BÁO: Ảnh này không giống cây trồng!")
                    st.warning(f"AI nhìn thấy: {', '.join(detected_text)}")
                    st.info("Vui lòng tải ảnh lá cây hoặc quả rõ nét.")

                else:
                    # --- BƯỚC 3: CHẨN ĐOÁN BỆNH ---
                    model_input = img_array / 255.0
                    predictions = disease_model.predict(model_input)

                    confidence = np.max(predictions[0])
                    class_idx = np.argmax(predictions[0])
                    class_key = labels_map[str(class_idx)]

                    # Hiển thị kết quả
                    if confidence < CONFIDENCE_THRESHOLD:
                        st.warning(f"⚠️ Kết quả không chắc chắn ({confidence * 100:.1f}%)")
                        st.write(f"Nghi ngờ là: **{class_key}** nhưng không khớp lắm.")
                        st.write("Lời khuyên: Hãy chụp ảnh gần hơn và đủ sáng.")
                    else:
                        st.success(f"✅ Đã phát hiện bệnh! (Độ tin cậy: {confidence * 100:.1f}%)")

                        # Tra cứu từ điển
                        if class_key in DISEASE_SOLUTIONS:
                            vi_name, solution = DISEASE_SOLUTIONS[class_key]
                            st.header(f"🦠 {vi_name}")

                            st.markdown("### 💊 Phác đồ điều trị:")
                            st.info(solution)
                        else:
                            st.header(f"Kết quả: {class_key}")
                            st.write("Chưa có dữ liệu thuốc cho bệnh này trong hệ thống.")

        except Exception as e:
            st.error(f"Có lỗi khi xử lý ảnh: {e}")

# ---streamlit run web_app.py ---