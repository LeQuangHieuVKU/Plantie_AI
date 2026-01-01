# File: predict_app.py
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import numpy as np
import json
import os
import mimetypes
from PIL import Image, UnidentifiedImageError

# --- CẤU HÌNH ---
MODEL_PATH = 'plant_disease_best_model.h5'
LABELS_PATH = 'class_indices.json'
CONFIDENCE_THRESHOLD = 0.70  # Ngưỡng tin cậy (70%). Dưới mức này sẽ báo "Không rõ"

# --- TỪ KHÓA NHẬN DIỆN CÂY (Dùng cho bộ lọc ImageNet) ---
# Các từ khóa tiếng Anh liên quan đến thực vật trong bộ dữ liệu ImageNet
PLANT_KEYWORDS = [
    'plant', 'leaf', 'flower', 'fruit', 'vegetable', 'tree', 'grass',
    'garden', 'pot', 'greenhouse', 'agriculture', 'orchard', 'broccoli',
    'cabbage', 'cauliflower', 'zucchini', 'cucumber', 'banana', 'corn',
    'lemon', 'orange', 'apple', 'grape', 'strawberry', 'potato', 'tomato'
]

# --- TỪ ĐIỂN GIẢI PHÁP ---
DISEASE_SOLUTIONS = {
    # --- CÀ CHUA (TOMATO) ---
    'Tomato_Bacterial_spot': (
        "Cà chua: Bệnh đốm vi khuẩn",
        "Phun chế phẩm gốc đồng (Copper). Cắt bỏ lá bệnh để tránh lây lan. Tránh tưới phun mưa lên lá."
    ),
    'Tomato_Early_blight': (
        "Cà chua: Bệnh đốm vòng (Sương mai sớm)",
        "Dùng thuốc hoạt chất Mancozeb hoặc Chlorothalonil. Tăng cường bón Kali, luân canh cây trồng."
    ),
    'Tomato_Late_blight': (
        "Cà chua: Bệnh mốc sương (Sương mai muộn)",
        "Bệnh rất nguy hiểm. Phun Metalaxyl hoặc Cymoxanil ngay. Tiêu hủy tàn dư cây bệnh sau thu hoạch."
    ),
    'Tomato_Leaf_Mold': (
        "Cà chua: Bệnh mốc lá",
        "Tăng độ thông thoáng cho vườn. Phun thuốc gốc Đồng hoặc lưu huỳnh. Giảm độ ẩm nhà kính."
    ),
    'Tomato_Septoria_leaf_spot': (
        "Cà chua: Đốm lá Septoria",
        "Vệ sinh đồng ruộng sạch sẽ. Phun thuốc gốc Carbendazim hoặc gốc Đồng."
    ),
    'Tomato_Spider_mites_Two_spotted_spider_mite': (
        "Cà chua: Nhện đỏ hai chấm",
        "Tưới nước áp lực mạnh để rửa trôi. Dùng dầu khoáng hoặc thuốc đặc trị nhện (Abamectin)."
    ),
    'Tomato_Target_Spot': (
        "Cà chua: Bệnh đốm đích",
        "Cải thiện lưu thông không khí. Phun thuốc trừ nấm phổ rộng."
    ),
    'Tomato_Yellow_Leaf_Curl_Virus': (
        "Cà chua: Virus xoăn vàng lá",
        "Không có thuốc chữa virus. Phải diệt bọ phấn trắng (vật trung gian) bằng bẫy dính hoặc thuốc sâu."
    ),
    'Tomato_Mosaic_virus': (
        "Cà chua: Virus khảm lá",
        "Nhổ bỏ và tiêu hủy cây bệnh ngay lập tức. Khử trùng dụng cụ cắt tỉa bằng cồn hoặc xà phòng."
    ),
    'Tomato_Healthy': (
        "Cà chua: Khỏe mạnh",
        "Cây phát triển tốt. Duy trì chế độ nước và phân bón hiện tại."
    ),

    # --- KHOAI TÂY (POTATO) ---
    'Potato_Early_blight': (
        "Khoai tây: Bệnh đốm vòng",
        "Phun thuốc Mancozeb. Tưới gốc thay vì tưới lá. Luân canh với cây họ đậu."
    ),
    'Potato_Late_blight': (
        "Khoai tây: Bệnh mốc sương",
        "Sử dụng thuốc diệt nấm thấm sâu (Metalaxyl). Thu hoạch vào ngày khô ráo."
    ),
    'Potato_Healthy': (
        "Khoai tây: Khỏe mạnh",
        "Tuyệt vời! Hãy tiếp tục theo dõi định kỳ."
    ),

    # --- NGÔ / BẮP (CORN) ---
    'Corn_(maize)__Common_rust_': (
        "Ngô: Bệnh rỉ sắt",
        "Thường không cần xử lý nếu nhẹ. Nếu nặng dùng thuốc chứa Azoxystrobin. Chọn giống kháng bệnh."
    ),
    'Corn_(maize)__Northern_Leaf_Blight': (
        "Ngô: Bệnh cháy lá lớn",
        "Luân canh cây trồng. Sử dụng giống kháng bệnh. Phun thuốc nếu bệnh xuất hiện sớm trước trổ cờ."
    ),
    'Corn_(maize)__Cercospora_leaf_spot_Gray_leaf_spot': (
        "Ngô: Bệnh đốm xám",
        "Vệ sinh đồng ruộng sau thu hoạch. Sử dụng thuốc trừ nấm nhóm Triazole."
    ),
    'Corn_(maize)__Healthy': (
        "Ngô: Khỏe mạnh",
        "Cây tốt. Chú ý bón phân cân đối đạm - lân - kali."
    ),

    # --- ỚT CHUÔNG (PEPPER BELL) ---
    'Pepper,_bell__Bacterial_spot': (
        "Ớt chuông: Đốm vi khuẩn",
        "Dùng thuốc gốc Đồng. Ngâm hạt giống vào nước ấm 50 độ C trước khi gieo để diệt khuẩn."
    ),
    'Pepper,_bell__Healthy': (
        "Ớt chuông: Khỏe mạnh",
        "Cây phát triển bình thường."
    ),

    # --- CÁC LOẠI KHÁC (VÍ DỤ) ---
    # Bạn có thể thêm các loại cây khác vào đây theo cấu trúc:
    # 'Ten_Folder_Trong_May_Ban': ("Tên Tiếng Việt", "Cách chữa trị"),
}


class SmartPredictor:
    def __init__(self):
        self.disease_model = None
        self.labels_map = None
        self.gatekeeper_model = None  # Model để kiểm tra xem có phải cây không
        self._load_resources()

    def _load_resources(self):
        print("⏳ Đang khởi tạo hệ thống...")
        if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
            self.disease_model = load_model(MODEL_PATH)
            with open(LABELS_PATH, 'r', encoding='utf-8') as f:
                self.labels_map = json.load(f)

            # Tải model MobileNetV2 gốc để làm "Bảo vệ" (Kiểm tra vật thể)
            self.gatekeeper_model = MobileNetV2(weights='imagenet')
            print("✅ Hệ thống đã sẵn sàng!")
        else:
            print("❌ LỖI: Thiếu file model hoặc file nhãn.")

    def is_valid_file(self, file_path):
        """Lớp 1: Kiểm tra file có phải là ảnh hợp lệ không"""
        # Check đuôi file cơ bản
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type or not mime_type.startswith('image'):
            return False, "Định dạng file không phải là ảnh (Video/Audio/Text...)."

        # Check thử mở file (tránh file ảnh bị lỗi/corrupt)
        try:
            with Image.open(file_path) as img:
                img.verify()  # Chỉ check header, rất nhanh
            return True, ""
        except (IOError, SyntaxError, UnidentifiedImageError):
            return False, "File ảnh bị lỗi hoặc định dạng không hỗ trợ."

    def is_plant_image(self, img_array):
        """Lớp 2: Dùng model ImageNet để check xem có phải cây cối không"""
        # Preprocess riêng cho MobileNet ImageNet
        gate_input = preprocess_input(img_array.copy())

        # Dự đoán top 5 vật thể trong ảnh
        preds = self.gatekeeper_model.predict(gate_input, verbose=0)
        decoded = decode_predictions(preds, top=5)[0]

        # decoded dạng: [('id', 'label', probability), ...]
        # Kiểm tra xem trong top 5 nhãn có từ khóa liên quan đến cây không
        detected_objects = []
        is_plant = False

        for _, label, prob in decoded:
            detected_objects.append(f"{label} ({prob:.2f})")
            # So sánh từ khóa (convert về lowercase)
            for keyword in PLANT_KEYWORDS:
                if keyword in label.lower():
                    is_plant = True
                    break
            if is_plant: break

        return is_plant, detected_objects

    def predict(self, image_path):
        # --- BƯỚC 1: KIỂM TRA FILE ---
        valid, msg = self.is_valid_file(image_path)
        if not valid:
            print(f"⛔ TỪ CHỐI: {msg}")
            return

        # Chuẩn bị ảnh
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

        # --- BƯỚC 2: KIỂM TRA CÓ PHẢI CÂY KHÔNG (GATEKEEPER) ---
        # Copy ảnh để check (tránh ảnh hưởng preprocess)
        print("🔍 Đang phân tích nội dung ảnh...")
        is_plant, objects_found = self.is_plant_image(img_array)

        if not is_plant:
            print("=" * 40)
            print(f"⛔ CẢNH BÁO: Ảnh này có vẻ không phải là cây cối/nông sản.")
            print(f"🤖 AI nhìn thấy: {', '.join(objects_found)}")
            print("➡️ Hệ thống từ chối chẩn đoán bệnh.")
            print("=" * 40 + "\n")
            return

        # --- BƯỚC 3: CHẨN ĐOÁN BỆNH & KIỂM TRA ĐỘ TIN CẬY ---
        # Preprocess cho model bệnh (Rescale 1./255 như lúc train)
        model_input = img_array / 255.0
        predictions = self.disease_model.predict(model_input, verbose=0)

        confidence = np.max(predictions[0])
        class_idx = np.argmax(predictions[0])
        class_key = self.labels_map[str(class_idx)]

        print("=" * 40)
        print(f"ẢNH: {os.path.basename(image_path)}")

        if confidence < CONFIDENCE_THRESHOLD:
            print(f"⚠️ KẾT QUẢ KHÔNG CHẮC CHẮN (Độ tin cậy: {confidence * 100:.1f}%)")
            print(f"Model nghi ngờ là: {class_key}, nhưng độ khớp quá thấp.")
            print("👉 Khả năng là loại bệnh chưa có trong dữ liệu hoặc ảnh mờ.")
        else:
            # Kết quả tốt
            print(f"✅ ĐỘ TIN CẬY: {confidence * 100:.1f}%")
            if class_key in DISEASE_SOLUTIONS:
                vi_name, solution = DISEASE_SOLUTIONS[class_key]
                print(f"🦠 CHẨN ĐOÁN: {vi_name}")
                print(f"💊 GIẢI PHÁP: {solution}")
            else:
                print(f"Chẩn đoán: {class_key}")
                print("Chưa có dữ liệu giải pháp.")

        print("=" * 40 + "\n")


if __name__ == "__main__":
    bot = SmartPredictor()

    if bot.disease_model:
        while True:
            path = input("Nhập đường dẫn ảnh (hoặc 'exit'): ").strip().strip('"')
            if path.lower() == 'exit': break
            bot.predict(path)

