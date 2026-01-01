# File: train_upgrade.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import json
import cv2  # <--- Thêm mới
import numpy as np  # <--- Thêm mới

# --- CẤU HÌNH ---
TRAIN_DIR = 'D:/Documents/Python/PythonProject/Plantie/train'
VAL_DIR = 'D:/Documents/Python/PythonProject/Plantie/valid'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
MODEL_NAME = 'plant_disease_best_model.h5'
LABELS_JSON = 'class_indices.json'


# --- HÀM XỬ LÝ ÁNH SÁNG (MỚI) ---
def adjust_brightness_contrast(image):
    """
    Hàm tự động cân bằng ánh sáng nếu ảnh quá tối hoặc quá sáng
    Sử dụng Gamma Correction trên kênh L (Lightness) của hệ màu LAB.
    """
    # Keras generator trả về ảnh dạng float, cần chuyển về uint8 để dùng OpenCV
    img = image.astype(np.uint8)

    # Chuyển từ RGB sang LAB
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Tính độ sáng trung bình của kênh L
    mean_brightness = np.mean(l)

    # Ngưỡng (Threshold): 0 (đen) -> 255 (trắng)
    # Nếu trung bình < 90 là hơi tối -> cần làm sáng (gamma < 1)
    # Nếu trung bình > 180 là hơi chói -> cần làm tối (gamma > 1)

    gamma = 1.0
    if mean_brightness < 90:
        gamma = 0.6  # Giảm gamma để làm sáng ảnh
    elif mean_brightness > 180:
        gamma = 1.5  # Tăng gamma để làm tối ảnh

    # Chỉ áp dụng nếu cần thiết để tiết kiệm thời gian tính toán
    if gamma != 1.0:
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        # Áp dụng bảng lookup table (LUT) cho kênh L
        l = cv2.LUT(l, table)

        # Gộp lại và chuyển về RGB
        lab = cv2.merge((l, a, b))
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return result.astype(np.float32)  # Trả về float cho Keras

    return image.astype(np.float32)


def train_model():
    # 1. Data Generators
    # Thêm tham số preprocessing_function vào đây
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        preprocessing_function=adjust_brightness_contrast  # <--- TÍCH HỢP Ở ĐÂY
    )

    # Validation cũng nên được xử lý ánh sáng tương tự để công bằng khi đánh giá
    val_datagen = ImageDataGenerator(
        rescale=1. / 255,
        preprocessing_function=adjust_brightness_contrast  # <--- TÍCH HỢP Ở ĐÂY
    )

    print("--- Đang tải dữ liệu ---")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
    )
    validation_generator = val_datagen.flow_from_directory(
        VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
    )

    # 2. Xây dựng Model (MobileNetV2)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(train_generator.num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # --- NÂNG CẤP 1: CHIẾN LƯỢC HUẤN LUYỆN 2 GIAI ĐOẠN ---

    # Giai đoạn 1: Đóng băng base_model
    print("--- Giai đoạn 1: Train các lớp đỉnh (Top layers) ---")
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_generator, epochs=8, validation_data=validation_generator)

    # Giai đoạn 2: Fine-tuning
    print("--- Giai đoạn 2: Fine-tuning (Tinh chỉnh sâu) ---")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

    # --- NÂNG CẤP 2: CALLBACKS ---
    callbacks = [
        ModelCheckpoint(MODEL_NAME, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)
    ]

    print(f"Bắt đầu Fine-tuning trong {EPOCHS} epochs...")
    model.fit(train_generator, epochs=EPOCHS, validation_data=validation_generator, callbacks=callbacks)

    # 3. Lưu Nhãn
    labels = {v: k for k, v in train_generator.class_indices.items()}
    with open(LABELS_JSON, 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=4)
    print(f"Đã lưu danh sách nhãn vào: {LABELS_JSON}")


if __name__ == "__main__":
    if os.path.exists(TRAIN_DIR):
        train_model()
    else:
        print("Vui lòng kiểm tra đường dẫn dữ liệu.")

