import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf

PREFIX = "TinyML"

print("1. Đang đọc dữ liệu...")
df = pd.read_csv("weather_classification_data.csv")

print("2. Chọn cột: Temperature, Humidity, Weather Type")
df = df[['Temperature', 'Humidity', 'Weather Type']]

print("3. Làm sạch dữ liệu (loại bỏ Outliers)...")
def remove_outliers(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

df = remove_outliers(df, ['Temperature', 'Humidity'])

print("4. Encode nhãn dự đoán...")
le = LabelEncoder()
y = le.fit_transform(df['Weather Type'])
print(f"   Các lớp tự động nhận diện: {le.classes_}")

X = df[['Temperature', 'Humidity']].values

print("5. Chia tập Train/Test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("6. Scaling đầu vào (StandardScaler)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"   -> Temp Mean: {scaler.mean_[0]:.4f}, Std: {scaler.scale_[0]:.4f}")
print(f"   -> Humid Mean: {scaler.mean_[1]:.4f}, Std: {scaler.scale_[1]:.4f}")

print("7. Xây dựng và Train Model (16 -> 8 -> 4)...")
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_data=(X_test_scaled, y_test))

print("8. Đánh giá Confusion Matrix và Accuracy...")
y_pred_probs = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_probs, axis=1)

acc = accuracy_score(y_test, y_pred)
print(f"   => Độ chính xác (Accuracy): {acc * 100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
print("   => Confusion Matrix:")
print(cm)

print("9. Convert thành TFLite...")
model.save(PREFIX + '.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(PREFIX + ".tflite", "wb") as f:
    f.write(tflite_model)

print("10. Xuất file TinyML.h cho MCU...")
tflite_path = PREFIX + '.tflite'
output_header_path = PREFIX + '.h'

with open(tflite_path, 'rb') as tflite_file:
    tflite_content = tflite_file.read()

hex_lines = [', '.join([f'0x{byte:02x}' for byte in tflite_content[i:i + 12]]) for i in range(0, len(tflite_content), 12)]
hex_array = ',\n  '.join(hex_lines)

with open(output_header_path, 'w', encoding='utf-8') as header_file:
    header_file.write(f'// Số lượng lớp: {len(le.classes_)}\n')
    header_file.write(f'// Các lớp bao gồm: {", ".join(le.classes_)}\n')
    header_file.write('const unsigned char model[] = {\n  ')
    header_file.write(f'{hex_array}\n')
    header_file.write('};\n\n')

# 11. Ghi tóm tắt ra file txt để code vô ESP32
with open("ESP32_Config.txt", "w", encoding="utf-8") as f:
    f.write("\n1. Tham số do StandardScaler cung cấp:\n")
    f.write(f"float temp_mean = {scaler.mean_[0]:.4f};\n")
    f.write(f"float temp_std = {scaler.scale_[0]:.4f};\n")
    f.write(f"float humid_mean = {scaler.mean_[1]:.4f};\n")
    f.write(f"float humid_std = {scaler.scale_[1]:.4f};\n")
    f.write("\n2. Công thức đưa dữ liệu vào:\n")
    f.write("float scaled_temp = (real_temp - temp_mean) / temp_std;\n")
    f.write("float scaled_humid = (real_humid - humid_mean) / humid_std;\n")
    f.write("input->data.f[0] = scaled_temp;\n")
    f.write("input->data.f[1] = scaled_humid;\n")
    f.write("\n3. Bảng tra mã nhãn:\n")
    for idx, name in enumerate(le.classes_):
        f.write(f"Index {idx} => {name}\n")

print("\n--- HOÀN TẤT! ---")
print("Đã phát sinh file ESP32_Config.txt chứa các thông tin chuẩn hoá để đem lên ESP32.")