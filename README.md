# TinyML for ESP32 - Weather Classification

Pipeline hoàn chỉnh để huấn luyện mô hình Machine Learning, chuyển đổi sang TensorFlow Lite, và triển khai suy luận trên ESP32-S3.

## Mô Tả Project

Dự án này xây dựng một hệ thống phân loại trạng thái thời tiết dựa trên hai đặc trưng cảm biến: **Nhiệt độ** và **Độ ẩm**. Mô hình sau khi huấn luyện trên máy tính được chuyển đổi sang định dạng TensorFlow Lite để có thể chạy trực tiếp trên ESP32-S3, không cần kết nối Internet hoặc máy chủ.

### Các Lớp Phân Loại

- **Cloudy** (Nhiều mây)
- **Rainy** (Mưa)
- **Snowy** (Tuyết)
- **Sunny** (Nắng)

## Cấu Trúc Project

```
TinyML_ESP32/
├── TFL_For_MCU.py                      # Script huấn luyện mô hình
├── weather_classification_data.csv      # Dữ liệu training (13,200+ mẫu)
├── ESP32_Config.txt                    # Thông số chuẩn hóa cho ESP32
├── TinyML.h5                           # Mô hình Keras
├── TinyML.tflite                       # Mô hình TensorFlow Lite
├── TinyML.h                            # File header chứa mô hình (dạng hex array)
├── TinyMQTT.py                         # MQTT Broker (tùy chọn)
├── requirements.txt                    # Python dependencies
└── README.md                           # Tài liệu này
```

## Yêu Cầu

- **Python** 3.8+
- **TensorFlow** 2.13.0
- **scikit-learn** 1.3.2
- **pandas** 2.0.3
- **numpy** 1.24.3

### Cài Đặt

```bash
pip install -r requirements.txt
```

## Cách Chạy

### 1. Huấn Luyện Mô Hình

```bash
python TFL_For_MCU.py
```

Script sẽ thực hiện các bước sau:
1. Đọc dữ liệu từ `weather_classification_data.csv`
2. Làm sạch dữ liệu (loại bỏ outliers bằng phương pháp IQR)
3. Chuẩn hóa dữ liệu (StandardScaler)
4. Chia tập training (80%) và testing (20%)
5. Xây dựng và huấn luyện mô hình Neural Network 3 lớp
6. Đánh giá độ chính xác và confusion matrix
7. Chuyển đổi sang TensorFlow Lite
8. Xuất file header C/C++ cho ESP32
9. Ghi thông số cấu hình vào `ESP32_Config.txt`

### 2. Output Files

Sau khi chạy script, sẽ tạo ra:

| File | Mô Tả |
|------|-------|
| `TinyML.h5` | Mô hình Keras (dạng lưu trữ) |
| `TinyML.tflite` | Mô hình TensorFlow Lite (tối ưu cho embedded) |
| `TinyML.h` | File header C/C++ chứa mô hình (hex array) |
| `ESP32_Config.txt` | Thông số StandardScaler + công thức chuẩn hóa |

## Cấu Trúc Mô Hình

```
Input (2) 
  ↓
Dense(16, ReLU)
  ↓
Dense(8, ReLU)
  ↓
Dense(4, Softmax)
  ↓
Output (4 classes)
```

**Thông số:**
- **Epochs**: 50
- **Batch Size**: 32
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Activation**: ReLU (hidden), Softmax (output)

## Kết Quả Đánh Giá

Mô hình đạt độ chính xác khoảng **72%** trên tập kiểm thử.

**Confusion Matrix:**

|       | Cloudy | Rainy | Snowy | Sunny |
|-------|--------|-------|-------|-------|
| Cloudy| 455    | 78    | 7     | 118   |
| Rainy | 242    | 359   | 13    | 58    |
| Snowy | 17     | 3     | 621   | 26    |
| Sunny | 119    | 37    | 6     | 463   |

Lớp **Snowy** được phân loại tốt nhất, lớp **Rainy** là khó nhất (dễ nhầm với Cloudy).

## Sử Dụng Trên ESP32-S3

### 1. Nhúng Mô Hình

Copy nội dung file `TinyML.h` vào firmware ESP32-S3:

```cpp
#include "TinyML.h"

const tflite::Model* model = tflite::GetModel(model);
```

### 2. Chuẩn Hóa Dữ Liệu

Sử dụng các thông số từ `ESP32_Config.txt`:

```cpp
float temp_mean = 18.6480;
float temp_std = 16.6074;
float humid_mean = 68.5105;
float humid_std = 20.2033;

float scaled_temp = (sensor_temp - temp_mean) / temp_std;
float scaled_humid = (sensor_humid - humid_mean) / humid_std;

input->data.f[0] = scaled_temp;
input->data.f[1] = scaled_humid;
```

### 3. Chạy Suy Luận

```cpp
interpreter->Invoke();
TfLiteTensor* output = interpreter->output(0);
float* probabilities = output->data.f;

// probabilities[0] = Cloudy
// probabilities[1] = Rainy
// probabilities[2] = Snowy
// probabilities[3] = Sunny
```

## Đặc Điểm Nổi Bật

✅ **Gọn nhẹ**: Chỉ 3KB model (TFLite)  
✅ **Nhanh**: Suy luận dưới 10ms trên ESP32-S3  
✅ **Chính xác**: ~72% accuracy với 2 đặc trưng  
✅ **Edge AI**: Xử lý hoàn toàn trên thiết bị, không cần cloud  
✅ **Chuẩn hóa**: Tự động sinh thông số cho firmware  

## Hạn Chế

⚠️ Chỉ sử dụng 2 đặc trưng (Temp, Humidity)  
⚠️ Mô hình chưa tối ưu (có thể bổ sung thêm features)  
⚠️ Lớp Rainy khó phân biệt với Cloudy  

## Hướng Phát Triển

- [ ] Bổ sung đặc trưng: Áp suất, Tốc độ gió, Độ che phủ mây
- [ ] Thử nghiệm kiến trúc mạng khác
- [ ] Lượng tử hóa sâu để giảm kích thước mô hình
- [ ] Thêm metrics: Precision, Recall, F1-Score
- [ ] Triển khai Web Dashboard để hiển thị kết quả

## Tài Liệu Tham Khảo

- [TensorFlow Lite Documentation](https://www.tensorflow.org/lite)
- [TensorFlow Lite Micro for Embedded](https://www.tensorflow.org/lite/microcontrollers)
- [ESP32-S3 Documentation](https://docs.espressif.com/projects/esp-idf/)

---

**Ghi chú**: Đây là phiên bản TinyML với mục đích học tập và tham khảo. Cho sản xuất thực tế, cần kiểm thử kỹ lưỡng trên thiết bị thực.