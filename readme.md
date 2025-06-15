# 😊 Ứng dụng Nhận diện Cảm xúc Khuôn mặt

Ứng dụng web sử dụng **Deep Learning** để nhận diện cảm xúc từ hình ảnh khuôn mặt. Được xây dựng bằng **Streamlit** và **TensorFlow**.

---

## 📋 Mục lục

- [✨ Tính năng](#-tính-năng)  
- [🔧 Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)  
- [🚀 Cài đặt](#-cài-đặt)  
- [🎯 Sử dụng](#-sử-dụng)  
- [📁 Cấu trúc dự án](#-cấu-trúc-dự-án)  
- [🧠 Thông tin Model](#-thông-tin-model)  
- [🔍 Khắc phục sự cố](#-khắc-phục-sự-cố)  

---

## ✨ Tính năng

- 🖼️ **Upload ảnh**: Tải ảnh từ máy tính để phân tích cảm xúc  
- 📷 **Chụp ảnh trực tiếp**: Sử dụng webcam để phân tích ngay lập tức  
- 🎭 **Nhận diện 7 cảm xúc**: Vui vẻ, Buồn bã, Tức giận, Sợ hãi, Ngạc nhiên, Ghê tởm, Bình thường  
- 🌐 **Hỗ trợ tiếng Việt**: Giao diện và kết quả hiển thị bằng tiếng Việt  
- 📊 **Thông tin model**: Hiển thị độ chính xác và kiến trúc model  

---

## 🔧 Yêu cầu hệ thống

**Phần mềm:**

- Python: `3.8` - `3.11` (khuyên dùng: `3.11.8`)  
- OS: Windows 10/11, macOS, Linux  
- RAM: ≥ 4GB (khuyên dùng: 8GB)  
- Dung lượng ổ cứng: ~2GB  

**Lưu ý:**

- ❌ Không hỗ trợ Python `3.12+`  
- ⚠️ Windows: Có thể cần cài [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)

---

## 🚀 Cài đặt

### Bước 1: Tạo môi trường ảo

```bash
cd path/to/your/project
C:\Python311\python.exe -m venv emotion_env
# Windows
emotion_env\Scripts\activate
# macOS/Linux
source emotion_env/bin/activate
```
### Bước 2: Cài dependencies
```bash
python -m pip install --upgrade pip
pip install tensorflow
pip install streamlit opencv-python pillow numpy
# Hoặc:
pip install -r requirements.txt
```
### Lưu ý: Vào link này để tải file trainmodel và đưa vào thư mục bạn clone or giải nén:
```
https://drive.google.com/file/d/1KF92yXGHlsFmkvAzWo28tPbirYvI6461/view?usp=sharing
```
##  Sử dụng ứng dụng
Sau khi chạy lệnh trên, trình duyệt sẽ tự động mở và truy cập:
```bash
streamlit run app.py
```
http://localhost:8501
### Hướng dẫn sử dụng

**Chọn chế độ**:

- 📤 Upload ảnh: Tải ảnh từ máy tính
- 📷 Chụp ảnh từ camera: Sử dụng webcam


**Upload ảnh**:

- Click "Browse files" hoặc kéo thả ảnh
- Hỗ trợ: JPG, JPEG, PNG, BMP
- Kết quả hiển thị ngay sau khi upload


**Chụp ảnh**:

- Cho phép truy cập camera
- Click "Take Photo" để chụp
- Kết quả hiển thị ngay lập tức

### 📁 Cấu trúc dự án
```
emotion-recognition/
│
├── app.py                          # File chính của ứng dụng Streamlit
├── emotion_predictor.py            # Module dự đoán cảm xúc
├── emotion_labels.json             # Labels cảm xúc (tiếng Anh và Việt)
├── model_info.json                 # Thông tin chi tiết về model
├── requirements.txt                # Danh sách dependencies
├── README.md                       # File hướng dẫn này
└── emotion_recognition_final.h5    # Model đã train (định dạng H5)
```
### Mô tả files quan trọng:

app.py: Giao diện chính, xử lý upload/camera và hiển thị kết quả
emotion_predictor.py: Module độc lập để dự đoán cảm xúc
emotion_labels.json: Mapping từ số sang tên cảm xúc
model_info.json: Thống kê về model (độ chính xác, số parameters, v.v.)
Model files: Chứa neural network đã được train

### 🧠 Thông tin Model
Kiến trúc:

Loại: Convolutional Neural Network (CNN)
Input: Ảnh grayscale 48x48 pixels
Output: 7 classes cảm xúc
Parameters: ~850,000 tham số

Hiệu suất:

Độ chính xác: ~65.5%
Dataset: FER-2013 (35,000+ ảnh)
Epochs: 50

Các cảm xúc nhận diện:

😊 Vui vẻ (Happy)
😢 Buồn bã (Sad)
😠 Tức giận (Angry)
😨 Sợ hãi (Fear)
😲 Ngạc nhiên (Surprise)
🤢 Ghê tởm (Disgust)
😐 Bình thường (Neutral)
### 🔍 Khắc phục sự cố
**ImportError: DLL load failed**
```bash
# Cài Visual C++ Redistributable
# Tải từ: https://aka.ms/vs/17/release/vc_redist.x64.exe
#Bắt buộc phải là python 3.9-3.11
```
### 📝 Ghi chú
Model được train trên dataset FER-2013, có thể không hoàn hảo 100%
Kết quả tốt nhất khi ảnh có khuôn mặt rõ ràng, ánh sáng đầy đủ
Ứng dụng chạy local, không gửi dữ liệu ra internet
Compatible với Windows, macOS, và Linux

Phiên bản: 1.0
Cập nhật lần cuối: Tháng 6/2025s
Tác giả: Nhóm 1 Trí Tuệ Nhân Tạo K25CNTTA Học viện Ngân Hàng