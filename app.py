import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
import os
import sys

# Cấu hình trang
st.set_page_config(
   page_title="🎭 Nhận diện cảm xúc khuôn mặt",
   page_icon="🎭",
   layout="wide"
)

# Kiểm tra và import TensorFlow
TF_AVAILABLE = False
TF_ERROR = None

def check_tensorflow():
   """Kiểm tra TensorFlow có sẵn không"""
   global TF_AVAILABLE, TF_ERROR
   
   try:
       # Tắt warnings và logs
       import warnings
       warnings.filterwarnings('ignore')
       os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
       os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
       
       # Import TensorFlow
       import tensorflow as tf
       
       # Hiển thị thông tin version
       tf_version = tf.__version__
       python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
       
       st.sidebar.success(f"✅ TensorFlow {tf_version}")
       st.sidebar.info(f"🐍 Python {python_version}")
       
       TF_AVAILABLE = True
       return True
       
   except ImportError as e:
       if "DLL load failed" in str(e):
           TF_ERROR = "❌ Lỗi DLL: Có thể thiếu Visual C++ Redistributable"
       else:
           TF_ERROR = f"❌ Lỗi import TensorFlow: {str(e)}"
       return False
   except Exception as e:
       TF_ERROR = f"❌ Lỗi không xác định: {str(e)}"
       return False

# Kiểm tra TensorFlow
if not check_tensorflow():
   st.error(TF_ERROR)
   st.markdown("""
   ## 🔧 Cách khắc phục:
   
   1. **Cài đặt TensorFlow:**
   ```bash
   pip install tensorflow==2.15.0
   ```
   
   2. **Nếu gặp lỗi DLL, cài Visual C++ Redistributable:**
   - Tải từ: https://aka.ms/vs/17/release/vc_redist.x64.exe
   
   3. **Hoặc thử TensorFlow CPU:**
   ```bash
   pip uninstall tensorflow
   pip install tensorflow-cpu==2.15.0
   ```
   """)
   st.stop()

# Import TensorFlow sau khi kiểm tra
import tensorflow as tf

# Custom CSS
st.markdown("""
<style>
.main-header {
   font-size: 3rem;
   color: #FF6B6B;
   text-align: center;
   margin-bottom: 2rem;
   text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}
.emotion-result {
   font-size: 2.5rem;
   font-weight: bold;
   text-align: center;
   padding: 1.5rem;
   border-radius: 15px;
   margin: 1rem 0;
   box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
.confidence-bar {
   background: linear-gradient(90deg, #ff4444, #ffaa00, #44ff44);
   height: 8px;
   border-radius: 4px;
   margin: 5px 0;
}
.stats-box {
   background-color: #f0f2f6;
   padding: 1rem;
   border-radius: 10px;
   border-left: 4px solid #4CAF50;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_labels():
   """Load model và labels với error handling tốt"""
   try:
       # Tìm file model
       model_files = [
           'emotion_recognition_final.keras',
           'emotion_recognition_final.h5'
       ]
       
       model = None
       for model_file in model_files:
           if os.path.exists(model_file):
               try:
                   model = tf.keras.models.load_model(model_file)
                   st.success(f"✅ Đã load model: {model_file}")
                   break
               except Exception as e:
                   st.warning(f"⚠️ Không thể load {model_file}: {e}")
                   continue
       
       if model is None:
           st.error("❌ Không tìm thấy file model!")
           return None, None, None
       
       # Load labels
       labels_data = None
       if os.path.exists('emotion_labels.json'):
           with open('emotion_labels.json', 'r', encoding='utf-8') as f:
               labels_data = json.load(f)
       else:
           # Default labels
           labels_data = {
               'emotion_labels_en': {
                   '0': 'Angry', '1': 'Disgust', '2': 'Fear', '3': 'Happy',
                   '4': 'Sad', '5': 'Surprise', '6': 'Neutral'
               },
               'emotion_labels_vn': {
                   '0': 'Tức giận', '1': 'Ghê tởm', '2': 'Sợ hãi', '3': 'Vui vẻ',
                   '4': 'Buồn bã', '5': 'Ngạc nhiên', '6': 'Bình thường'
               }
           }
       
       # Load model info
       model_info = None
       if os.path.exists('model_info.json'):
           with open('model_info.json', 'r') as f:
               model_info = json.load(f)
       
       return model, labels_data, model_info
       
   except Exception as e:
       st.error(f"❌ Lỗi khi load model: {e}")
       return None, None, None

def preprocess_image(image):
   """Tiền xử lý ảnh cho model"""
   try:
       if isinstance(image, Image.Image):
           image = np.array(image)
       
       # Convert to grayscale
       if len(image.shape) == 3:
           if image.shape[2] == 3:
               gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
           elif image.shape[2] == 4:
               gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
           else:
               gray = image[:,:,0]
       else:
           gray = image
       
       # Resize to 48x48
       resized = cv2.resize(gray, (48, 48))
       
       # Normalize
       normalized = resized / 255.0
       
       # Reshape for model
       input_image = normalized.reshape(1, 48, 48, 1)
       
       return input_image, resized
   
   except Exception as e:
       st.error(f"Lỗi xử lý ảnh: {e}")
       return None, None

def predict_emotion(model, image, labels_data):
   """Predict cảm xúc từ ảnh"""
   try:
       processed_image, processed_display = preprocess_image(image)
       
       if processed_image is None:
           return None, None, 0, None, None
       
       # Predict
       predictions = model.predict(processed_image, verbose=0)[0]
       
       # Get results
       predicted_class = np.argmax(predictions)
       confidence = predictions[predicted_class]
       
       emotion_en = labels_data['emotion_labels_en'][str(predicted_class)]
       emotion_vn = labels_data['emotion_labels_vn'][str(predicted_class)]
       
       return emotion_en, emotion_vn, confidence, predictions, processed_display
   
   except Exception as e:
       st.error(f"Lỗi dự đoán: {e}")
       return None, None, 0, None, None

def get_emotion_color(emotion_en):
   """Lấy màu theo cảm xúc"""
   colors = {
       'Happy': '#4CAF50',
       'Sad': '#2196F3', 
       'Angry': '#F44336',
       'Fear': '#9C27B0',
       'Surprise': '#FF9800',
       'Disgust': '#795548',
       'Neutral': '#607D8B'
   }
   return colors.get(emotion_en, '#607D8B')

def main():
   # Header
   st.markdown('<h1 class="main-header">🎭 Nhận diện cảm xúc khuôn mặt</h1>', unsafe_allow_html=True)
   st.markdown("---")
   
   # Load model
   with st.spinner("🔄 Đang load model..."):
       model, labels_data, model_info = load_model_and_labels()
   
   if model is None:
       st.stop()
   
   # Sidebar
   st.sidebar.title("⚙️ Thông tin")
   
   if model_info:
       st.sidebar.markdown('<div class="stats-box">', unsafe_allow_html=True)
       st.sidebar.markdown("### 📊 Thông tin Model")
       st.sidebar.write(f"**Độ chính xác:** {model_info['test_accuracy']:.1%}")
       st.sidebar.write(f"**Parameters:** {model_info['total_params']:,}")
       st.sidebar.write(f"**Epochs:** {model_info['epochs_trained']}")
       st.sidebar.markdown('</div>', unsafe_allow_html=True)
   
   st.sidebar.markdown("### 🎭 Các cảm xúc")
   emotions_vn = list(labels_data['emotion_labels_vn'].values())
   emotions_en = list(labels_data['emotion_labels_en'].values())
   
   for emotion_vn, emotion_en in zip(emotions_vn, emotions_en):
       color = get_emotion_color(emotion_en)
       st.sidebar.markdown(f'<span style="color: {color};">● {emotion_vn}</span>', unsafe_allow_html=True)
   
   # App mode
   app_mode = st.sidebar.selectbox(
       "Chọn chế độ:",
       ["📤 Upload ảnh", "📷 Chụp ảnh từ camera"]
   )
   
   # Main content
   if app_mode == "📤 Upload ảnh":
       st.header("📤 Upload ảnh để nhận diện cảm xúc")
       
       uploaded_file = st.file_uploader(
           "Chọn ảnh khuôn mặt...", 
           type=['jpg', 'jpeg', 'png', 'bmp'],
           help="Hỗ trợ các định dạng: JPG, JPEG, PNG, BMP"
       )
       
       if uploaded_file is not None:
           image = Image.open(uploaded_file)
           
           col1, col2, col3 = st.columns([2, 1, 2])
           
           with col1:
               st.subheader("🖼️ Ảnh gốc")
               st.image(image, caption="Ảnh đã upload", use_column_width=True)
           
           with col3:
               st.subheader("🎯 Kết quả nhận diện")
               
               with st.spinner("🔍 Đang phân tích..."):
                   result = predict_emotion(model, image, labels_data)
               
               if result[0] is not None:
                   emotion_en, emotion_vn, confidence, all_predictions, processed_img = result
                   
                   # Hiển thị ảnh đã xử lý
                   st.write("**Ảnh đã xử lý (48x48):**")
                   st.image(processed_img, caption="Input cho model", width=150)
                   
                   # Hiển thị kết quả chính
                   emotion_color = get_emotion_color(emotion_en)
                   st.markdown(f"""
                   <div class="emotion-result" style="background-color: {emotion_color}20; border: 2px solid {emotion_color};">
                       🎭 {emotion_vn}<br>
                       <small>({emotion_en})</small>
                   </div>
                   """, unsafe_allow_html=True)
                   
                   # Thanh độ tin cậy
                   st.markdown(f"**📊 Độ tin cậy:** {confidence:.1%}")
                   st.progress(float(confidence))
                   
                   # Chi tiết dự đoán
                   st.subheader("📊 Chi tiết dự đoán")
                   emotions_list_vn = list(labels_data['emotion_labels_vn'].values())
                   emotions_list_en = list(labels_data['emotion_labels_en'].values())
                   
                   for prob, label_vn, label_en in zip(all_predictions, emotions_list_vn, emotions_list_en):
                       col_name, col_bar, col_val = st.columns([2, 3, 1])
                       
                       emotion_color = get_emotion_color(label_en)
                       
                       with col_name:
                           st.markdown(f'<span style="color: {emotion_color}; font-weight: bold;">{label_vn}</span>', unsafe_allow_html=True)
                       with col_bar:
                           st.progress(float(prob))
                       with col_val:
                           st.write(f"{prob:.1%}")
   
   elif app_mode == "📷 Chụp ảnh từ camera":
       st.header("📷 Chụp ảnh từ camera")
       
       camera_image = st.camera_input("📸 Chụp ảnh khuôn mặt")
       
       if camera_image is not None:
           image = Image.open(camera_image)
           
           col1, col2 = st.columns([1, 1])
           
           with col1:
               st.image(image, caption="Ảnh từ camera", use_column_width=True)
           
           with col2:
               with st.spinner("🔍 Đang phân tích..."):
                   result = predict_emotion(model, image, labels_data)
               
               if result[0] is not None:
                   emotion_en, emotion_vn, confidence = result[:3]
                   
                   emotion_color = get_emotion_color(emotion_en)
                   st.markdown(f"""
                   <div class="emotion-result" style="background-color: {emotion_color}20; border: 2px solid {emotion_color};">
                       🎭 {emotion_vn}<br>
                       <small>({emotion_en})</small>
                   </div>
                   """, unsafe_allow_html=True)
                   
                   st.markdown(f"**📊 Độ tin cậy:** {confidence:.1%}")
                   st.progress(float(confidence))

if __name__ == "__main__":
   main()