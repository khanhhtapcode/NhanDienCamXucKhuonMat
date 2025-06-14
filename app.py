import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
import os

# Cấu hình trang
st.set_page_config(
   page_title=" Nhận diện cảm xúc khuôn mặt",
   page_icon="🎭",
   layout="wide"
)

# Thử import TensorFlow
TF_AVAILABLE = False
try:
   import warnings
   warnings.filterwarnings('ignore')
   
   import os
   os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
   
   # Import tensorflow directly
   import tensorflow as tf
   
   TF_AVAILABLE = True
   st.success("✅ TensorFlow đã sẵn sàng!")
       
except Exception as e:
   st.error(f"TensorFlow error: {e}")
   TF_AVAILABLE = False

# Custom CSS
st.markdown("""
<style>
.main-header {
   font-size: 3rem;
   color: #FF6B6B;
   text-align: center;
   margin-bottom: 2rem;
}
.emotion-result {
   font-size: 3rem;
   font-weight: bold;
   text-align: center;
   padding: 2rem;
   border-radius: 20px;
   margin: 2rem 0;
   box-shadow: 0 6px 12px rgba(0,0,0,0.15);
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_labels():
   """Load model và labels với error handling tốt"""
   if not TF_AVAILABLE:
       return None, None, None
   
   try:
       model = None
       
       # Thử load model từ các file có sẵn
       if os.path.exists('emotion_recognition_final.h5'):
           model = tf.keras.models.load_model('emotion_recognition_final.h5')
           st.success("✅ Đã load model từ file .h5")
       elif os.path.exists('emotion_recognition_final.keras'):
           model = tf.keras.models.load_model('emotion_recognition_final.keras')
           st.success("✅ Đã load model từ file .keras")
       else:
           st.error("❌ Không tìm thấy file model!")
           return None, None, None
       
       # Load labels
       labels_data = None
       if os.path.exists('emotion_labels.json'):
           with open('emotion_labels.json', 'r', encoding='utf-8') as f:
               labels_data = json.load(f)
       else:
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
       st.error(f"Lỗi khi load model: {e}")
       return None, None, None

def preprocess_image(image):
   """Tiền xử lý ảnh cho model"""
   if isinstance(image, Image.Image):
       image = np.array(image)
   
   if len(image.shape) == 3:
       if image.shape[2] == 3:
           gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
       elif image.shape[2] == 4:
           gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
       else:
           gray = image[:,:,0]
   else:
       gray = image
   
   resized = cv2.resize(gray, (48, 48))
   normalized = resized / 255.0
   input_image = normalized.reshape(1, 48, 48, 1)
   
   return input_image

def predict_emotion(model, image, labels_data):
   """Predict cảm xúc từ ảnh"""
   processed_image = preprocess_image(image)
   
   predictions = model.predict(processed_image, verbose=0)[0]
   
   predicted_class = np.argmax(predictions)
   
   emotion_en = labels_data['emotion_labels_en'][str(predicted_class)]
   emotion_vn = labels_data['emotion_labels_vn'][str(predicted_class)]
   
   return emotion_en, emotion_vn

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

def get_emotion_emoji(emotion_en):
   """Lấy emoji theo cảm xúc"""
   emojis = {
       'Happy': '😊',
       'Sad': '😢',
       'Angry': '😠',
       'Fear': '😨',
       'Surprise': '😲',
       'Disgust': '🤢',
       'Neutral': '😐'
   }
   return emojis.get(emotion_en, '🎭')

def main():
   # Header
   st.markdown('<h1 class="main-header">🎭 Nhận diện cảm xúc khuôn mặt</h1>', unsafe_allow_html=True)
   st.markdown("---")
   
   # Load model
   model, labels_data, model_info = load_model_and_labels()
   
   if model is None:
       st.stop()
   
   # Sidebar
   st.sidebar.title("⚙️ Thông tin")
   
   if model_info:
       st.sidebar.markdown("### 📊 Thông tin Model")
       st.sidebar.write(f"**Độ chính xác:** {model_info['test_accuracy']:.3f}")
       st.sidebar.write(f"**Parameters:** {model_info['total_params']:,}")
   
   st.sidebar.markdown("### 🎭 Các cảm xúc")
   emotions_vn = list(labels_data['emotion_labels_vn'].values())
   emotions_en = list(labels_data['emotion_labels_en'].values())
   
   for emotion_vn, emotion_en in zip(emotions_vn, emotions_en):
       emoji = get_emotion_emoji(emotion_en)
       st.sidebar.write(f"{emoji} {emotion_vn}")
   
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
           type=['jpg', 'jpeg', 'png', 'bmp']
       )
       
       if uploaded_file is not None:
           image = Image.open(uploaded_file)
           
           col1, col2 = st.columns([1, 1])
           
           with col1:
               st.image(image, caption="Ảnh đã upload", use_column_width=True)
           
           with col2:
               with st.spinner("🔍 Đang phân tích..."):
                   emotion_en, emotion_vn = predict_emotion(model, image, labels_data)
               
               # Hiển thị kết quả
               emotion_color = get_emotion_color(emotion_en)
               emotion_emoji = get_emotion_emoji(emotion_en)
               
               st.markdown(f"""
               <div class="emotion-result" style="background-color: {emotion_color}20; border: 3px solid {emotion_color};">
                   {emotion_emoji} {emotion_vn}<br>
                   <small style="font-size: 1.5rem;">({emotion_en})</small>
               </div>
               """, unsafe_allow_html=True)
   
   elif app_mode == "📷 Chụp ảnh từ camera":
       st.header("📷 Chụp ảnh từ camera")
       
       camera_image = st.camera_input("📸 Chụp ảnh")
       
       if camera_image is not None:
           image = Image.open(camera_image)
           
           col1, col2 = st.columns([1, 1])
           
           with col1:
               st.image(image, caption="Ảnh từ camera", use_column_width=True)
           
           with col2:
               with st.spinner("🔍 Đang phân tích..."):
                   emotion_en, emotion_vn = predict_emotion(model, image, labels_data)
               
               emotion_color = get_emotion_color(emotion_en)
               emotion_emoji = get_emotion_emoji(emotion_en)
               
               st.markdown(f"""
               <div class="emotion-result" style="background-color: {emotion_color}20; border: 3px solid {emotion_color};">
                   {emotion_emoji} {emotion_vn}<br>
                   <small style="font-size: 1.5rem;">({emotion_en})</small>
               </div>
               """, unsafe_allow_html=True)

if __name__ == "__main__":
   main()