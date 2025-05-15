import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, ttk


label_mapping = {
    "Persian_cat": "Mèo Ba Tư",
    "Siamese_cat": "Mèo Xiêm",
    "Egyptian_cat": "Mèo Ai Cập",
    "tiger_cat": "Mèo vằn",
    "tabby": "Mèo mướp",
    "lynx": "Mèo rừng",
    "kitten": "Mèo con",
    "Chihuahua": "Chó Chihuahua",
    "golden_retriever": "Chó Golden Retriever",
    "Labrador_retriever": "Chó Labrador",
    "German_shepherd": "Chó Becgie Đức",
    "poodle": "Chó Poodle",
    "bulldog": "Chó Bulldog"
}

def load_model():
    """Tải mô hình ResNet50 đã được huấn luyện sẵn trên ImageNet."""
    return ResNet50(weights='imagenet')

def preprocess_image(image_path):
    """Tiền xử lý ảnh để phù hợp với ResNet50."""
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_image(model, image_path):
    """Dự đoán nhãn của ảnh đầu vào sử dụng ResNet50."""
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    return decoded_predictions[0]

def select_image():
    """Mở hộp thoại để chọn ảnh từ máy tính."""
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")])
    if file_path:
        display_image(file_path)
        btn_analyze.config(state=tk.NORMAL)
        lbl_result.config(text="")
        global selected_image
        selected_image = file_path

def display_image(image_path):
    """Hiển thị ảnh lên giao diện Tkinter."""
    img = Image.open(image_path).resize((300, 300))
    img = ImageTk.PhotoImage(img)
    lbl_image.config(image=img)
    lbl_image.image = img

def analyze_image():
    """Phân tích ảnh bằng mô hình ResNet50 và hiển thị kết quả bằng tiếng Việt."""
    label, score = predict_image(model, selected_image)[1:]  
    vietnamese_label = label_mapping.get(label, label)  # Lấy tên tiếng Việt nếu có
    result_text = f"Dự đoán: {vietnamese_label} (Độ chính xác: {score:.2%})"  
    lbl_result.config(text=result_text)

# Khởi tạo mô hình
model = load_model()

# Tạo giao diện Tkinter
root = tk.Tk()
root.title("Nhận diện ảnh với ResNet50")
root.geometry("400x500")
root.configure(bg="#f0f0f0")

frame = ttk.Frame(root, padding=10)
frame.pack(expand=True)

btn_select = ttk.Button(frame, text="Chọn ảnh", command=select_image)
btn_select.pack(pady=5)

lbl_image = ttk.Label(frame)
lbl_image.pack(pady=5)

btn_analyze = ttk.Button(frame, text="Phân tích", command=analyze_image, state=tk.DISABLED)
btn_analyze.pack(pady=5)

lbl_result = ttk.Label(frame, text="", font=("Arial", 12), background="#f0f0f0", anchor="center")
lbl_result.pack(pady=10)

root.mainloop()
