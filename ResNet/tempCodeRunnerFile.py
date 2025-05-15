
model = load_model()

# Tạo giao diện Tkinter
root = tk.Tk()
root.title("Nhận diện ảnh với ResNet50")
root.geometry("400x500")
root.configure(bg="#f0f0f0")

frame = ttk.Frame(root, padding=10)