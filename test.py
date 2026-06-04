from ultralytics import YOLO

# modeli yükle
model = YOLO("yolov8n.pt")

# resmi analiz et
results = model("test.jpeg")

# sonucu göster
results[0].show()

print("Analiz tamamlandı!") 