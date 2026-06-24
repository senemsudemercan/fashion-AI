from ultralytics import YOLO
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()

    # Modeli yükle
    model = YOLO("yolov8n.pt")

    # Eğitimi başlat
    model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,  # GPU kullan

        # Augmentation
        degrees=10,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        mosaic=1.0,
        mixup=0.1,

        workers=0  # Windows hatasını önler
    )

    print("Eğitim tamamlandı!")