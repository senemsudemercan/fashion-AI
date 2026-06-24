import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans
import math

st.set_page_config(page_title="Fashion Harmony AI", layout="centered")

st.title("👗 Fashion Harmony AI")
st.write("Kıyafetleri tespit eder, renklerini çıkarır ve kombin uyumunu analiz eder.")

model = YOLO("runs/detect/train/weights/best.pt")

uploaded_file = st.file_uploader(
    "Bir kombin fotoğrafı yükle",
    type=["jpg", "jpeg", "png"]
)

class_tr = {
    "bag": "Çanta",
    "belt": "Kemer",
    "boots": "Bot",
    "dress": "Elbise",
    "footwear": "Ayakkabı",
    "headwear": "Baş aksesuarı",
    "outer": "Dış giyim",
    "pants": "Pantolon",
    "scarf-tie": "Atkı/Kravat",
    "shorts": "Şort",
    "skirt": "Etek",
    "sunglasses": "Gözlük",
    "top": "Üst giyim"
}

renkler = {
    "siyah": (20, 20, 20),
    "beyaz": (245, 245, 245),
    "krem": (245, 235, 210),
    "bej": (210, 190, 160),
    "gri": (135, 135, 135),
    "füme": (70, 75, 80),
    "kahverengi": (105, 70, 40),
    "taba": (160, 100, 55),
    "lacivert": (20, 35, 90),
    "mavi": (40, 90, 190),
    "açık mavi": (130, 180, 230),
    "yeşil": (30, 120, 70),
    "haki": (90, 100, 60),
    "kırmızı": (185, 40, 40),
    "bordo": (95, 10, 35),
    "turuncu": (230, 110, 25),
    "sarı": (230, 200, 50),
    "pembe": (220, 120, 170),
    "mor": (120, 70, 160),
    "lila": (180, 150, 210)
}

def renk_adi(rgb):
    en_yakin = None
    en_kucuk = 999999

    for ad, deger in renkler.items():
        mesafe = math.sqrt(sum((int(rgb[i]) - deger[i]) ** 2 for i in range(3)))

        if mesafe < en_kucuk:
            en_kucuk = mesafe
            en_yakin = ad

    return en_yakin


def renk_paleti(crop, adet=2):
    crop = cv2.resize(crop, (160, 160))
    pixels = crop.reshape(-1, 3)

    filtered = []

    for p in pixels:
        r, g, b = map(int, p)
        brightness = (r + g + b) / 3

        if 25 < brightness < 245:
            filtered.append(p)

    if len(filtered) < 50:
        filtered = pixels

    filtered = np.array(filtered)

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(filtered)

    centers = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    counts = np.bincount(labels)

    order = np.argsort(counts)[::-1]

    secilenler = []
    secilen_adlar = []

    for idx in order:
        color = centers[idx]
        ad = renk_adi(color)

        if ad not in secilen_adlar:
            secilen_adlar.append(ad)
            secilenler.append(color)

        if len(secilenler) == adet:
            break

    return secilenler, secilen_adlar


def text_color(rgb):
    r, g, b = map(int, rgb)
    brightness = (r * 299 + g * 587 + b * 114) / 1000

    return "black" if brightness > 160 else "white"


if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.subheader("📷 Yüklenen Görsel")
    st.image(image, use_container_width=True)

    results = model.predict(image_np, conf=0.25, iou=0.45)

    annotated = results[0].plot()

    st.subheader("🧥 Tespit Edilen Kıyafetler")
    st.image(annotated, use_container_width=True)

    tum_renkler = []

    st.subheader("🎨 Kıyafetlerden Çıkarılan Renkler")

    for box in results[0].boxes:

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        class_name = model.names[cls_id]
        class_display = class_tr.get(class_name, class_name)

        crop = image_np[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        renk_rgbleri, renk_adlari = renk_paleti(crop, adet=2)
        tum_renkler.extend(renk_adlari)

        main_color = renk_rgbleri[0]
        hex_color = '#%02x%02x%02x' % tuple(main_color)
        font_color = text_color(main_color)

        st.markdown(
            f"""
            <div style="
            background-color:{hex_color};
            padding:15px;
            border-radius:12px;
            margin:8px 0;
            color:{font_color};
            font-weight:bold;">
            {class_display} → {' + '.join(renk_adlari)}
            | Güven: %{conf*100:.1f}
            </div>
            """,
            unsafe_allow_html=True
        )

    unique_colors = set(tum_renkler)

    score = 50

    notr_renkler = {
        "siyah", "beyaz", "gri", "füme",
        "bej", "krem", "kahverengi", "taba"
    }

    score += len(unique_colors.intersection(notr_renkler)) * 5

    tamamlayici = [
        {"mavi", "turuncu"},
        {"mor", "sarı"},
        {"kırmızı", "yeşil"},
        {"lacivert", "bej"},
        {"bordo", "krem"},
        {"siyah", "beyaz"}
    ]

    for cift in tamamlayici:
        if cift.issubset(unique_colors):
            score += 15

    if len(unique_colors) == 1:
        score += 25
    elif len(unique_colors) == 2:
        score += 20
    elif len(unique_colors) == 3:
        score += 15
    elif len(unique_colors) == 4:
        score += 10
    elif len(unique_colors) >= 6:
        score -= 15

    canli_renkler = {
        "kırmızı", "turuncu",
        "sarı", "pembe", "mor"
    }

    if len(unique_colors.intersection(canli_renkler)) >= 3:
        score -= 10

    score = max(0, min(score, 100))

    st.subheader("✨ Uyum Skoru")
    st.progress(score / 100)
    st.write(f"Skor: {score}/100")

    st.write(
        "Algılanan renkler:",
        ", ".join(sorted(unique_colors))
    )

    if score >= 80:
        st.success("Renk uyumu güçlü.")
    elif score >= 60:
        st.warning("Kombin genel olarak uyumlu.")
    else:
        st.error("Renk geçişleri biraz daha dengelenebilir.")
