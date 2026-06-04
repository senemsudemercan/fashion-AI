import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import torch
torch.classes.__path__ = []

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

@st.cache_resource
def load_model():
    return YOLO("runs/detect/train/weights/best.pt")

model = load_model()

uploaded_file = st.file_uploader("Bir kombin fotoğrafı yükle", type=["jpg", "jpeg", "png"])

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
    crop = cv2.resize(crop, (100, 100))
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

    k = min(3, len(filtered))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=3)
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

def iou(box1, box2):
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2

    ix1 = max(x1, a1)
    iy1 = max(y1, b1)
    ix2 = min(x2, a2)
    iy2 = min(y2, b2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area1 = max(0, x2 - x1) * max(0, y2 - y1)
    area2 = max(0, a2 - a1) * max(0, b2 - b1)

    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def kutulari_temizle(boxes):
    temiz = []

    for box in boxes:
        x1, y1, x2, y2 = box["xyxy"]
        area = (x2 - x1) * (y2 - y1)

        if box["conf"] < 0.35:
            continue

        if area < 2500:
            continue

        ayni_var = False

        for mevcut in temiz:
            if box["class"] == mevcut["class"] and iou(box["xyxy"], mevcut["xyxy"]) > 0.25:
                ayni_var = True
                if box["conf"] > mevcut["conf"]:
                    temiz.remove(mevcut)
                    temiz.append(box)
                break

        if not ayni_var:
            temiz.append(box)

    final = []
    gorulen = {}

    for box in sorted(temiz, key=lambda x: x["conf"], reverse=True):
        cls = box["class"]

        if cls not in gorulen:
            gorulen[cls] = 0

        if gorulen[cls] < 1:
            final.append(box)
            gorulen[cls] += 1

    return final

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    max_width = 800
    if image.width > max_width:
        ratio = max_width / image.width
        image = image.resize((max_width, int(image.height * ratio)))

    image_np = np.array(image)

    st.subheader("📷 Yüklenen Görsel")
    st.image(image, use_container_width=True)

    results = model.predict(image_np, conf=0.35, iou=0.45, verbose=False)

    annotated = results[0].plot()

    st.subheader("🧥 Tespit Edilen Kıyafetler")
    st.image(annotated, use_container_width=True)

    raw_boxes = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls_id]

        raw_boxes.append({
            "xyxy": [x1, y1, x2, y2],
            "class": class_name,
            "conf": conf
        })

    boxes = kutulari_temizle(raw_boxes)

    st.subheader("🎨 Kıyafetlerden Çıkarılan Renkler")

    tum_renkler = []

    if len(boxes) == 0:
        st.warning("Kıyafet tespit edilemedi. Daha net bir fotoğraf deneyin.")
    else:
        for box in boxes:
            x1, y1, x2, y2 = box["xyxy"]
            class_name = box["class"]
            conf = box["conf"]

            crop = image_np[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            renk_rgbleri, renk_adlari = renk_paleti(crop, adet=2)
            tum_renkler.extend(renk_adlari)

            class_display = class_tr.get(class_name, class_name)
            renk_yazisi = " + ".join(renk_adlari)

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
                font-weight:bold;
                ">
                {class_display} → {renk_yazisi} | Güven: %{conf*100:.1f}
                </div>
                """,
                unsafe_allow_html=True
            )

        unique_colors = set(tum_renkler)

        uyumlu_setler = [
            {"siyah", "beyaz", "gri", "füme"},
            {"siyah", "bej", "kahverengi", "taba"},
            {"bej", "kahverengi", "krem", "beyaz"},
            {"lacivert", "mavi", "beyaz", "gri"},
            {"mavi", "yeşil", "turuncu"},
            {"yeşil", "bej", "kahverengi", "krem"},
            {"haki", "bej", "kahverengi", "siyah"},
            {"bordo", "siyah", "gri", "krem"},
            {"turuncu", "mavi", "yeşil"},
            {"pembe", "beyaz", "gri", "bej"},
            {"mor", "siyah", "gri", "lila"}
        ]

        score = 50

        for set_renk in uyumlu_setler:
            ortak = unique_colors.intersection(set_renk)
            if len(ortak) >= 2:
                score += 15

        if len(unique_colors) <= 4:
            score += 10

        if len(unique_colors) >= 7:
            score -= 10

        score = max(0, min(score, 100))

        st.subheader("✨ Uyum Skoru")
        st.progress(score / 100)
        st.write(f"Skor: {score}/100")
        st.write("Algılanan renkler:", ", ".join(sorted(unique_colors)))

        if score >= 80:
            st.success("Renk uyumu güçlü.")
        elif score >= 60:
            st.warning("Kombin genel olarak uyumlu.")
        else:
            st.error("Renk geçişleri biraz daha dengelenebilir.")
            