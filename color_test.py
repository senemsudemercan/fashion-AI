import cv2
from sklearn.cluster import KMeans
import math

def renk_adi(rgb):
    renkler = {
        "siyah": (0, 0, 0),
        "beyaz": (255, 255, 255),
        "bej": (230, 220, 200),
        "gri": (150, 150, 150),
        "kahverengi": (100, 70, 45),
        "yeşil": (80, 120, 80),
        "lacivert": (20, 35, 80),
        "mavi": (60, 120, 200),
        "kırmızı": (180, 40, 40)
    }

    en_yakin = None
    en_kucuk = 999999

    for ad, deger in renkler.items():
        mesafe = math.sqrt(sum((rgb[i] - deger[i]) ** 2 for i in range(3)))
        if mesafe < en_kucuk:
            en_kucuk = mesafe
            en_yakin = ad

    return en_yakin

image = cv2.imread("test.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
small_image = cv2.resize(image, (200, 200))
pixels = small_image.reshape(-1, 3)

kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(pixels)

colors = kmeans.cluster_centers_.astype(int)

renk_isimleri = []

print("Tespit edilen renkler:")
for color in colors:
    ad = renk_adi(color)
    renk_isimleri.append(ad)
    print(color, "->", ad)

uyumlu_setler = [
    {"siyah", "beyaz", "gri"},
    {"bej", "kahverengi", "beyaz"},
    {"bej", "yeşil", "kahverengi"},
    {"lacivert", "beyaz", "gri"},
    {"mavi", "beyaz", "bej"}
]

score = 50

for set_renk in uyumlu_setler:
    ortak = set(renk_isimleri).intersection(set_renk)
    if len(ortak) >= 2:
        score += 20

score = min(score, 100)

print("\nKombin renkleri:", renk_isimleri)
print("Uyum skoru:", score)

if score >= 80:
    print("Yorum: Renk uyumu güçlü.")
elif score >= 60:
    print("Yorum: Kombin genel olarak uyumlu.")
else:
    print("Yorum: Renk geçişleri biraz daha dengelenebilir.")