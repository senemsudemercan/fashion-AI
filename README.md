# Fashion Harmony AI

## Proje Hakkında

Bu proje, kullanıcı tarafından yüklenen kombin fotoğraflarındaki kıyafetleri tespit eden, kıyafetlerin baskın renklerini belirleyen ve kombinin genel renk uyumunu değerlendiren bir yapay zeka uygulamasıdır.

Proje kapsamında nesne tespiti için YOLOv8 modeli kullanılmıştır. Tespit edilen kıyafetlerden baskın renklerin elde edilmesi amacıyla K-Means kümeleme algoritmasından yararlanılmıştır. Elde edilen renkler belirli kurallar çerçevesinde analiz edilerek kombin için bir uyum skoru oluşturulmuştur.

## Kullanılan Teknolojiler

* Python
* YOLOv8
* Streamlit
* OpenCV
* NumPy
* Scikit-Learn
* Pillow

## Veri Seti

Çalışmada Roboflow platformundan elde edilen moda veri seti kullanılmıştır. Veri seti toplam 1711 görüntüden oluşmaktadır ve YOLOv8 formatında etiketlenmiştir.

Veri setinde aşağıdaki sınıflar bulunmaktadır:

* bag
* belt
* boots
* dress
* footwear
* headwear
* outer
* pants
* scarf-tie
* shorts
* skirt
* sunglasses
* top

## Model Eğitimi

Model eğitimi sırasında YOLOv8n mimarisi kullanılmıştır.

Eğitim parametreleri:

* Epoch sayısı: 100
* Görüntü boyutu: 640x640
* Batch boyutu: 16
* Kullanılan donanım: NVIDIA RTX 4060 GPU

Modelin genelleme başarısını artırmak amacıyla veri artırma teknikleri kullanılmıştır.

Uygulanan veri artırma yöntemleri:

* Döndürme (Rotation)
* Ölçeklendirme (Scaling)
* Kaydırma (Translation)
* Yatay çevirme (Horizontal Flip)
* HSV dönüşümleri
* Mosaic
* MixUp

## Uygulamanın Çalışma Prensibi

1. Kullanıcı sisteme bir kombin fotoğrafı yükler.
2. YOLOv8 modeli fotoğraftaki kıyafetleri tespit eder.
3. Her kıyafet bölgesi ayrı olarak işlenir.
4. K-Means algoritması ile kıyafetin baskın renkleri belirlenir.
5. Elde edilen renkler önceden tanımlanan renk grupları ile karşılaştırılır.
6. Kombin için 0 ile 100 arasında bir uyum skoru oluşturulur.
7. Sonuçlar kullanıcıya arayüz üzerinden gösterilir.

## Kurulum

Gerekli kütüphanelerin yüklenmesi:

```bash
pip install ultralytics streamlit opencv-python scikit-learn pillow numpy
```

## Modelin Eğitilmesi

```bash
python train_model.py
```

## Uygulamanın Çalıştırılması

```bash
streamlit run app.py
```

veya

```bash
py -3.12 -m streamlit run app.py
```

## Proje Çıktıları

Model eğitimi tamamlandıktan sonra aşağıdaki dosyalar oluşturulmaktadır:

* best.pt
* results.png
* confusion_matrix.png
* PR_curve.png
* F1_curve.png

Eğitim sonucunda elde edilen en iyi model dosyası aşağıdaki dizinde saklanmaktadır:

```text
runs/detect/train/weights/best.pt
```

## Sonuç

Bu çalışmada görüntü işleme ve derin öğrenme yöntemleri kullanılarak kıyafet analizi gerçekleştiren bir sistem geliştirilmiştir. Sistem, kullanıcıların kombinlerindeki renk uyumunu değerlendirmelerine yardımcı olmayı amaçlamaktadır. Gelecek çalışmalarda aksesuar önerisi, mevsime uygun kombin önerisi ve kişiselleştirilmiş stil önerileri gibi özelliklerin sisteme eklenmesi planlanmaktadır.
