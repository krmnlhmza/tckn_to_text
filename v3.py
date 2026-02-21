import cv2
from torch.utils.bundled_inputs import augment_model_with_bundled_inputs
from ultralytics import YOLO
import easyocr
import matplotlib.pyplot as plt


resim_yolu = "test_kimlik.jpg"
model_yolu = "best.pt"


print("Yapay Zeka ve OCR yükleniyor... (Biraz bekletebilir)")
model = YOLO(model_yolu)
reader = easyocr.Reader(['tr'], gpu=False)  # GPU varsa True 


img = cv2.imread(resim_yolu)
if img is None:
    print("HATA: Resim bulunamadı! Dosya adını kontrol et.")
    exit()


results = model(img)

print("-" * 30)
print("SONUÇLAR:")


for result in results:
    for box in result.boxes:
        
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        cls = int(box.cls[0])
        sinif_ismi = model.names[cls]

        
        if conf < 0.4:
            continue

       
        h, w, _ = img.shape
        pad = 5
        y1_c = max(0, y1 - pad)
        y2_c = min(h, y2 + pad)
        x1_c = max(0, x1 - pad)
        x2_c = min(w, x2 + pad)

        kesilen_parca = img[y1_c:y2_c, x1_c:x2_c]

        
        okunan_metin_listesi = reader.readtext(kesilen_parca, detail=0)
        okunan_metin = " ".join(okunan_metin_listesi)

        print(f"✅ {sinif_ismi.upper()}: {okunan_metin} (Güven: {conf:.2f})")

        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{sinif_ismi}: {okunan_metin}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
