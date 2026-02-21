import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import cv2 as cv
import json
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'/Users/muhammedhamzakaramanli/OpenCV_project/PHOTO-2025-12-18-00-01-25.jpg'


card = cv.imread(r"path-id")
if card is None:
    print("Hata: kimlik.jpg bulunamadı!")
else:
    gray = cv.cvtColor(card, cv.COLOR_BGR2GRAY)
    blurlu = cv.GaussianBlur(gray, (5, 5), 0)
    canny = cv.Canny(blurlu, 75, 200)

  
    cv.imshow("1.p1", canny)
    cv.waitKey(0)
    cv.destroyAllWindows()

    contours, hierarchy = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv.contourArea, reverse=True)

    kimlik_alan = contours[0]

    
    epsilon = 0.02 * cv.arcLength(kimlik_alan, True)
    number = cv.approxPolyDP(kimlik_alan, epsilon, True)

    if (len(number) == 4):
        print("ID Card Found")

        doc_corners = number.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = doc_corners.sum(axis=1)
        rect[0] = doc_corners[np.argmin(s)]
        rect[2] = doc_corners[np.argmax(s)]
        diff = np.diff(doc_corners, axis=1)
        rect[1] = doc_corners[np.argmin(diff)]
        rect[3] = doc_corners[np.argmax(diff)]
        (tl, tr, br, bl) = rect


        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))

     
        print(f"Yeni görüntü boyutları: {maxWidth}w x {maxHeight}h")


        dst_points = np.array([[0, 0],[maxWidth - 1, 0],[maxWidth - 1, maxHeight - 1],[0, maxHeight - 1]  ], dtype="float32")

        matrix = cv.getPerspectiveTransform(rect, dst_points)

        duz_goruntu = cv.warpPerspective(card, matrix, (maxWidth, maxHeight))
        cv.imwrite("duz_goruntu_cikti.jpg", duz_goruntu)
       
        cv.imshow("Orijinal", card)
        cv.imshow("duz goruntu", duz_goruntu)


        duz_gri = cv.cvtColor(duz_goruntu, cv.COLOR_BGR2GRAY)


        
        roi_tc = duz_gri[96:120, 35:185]
        roi_ad = duz_gri[190:208, 201:255]
        roi_soyad = duz_gri[145 :165 , 201:266]


       
        temiz_tc = cv.adaptiveThreshold(roi_tc, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        temiz_ad = cv.adaptiveThreshold(roi_ad, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        temiz_soyad = cv.adaptiveThreshold(roi_soyad, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)


      
        config = "--psm 6"

        ham_metin_tc = pytesseract.image_to_string(temiz_tc, lang='tur', config=config)
        ham_metin_ad = pytesseract.image_to_string(temiz_ad, lang='tur', config=config)
        ham_metin_soyad = pytesseract.image_to_string(temiz_soyad, lang='tur', config=config)

        
        print(f"Ham TC Metni: {ham_metin_tc}")
        print(f"Ham Ad Metni: {ham_metin_ad}")
        print(f"Ham Soyad Metni: {ham_metin_soyad}")
        



        temiz_veri = {}


        tc_sonuc = re.search(r"\d{11}", ham_metin_tc)
        if tc_sonuc:
            temiz_veri["tc_kimlik_no"] = tc_sonuc.group(0)
        else:
            temiz_veri["tc_kimlik_no"] = "Bulunamadı"

        ad_sonuc = re.search(r"[A-ZÇĞİÖŞÜ]+", ham_metin_ad)
        if ad_sonuc:
            temiz_veri["ad"] = ad_sonuc.group(0)
        else:
            temiz_veri["ad"] = "Bulunamadı"

        soyad_sonuc = re.search(r"[A-ZÇĞİÖŞÜ]+", ham_metin_soyad)
        if soyad_sonuc:
            temiz_veri["soyad"] = soyad_sonuc.group(0)
        else:
            temiz_veri["soyad"] = "Bulunamadı"


        # ensure_ascii=False türkçe karakterler için kulandım
        json_cikti = json.dumps(temiz_veri, indent=2, ensure_ascii=False)

        print(json_cikti)

        
        with open("kimlik_bilgileri.json", "w", encoding="utf-8") as f:
            f.write(json_cikti)

        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        print("ID card not found, Scan your ID card again.")

