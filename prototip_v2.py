
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import cv2 as cv
import json
import pytesseract
import easyocr


card = cv.imread(r"path-id")

if card is None:
    print("image could not be loaded!")
else:

    gray = cv.cvtColor(card, cv.COLOR_BGR2GRAY)
    blurlu = cv.GaussianBlur(gray, (5, 5), 0)
    canny = cv.Canny(blurlu, 75, 200)


    contours, hierarchy = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    kimlik_alan = contours[0]


    epsilon = 0.02 * cv.arcLength(kimlik_alan, True)
    number = cv.approxPolyDP(kimlik_alan, epsilon, True)

    if len(number) == 4:
        print("ID Card Found!")

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

        dst_points = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        matrix = cv.getPerspectiveTransform(rect, dst_points)

        flat_image = cv.warpPerspective(card, matrix, (maxWidth, maxHeight))

        cv.imwrite("flat_image.jpg", flat_image)
        cv.imshow("flat_image(EasyOCR'a Gidecek)", flat_image)


        reader = easyocr.Reader(['tr', 'en'], gpu=True)

        print("Scanning image, please wait...")

        results = reader.readtext(flat_image, detail=0, paragraph=True)

        all_metin = " ".join(results)

        print(f"\n[TEXT READ]:\n{all_metin}\n")


        safeData = {}


        tc_match = re.search(r"\d{11}", all_metin)
        if tc_match:
            safeData["tc_kimlik_no"] = tc_match.group(0)
        else:
            safeData["tc_kimlik_no"] = "not found"


        soyad_match = re.search(r"Surname\s+([A-ZÇĞİÖŞÜ]+)", all_metin, re.IGNORECASE)

        if soyad_match:
            safeData["soyad"] = soyad_match.group(1)
        else:

            spare_soyad = re.search(r"(?:Soyadı|Soyodi)\s+([A-ZÇĞİÖŞÜ]+)", all_metin, re.IGNORECASE)
            if spare_soyad:
                safeData["soyad"] = spare_soyad.group(1)
            else:
                safeData["soyad"] = "not found"


        ad_match = re.search(r"Name\(s\)\s+(.*?)\s+(?:Doqum|Doğum|Date)", all_metin, re.IGNORECASE)

        if ad_match:
            first_name = ad_match.group(1)

            safeData["ad"] =  first_name.strip()
        else:
            safeData["ad"] = "Not found"


        json_output = json.dumps(safeData, indent=2, ensure_ascii=False)

        print("--- (JSON) ---")
        print(json_output)

        with open("kimlik_information_v2.json", "w", encoding="utf-8") as f:
            f.write(json_output)

        print("\n'kimlik_information_v2.json' saved to file.")

        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        print("eror^_last")