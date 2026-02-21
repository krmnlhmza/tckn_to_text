Turkish ID Card OCR Evolution

This project explores various Optical Character Recognition (OCR) and computer vision techniques to automate data extraction from Turkish ID cards.

Version 1: Tesseract OCR with Manual ROI
The initial approach utilized Tesseract OCR combined with manual Region of Interest (ROI) selection. Because the system relied on fixed coordinates, it required the ID card to be perfectly aligned within the frame. While it successfully extracted basic text, the lack of flexibility in camera positioning made this method impractical for real-world usage.

Version 2: EasyOCR and Regex-Based Extraction
In the second iteration, the focus shifted to a more dynamic workflow. The system first applies image preprocessing, including perspective transformation and scaling, to normalize the card's appearance. Instead of using hardcoded coordinates, the entire card is scanned using EasyOCR. The resulting raw text is then filtered using specific Regular Expressions (Regex) to accurately extract the TR ID number, Name, and Surname. This version provides much higher flexibility and better text recognition compared to the manual coordinate method.

Version 3: YOLOv8 Object Detection Attempt
The third phase aimed to further increase the success rate by using YOLOv8 to detect and crop specific fields (like name or ID number) before passing them to the OCR engine. The goal was to eliminate background noise and focus only on relevant areas. However, obtaining a sufficiently large and diverse dataset of Turkish ID cards for labeling on Roboflow was not possible due to privacy concerns. Consequently, the YOLOv8 model did not reach the required accuracy levels and performed unsuccessfully compared to the Version 2 approach.

Requirements
Python 3.x

OpenCV

Pytesseract

EasyOCR

Ultralytics (YOLOv8)

Numpy / Pandas
