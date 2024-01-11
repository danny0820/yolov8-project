from ultralytics import YOLO
import cv2

import concurrent.futures 
import util
from sort.sort import *
from util import read_license_plate
import keyboard
import pytesseract # pip install pytesseract
from v1_1 import grabScreen

mot_tracker = Sort()

# load models
license_plate_detector = YOLO('./number_plate2.pt')


cap = cv2.VideoCapture('./sample4.mp4')

ouo=True
keyboard.add_hotkey("f4", lambda: stop2()) #按下f4結束
def stop2():
    global ouo
    ouo = not ouo
    print(ouo)


def addText(text,x1,y1): # 上車牌文字
    cv2.putText(output, text, (int(x1),int(y1)-20 if int(y1)!=0 else 0),
        cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255,255,0), 2)
    
while ouo:
    flag, frame = cap.read()
    ouo=False if flag == False else True
    if ouo==False:
        break
    # screenshot = grabScreen(region=(0, 200, 800, 900)) #抓螢幕 x y width height
    # frame = np.array(screenshot)# 截圖的轉np
    
    license_plates = license_plate_detector(frame)[0]

    output=license_plates.plot()#顯示license_plate訓練的label+score 
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

        # process license plate
        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
        _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
   
   
        # license_plate_text= read_license_plate(license_plate_crop_gray) 
        license_plate_text=pytesseract.image_to_string(license_plate_crop_thresh) #用google Ocr的方法  //不會用到Util.py 
        print(license_plate_text)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(addText,license_plate_text,x1,y1)
            concurrent.futures.as_completed(None)
        
    cv2.imshow("window",output)
    cv2.waitKey(1)
        
