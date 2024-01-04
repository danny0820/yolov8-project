from ultralytics import YOLO
import cv2

import concurrent.futures 
import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv
import keyboard

from v1_1 import grabScreen
results = {}
#窩不知道
mot_tracker = Sort()

# load models
# coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./number_plate.pt')

# load video
# cap = cv2.VideoCapture('./sample.mp4')

vehicles = [2, 3, 5, 7]

def stop():
    global ret
    ret = not ret
    print(ret)

ouo=True
keyboard.add_hotkey("f4", lambda: stop())
def stop2():
    global ouo
    ouo = not ouo
    print(ouo)


keyboard.add_hotkey("q", lambda: stop2())
# read frames
frame_nmr = -1
ret = True
def addText(output,text,x1,y1):
    cv2.putText(output, text, (int(x1) if int(x1)!=0 else 0,int(y1)-20 if int(y1)!=0 else 0),
        cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255,255,255), 2)
    cv2.imshow("window",output)
    cv2.waitKey(1)
while ouo:
    # frame_nmr += 1
    if ret:
        # flag, frame = cap.read()
        # ouo=False if flag == False else True
        # detect license plates
        
        screenshot = grabScreen(region=(0, 200, 800, 900))
        frame = np.array(screenshot)
        license_plates = license_plate_detector(frame)[0]
        # print(license_plates.plot())
        output=license_plates.plot()
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            # xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate)

            # if car_id != -1:

                # crop license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

            # process license plate
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

            # read license plate number
            # license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
        
            license_plate_text= read_license_plate(license_plate_crop_thresh)
            print(license_plate_text)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.submit(addText,frame,license_plate_text,x1,y1)
                concurrent.futures.as_completed(None)
                # if license_plate_text is not None:
                #     results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                #                                   'license_plate': {'bbox': [x1, y1, x2, y2],
                #                                                     'text': license_plate_text,
                #                                                     'bbox_score': score,
                #                                                     'text_score': license_plate_text_score}}
            # cv2.putText(output, 'This is Jimmy.', (int(x1) if int(x1)==0 else 0,int(y1)-20 if int(y1)==0 else 0),
            #         cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0,255,0), 2)
        cv2.imshow("window",output)
        cv2.waitKey(1)
        
# write results
# write_csv(results, './test.csv')