import cv2 as cv
import numpy as np
import concurrent.futures  # thread
import time
import functools
# import win32api
# import win32con
import ctypes  # mouse 2
from v1_1 import grabScreen
from ultralytics import YOLO
# import keyboard  
# import mss #use this more better ,del v1_1
# trainingInTerminal: yolo detect train data=Customized.yaml model=yolov8n.pt epochs=100 imgsz=640
prev_time = 0
flag = True
def stop():
    global flag
    flag = not flag
    print(flag)

# keyboard.add_hotkey("f4", lambda: stop())
timeset = 0.008
print("start")

model = YOLO('C:/Users/Danny/Documents/GitHub/yolov8-segmentation/runs/detect/train/weights/best.pt')

# def movement(results):
#     Xlist = int((640+(int(results[0][0])+int(results[0][2]))/2-960)/1.7)
#     Ylist = int((340+(int(results[0][1])+int(results[0][3]))/2-540)/1.7)
#     win32api.mouse_event(win32con.MOUSEEVENTF_MOVE,
#                          Xlist, Ylist-2, 0, 0)  # 有 X,Y會被x1.5倍

while True:
    if flag==True:
        screenshot = grabScreen(region=(640, 360, 1280, 720))
        array_to_image = np.array(screenshot)

        results = model(array_to_image)
        # with concurrent.futures.ThreadPoolExecutor() as executor:
            # executor.submit(movement, results[0].boxes.xyxy.tolist())
            # concurrent.futures.as_completed(None)
    output = results[0].plot()
    cv.putText(output, 'fps: ' + str(int(1 / (time.time() - prev_time))), (20, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv.LINE_AA)
    cv.imshow("win", output)
    prev_time = time.time()
    if cv.waitKey(1) & 0xFF == 27:
        break
