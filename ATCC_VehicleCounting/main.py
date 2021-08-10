#from flask import Flask, render_template, Response
#from camera import VideoCamera
import cv2
from maitry import *
from YOLO import Yolo

path = 'street.mp4'

#Making custom region of interest
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
cap = cv2.VideoCapture(path)
s, img = cap.read()
img = cv2.resize(img,(600,420))
Box = cv2.selectROI('FR', img)
x, y, w, h = int(Box[0]), int(Box[1]), int(Box[2]), int(Box[3])
ax, bx, ay, by = x, x + w, y, y + h
cv2.waitKey(0)
cap.release()
cv2.destroyWindow('FR')
vid = cv2.VideoCapture(path)
vid.set(3,600)
vid.set(4,420)
while True:
    try:
        t = cv2.getTickCount()
        _, frame = vid.read()
        frame = cv2.resize(frame, (600, 420))
        roi = Roi(frame, ay, by, ax, bx)
        print(roi.shape)
        boxes = Yolo(roi)
        Tracking(frame, roi, boxes)  # tracking and cropping is here
        Fps(frame, t)
        cv2.imshow('Vehicle Counting',frame)
        if cv2.waitKey(1) == 13:
            break
        vid.release()
        cv2.destroyAllWindows()
    except:
        pass

