import cv2
import numpy as np
from faceDetect import face_Detection

net = cv2.dnn.readNetFromDarknet('helmet2/yolov4_tiny_train.cfg','helmet2/yolov4_tiny_train_best.weights')
classes = ['helmet','Nohlmt']
col =(0,255,255)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

    def get_frame(self):
        frame = cv2.resize(image, (640, 400), fx=0.1, fy=0.1)
        while True:
            face_Detection(frame)
            ht, wt, _ = frame.shape
            blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
            net.setInput(blob)
            last_layer = net.getUnconnectedOutLayersNames()
            layer_out = net.forward(last_layer)
            boxes = []
            confidences = []
            cls_ids = []
            for output in layer_out:
                for detection in output:
                    score = detection[5:]
                    clsid = np.argmax(score)
                    conf = score[clsid]
                    if conf > 0.32:
                        centreX = int(detection[0] * wt)
                        centreY = int(detection[1] * ht)
                        w = int(detection[2] * wt)
                        h = int(detection[3] * ht)
                        x = int(centreX - w / 2)
                        y = int(centreY - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append((float(conf)))
                        cls_ids.append(clsid)
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, .3, .2)
            try:
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    label = classes[i]
                    if label == 'helmet':
                        cv2.rectangle(frame, (x, y), (x + 54, y - 12), col, cv2.FILLED)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), col, 2)
                        cv2.putText(frame, label, (x, y - 2), font, 0.6, color=(255, 0, 0))
                        
                   
            except:
                pass
                
