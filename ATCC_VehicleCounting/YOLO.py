import numpy as np
import cv2

classes = 'NoPlate'

net = cv2.dnn.readNetFromDarknet('yolov4_tiny_training.cfg','yolov4_tiny_training_best.weights')
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

def Yolo(frame):
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
            if conf > 0.25:
                centreX = int(detection[0] * wt)
                centreY = int(detection[1] * ht)
                w = int(detection[2] * wt)
                h = int(detection[3] * ht)
                x = int(centreX - w / 2)
                y = int(centreY - h / 2)
                boxes.append([x, y, w, h])
                confidences.append((float(conf)))
                cls_ids.append(clsid)
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, .25, .1)
            color = (0, 255, 0)
            try:
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    label = classes
                    cv2.rectangle(frame, (x-5, y-5), (x + w+5, y + h+5), color, 1)
                    cv2.putText(frame, label, (x, y - 12), font, 0.8, color, 1)

            except:
                pass
    return boxes