import cv2
import mediapipe as mp      # google face_detection module
fd = mp.solutions.face_detection
detect = fd.FaceDetection(min_detection_confidence=0.25)
draw = mp.solutions.drawing_utils
col =(0,255,255)
def face_Detection(frame):
    ht, wd, _ = frame.shape
    #*********Processing rgb images**************************
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    output = detect.process(rgb)
    if output.detections:
        for id,det in enumerate(output.detections):
            score = det.score[0]
            bbox = det.location_data.relative_bounding_box
            x,y,w,h = int(bbox.xmin*wd),int(bbox.ymin*ht),int(bbox.width*wd),int(bbox.height*ht)
            cv2.rectangle(frame, (x, y), (x + w, y + h), col, 1)
            cv2.rectangle(frame, (x, y), (x+68, y-13), col,cv2.FILLED)
            cv2.putText(frame,f'NoHelmet',(x,y-2),cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6,color=(0,0,255))
    #***********processing completed*************************

