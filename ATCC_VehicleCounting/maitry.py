import cv2
from get_number import*
pat = 'data/plate_img'
IDS = []

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
def Roi(frame,ay,by,ax,bx):
    roi = frame[ay:by, ax:bx]
    cv2.line(frame, (ax, ay), (bx, ay), (0, 255, 255), 2)
    cv2.line(frame, (ax, by), (bx, by), (0, 255, 255), 2)
    cv2.line(frame, (ax, ay), (ax, by), (0, 255, 255), 2)
    cv2.line(frame, (bx, ay), (bx, by), (0, 255, 255), 2)
    return roi

def Fps(img, t):
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - t)
    cv2.putText(img, 'FPS: ' + str(int(fps)), (50, 50), font, 1.6, (255, 0, 255), 2)

def crop(roi, x, y, w, h, ids):
    area = int(w*h)
    num = str(1) + '.jpg'
    img = roi[y-3:y+h+3,x-3:x+w+3]
    img = cv2.resize(img,(img.shape[1]*4,img.shape[0]*4))
    if ids not in IDS and (area > 320) and (w > h*2) :
        cv2.imwrite('data/plate_img/'+num,img)

class Tracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        self.id_count = 0

    def update(self, x,y,w,h):
        objects_bbs_ids = []
        cx = (x + x + w) // 2
        cy = (y + y + h) // 2
        same_object_detected = False
        if same_object_detected is False:
            self.center_points[self.id_count] = (cx, cy)
            objects_bbs_ids.append([x, y, w, h, self.id_count])
            self.id_count += 1
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center
        self.center_points = new_center_points.copy()
        return objects_bbs_ids


tracking = Tracker()
def Tracking(frame,roi,boxes):
    if len(boxes) > 0:
        for box in boxes:
            x,y,w,h = box[0],box[1],box[2],box[3]
            boxes_ids = tracking.update(x,y,w,h)
            for box_id in boxes_ids:
                x, y, w, h, ids = box_id
                crop(roi, x, y, w, h, ids)
                Noplate_extract('data/plate_img/1.jpg')
                remove('data/plate_img/1.jpg')
                IDS.append(ids)

