from flask import Flask, render_template, Response
from camera import VideoCamera

path = 'videos/helmet_Nohelmet.mp4'
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
vid = cv2.Videocapture(path)
while True:
	try:
		_,frame =vid.read()
		frame = cv2.resize(image, (640, 400), fx=0.1, fy=0.1)
		get_frame()
		face_Detection(frame)
		cv2.imshow("Helmet Detection",frame)
		if cv2.waitkey(1)==13:
			break
		vid.release()
		cv.destroyAllWindows()
	except:
		pass