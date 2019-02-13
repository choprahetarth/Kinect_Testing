import freenect
import cv2
import numpy as np 
import os

#### include the haar-cascade for frontal face recognition ########
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

#### invite the rgb streams from kinect ###########################
def get_video():
    array,_ = freenect.sync_get_video()
    array = cv2.cvtColor(array,cv2.COLOR_RGB2BGR)
    return array
#### specify a loop ################################################

if __name__ == "__main__":
    i=0
    while 1:
        #get the frame from the RGB Camera//// read the video into a img 
        frame= get_video()
        #decolorize the video
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #make squares around your face 
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
            # show the image
            cv2.imshow('Face Detected', frame)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:         # wait for ESC key to exit
            cv2.destroyAllWindows()

