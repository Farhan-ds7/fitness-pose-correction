import cv2
import mediapipe as mp
import numpy as np
import math

class PoseDetector:
    def __init__(self):
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(
                img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def getLandmarkPositions(self, img):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))
        return lmList
    
    def findAngle(self,img,p1,p2,p3,draw=True):
        x1,y1=p1[1],p1[2]
        x2,y2=p2[1],p2[2]
        x3,y3=p3[1],p3[2]

        angle=math.degrees(math.atan2(y3-y2,x3-x2)-math.atan2(y1-y2,x1-x2))

        angle=abs(angle)
        if angle>180:
            angle=360-angle
        
        cv2.circle(img,(x1,y1),10,(255,0,0),cv2.FILLED)
        cv2.line(img,(x1,y1),(x2,y2),(255,255,255),2)
        cv2.putText(img,str(int(angle)),(x2-50,y2-20),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)

        return angle