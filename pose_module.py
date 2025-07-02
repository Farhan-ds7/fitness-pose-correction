import cv2
import mediapipe as mp
import numpy as np
import math
import pyttsx3

engine=pyttsx3.init()
voices=engine.getProperty('voices')
for voice in voices:
    if "Albert" in voice.name.lower():
        engine.setProperty('voice',voice.id)
        break
engine.setProperty('rate',150)
engine.setProperty('volume',1.0)

class PoseDetector:
    def __init__(self):
        self.mpDraw=mp.solutions.drawing_utils
        self.mpPose=mp.solutions.pose
        self.pose=self.mpPose.Pose()

        self.count_right=0
        self.dir_right=0
        self.count_left=0
        self.dir_left=0
        
    def speak(self,text):
        engine.say(text)
        engine.runAndWait()
        
    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(
                img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def getLandmarkPositions(self,img):
        lmList=[]
        if self.results.pose_landmarks:
            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,_= img.shape
                cx,cy=int(lm.x * w),int(lm.y*h)
                lmList.append((id,cx,cy))
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
        cv2.circle(img,(x2, y2),10,(255,0,0),cv2.FILLED)
        cv2.circle(img,(x3, y3),10,(255,0,0),cv2.FILLED)
        cv2.line(img,(x1,y1),(x2,y2),(255,255,255),3)
        cv2.line(img,(x3,y3),(x2,y2),(255, 255, 255),3)
        cv2.putText(img,f'{int(angle)}',(x2 - 50, y2 + 40),
                        cv2.FONT_HERSHEY_PLAIN,2,(0, 255, 0), 2)
        return angle
    
    def countBicepCurls(self,img,lmList):
        if not lmList or len(lmList)<25:
            return img
        
        left_shoulder_angle=self.findAngle(img,lmList[13],lmList[11],lmList[23],draw=True)  # Elbow-Shoulder-Hip
        right_shoulder_angle=self.findAngle(img,lmList[14],lmList[12],lmList[24],draw=True)  # Elbow-Shoulder-Hip
        shoulders_straight=(0<=left_shoulder_angle<=25)and(0<=right_shoulder_angle<=25)

        cv2.putText(img,f'Right Reps:{self.count_right}',(20,50),
                cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
        cv2.putText(img,f'Left Reps:{self.count_left}',(20,90),
                cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

        if not shoulders_straight:
                cv2.putText(img, "Keep Shoulders Straight!!", (20, 130),cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
                return img
        
        angle_right = self.findAngle(img,lmList[12],lmList[14],lmList[16])        
        if angle_right>160:
            if self.dir_right==1:
                self.count_right+=1
                self.dir_right=0
                self.speak(f"Right rep {self.count_right}")
        elif angle_right<50:
            if self.dir_right==0:
                self.dir_right=1

        angle_left = self.findAngle(img,lmList[11],lmList[13],lmList[15])
        if angle_left>160:
            if self.dir_left==1:
                self.count_left+=1
                self.dir_left=0
                self.speak(f"Left rep {self.count_left}")
        elif angle_left<50:
            if self.dir_left==0:
                self.dir_left=1
        return img
