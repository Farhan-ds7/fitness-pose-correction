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

        self.squat_count=0
        self.squat_dir=0

        self.pushup_counter=0
        self.pushup_dir=0

    def speak(self,text):
        engine.say(text)
        engine.runAndWait()
        
    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks and draw:
            landmark_style=self.mpDraw.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=4)
            connection_style=self.mpDraw.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2)
            self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS,landmark_drawing_spec=landmark_style,connection_drawing_spec=connection_style)
        return img

    def getLandmarkPositions(self,img):
        lmList=[]
        if self.results.pose_landmarks:
            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,_= img.shape
                cx,cy=int(lm.x * w),int(lm.y*h)
                lmList.append((id,cx,cy))
        return lmList
    
    def areallkeypointsvisible(self,img,p1,p2,p3,draw=True):
        return all(id<len(lmList)for id in required_ids)
    
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
        required_ids = [11,13,15,27,12,14,16,28,23,24]
        if not lmList or len(lmList)<25:
            return img
        
        left_shoulder_angle=self.findAngle(img,lmList[13],lmList[11],lmList[23],draw=True)  # Elbow-Shoulder-Hip
        right_shoulder_angle=self.findAngle(img,lmList[14],lmList[12],lmList[24],draw=True)  # Elbow-Shoulder-Hip
        shoulders_straight=(0<=left_shoulder_angle<=25)and(0<=right_shoulder_angle<=25)

        left_hip_y=lmList[23][2]
        right_hip_y=lmList[24][2]
        hip_diff=abs(left_hip_y-right_hip_y)
        hip_stable=hip_diff<30

        cv2.putText(img,f'Right Reps:{self.count_right}',(20,50),
                cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
        cv2.putText(img,f'Left Reps:{self.count_left}',(20,90),
                cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

        if not shoulders_straight or not hip_stable:
                warning_text=""
                if not shoulders_straight:
                    warning_text+="Fix Shoulder"
                if not hip_stable:
                    warning_text+="Fix hip"
                cv2.putText(img,warning_text.strip(), (20, 130),cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
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

    def countSquats(self,img,lmList):
        if not lmList or len(lmList)<33:
            return img
        right_shoulder=lmList[12]
        right_hip=lmList[24]
        right_knee=lmList[26]
        right_ankle=lmList[28]
        right_foot=lmList[32]


        left_shoulder=lmList[11]
        left_hip=lmList[23]
        left_knee=lmList[25]
        left_ankle=lmList[27]
        left_foot=lmList[31]


        knee_angle_right=self.findAngle(img,right_hip,right_knee,right_ankle)
        knee_angle_left=self.findAngle(img,left_hip,left_knee,left_ankle)

        hip_angle_right=self.findAngle(img,right_shoulder,right_hip,right_knee)
        hip_angle_left=self.findAngle(img,left_shoulder,left_hip,left_knee)

        torso_angle_right=self.findAngle(img,right_shoulder,right_hip,right_knee,draw=False)
        torso_angle_left=self.findAngle(img,left_shoulder,left_hip,left_knee,draw=False)

        right_knee_ok = right_knee[1]<right_foot[1]
        left_knee_ok = left_knee[1]<left_foot[1]

        cv2.putText(img, f'KneeR: {int(knee_angle_right)}',(20,100),
                cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
        cv2.putText(img, f'KneeL: {int(knee_angle_left)}',(20,130),
                cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
        cv2.putText(img, f'HipR: {int(hip_angle_right)}',(20,160),
                cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
        cv2.putText(img, f'HipL: {int(hip_angle_left)}', (20,190),
                cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
        cv2.putText(img, f'TorsoR: {int(torso_angle_right)}',(20, 220),
            cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 2)

        
        if knee_angle_right<60 and knee_angle_left<60 and hip_angle_right<60 and hip_angle_left<60:
            if self.squat_dir == 0:  # going down
                self.squat_dir = 1
        
        elif knee_angle_right > 150 and knee_angle_left > 150 and hip_angle_right > 150 and hip_angle_left > 150:
            if self.squat_dir == 1:  # coming up
                self.squat_count += 1
                self.squat_dir = 0
                self.speak(f"Squat rep {self.squat_count}")
        
        warning = ""
        if hip_angle_right > 70 or hip_angle_left > 70:
            warning += "Go deeper. "
        if torso_angle_right > 130 or torso_angle_left > 130:
            warning += "Keep your back straight. "
        if not right_knee_ok or not left_knee_ok:
            warning += "Knees behind toes."

        if warning:
            cv2.putText(img, warning.strip(), (20, 250),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv2.rectangle(img,(10,10),(310,60),(255,255,255),-1)

        cv2.putText(img, f'Squats: {self.squat_count}', (20, 50),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        return img
    
    def countPushups(self,img,lmList):
        if not lmList or len(lmList)<33:
            self.pushup_ready=False
            return img
        
        left_shoulder=lmList[11]
        left_elbow=lmList[13]
        left_wrist=lmList[15]
        left_ankle=lmList[27]

        right_shoulder=lmList[12]
        right_elbow=lmList[14]
        right_wrist=lmList[16]
        right_ankle=lmList[28]

        left_hip=lmList[23]
        right_hip=lmList[24]

        angle_left=self.findAngle(img,left_shoulder,left_elbow,left_wrist)
        angle_right=self.findAngle(img,right_shoulder,right_elbow,right_wrist)

        torso_angle_left=self.findAngle(img,left_shoulder,left_hip,left_ankle,draw=False)
        torso_angle_right=self.findAngle(img,right_shoulder,right_hip,right_ankle,draw=False)

        cv2.putText(img,f"L-Elbow:{int(angle_left)}",(20,100),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
        cv2.putText(img,f"R-Elbow:{int(angle_right)}",(20,130),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)

        if not self.pushup_ready:
            if angle_left > 160 and angle_right > 160 and torso_angle_left > 160 and torso_angle_right > 160:
                self.pushup_ready = True
                self.speak("Push-up position detected. Start now.")
            else:
                cv2.putText(img, "Get into push-up position...", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                return img  # Don't proceed until ready
        
        if angle_left<90 and angle_right<90:
            if self.pushup_dir==0:
                self.pushup_dir=1
        elif angle_left>160 and angle_right>160:
            if self.pushup_dir==1:
                self.pushup_counter+=1
                self.pushup_dir=0
                self.speak(f"push-up rep {self.pushup_counter}")
        cv2.putText(img,f"push-up rep {self.pushup_counter}",(20,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

        return img