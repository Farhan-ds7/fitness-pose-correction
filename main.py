import cv2
from pose_module import PoseDetector

cap = cv2.VideoCapture(0)
detector = PoseDetector()
count=0
dir=0

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.getLandmarkPositions(img)

    if len(lmList)!=0:
        angle=detector.findAngle(img,lmList[11],lmList[13],lmList[15])
        if angle>160:
            if dir==1:
                count+=1
                dir=0
        if angle<50:
            if dir==0:
                dir=1
        cv2.putText(img,f'Reps:{count}',(20,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
        
        print("Elbow Angle:", angle)

    cv2.imshow("Pose Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()