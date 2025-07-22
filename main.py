import cv2
from pose_module import PoseDetector

video=cv2.VideoCapture(0)
detector=PoseDetector(min_detection_confidence=0.7,min_tracking_confidence=0.6)
current_mode="biceps"
print("press 'b' for Bicep curl | 's' for squat | 'p' for pushups | 'q' to Quit")

while True:
    success,img =video.read()
    if not success:
        break
    img=detector.findPose(img)
    lmList=detector.getLandmarkPositions(img)

    if current_mode=="biceps":
        img=detector.countBicepCurls(img,lmList)
        cv2.putText(img, "Mode: BICEP CURL", (400,50),
                    cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        
    elif current_mode=="squat":
        img=detector.countSquats(img,lmList)
        cv2.putText(img, "Mode: SQUAT", (400, 50),
                    cv2.FONT_HERSHEY_PLAIN,3,(0,255,255),3)
        
    elif current_mode=="Pushups":
        img=detector.countPushups(img,lmList)
        cv2.putText(img,"Mode: Pushups",(400,50),
                    cv2.FONT_HERSHEY_PLAIN,2,(0,255,255),2)
        
    cv2.imshow("Fitness pose correction",img)
    key=cv2.waitKey(1)&0xFF
    if key==ord('q'):
        break
    elif key==ord('b'):
        current_mode="biceps"
    elif key==ord('s'):
        current_mode="squat"
    elif key==ord('p'):
        current_mode="Pushups"
video.release()
cv2.destroyAllWindows()