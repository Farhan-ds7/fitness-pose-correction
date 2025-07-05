import cv2
from pose_module import PoseDetector

cap=cv2.VideoCapture(0)
detector=PoseDetector(min_detection_confidence=0.7,min_tracking_confidence=0.6)
while True:
    success, img = cap.read()
    img=detector.findPose(img)
    lmList=detector.getLandmarkPositions(img)
    img=detector.countBicepCurls(img,lmList)        
    cv2.imshow("Bicep Curl Counter - Both Arms",img)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()