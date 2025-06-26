import cv2
from pose_module import PoseDetector

cap = cv2.VideoCapture(0)
detector = PoseDetector()

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.getLandmarkPositions(img)

    if len(lmList)!=0:
        angle=detector.findAngle(img,lmList[11],lmList[13],lmList[15])
        print("Elbow Angle:", angle)

    cv2.imshow("Pose Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()