import cv2
from pose_module import PoseDetector

video=cv2.VideoCapture(0)
detector=PoseDetector(min_detection_confidence=0.7,min_tracking_confidence=0.6)
current_mode="biceps"
print("press 'b' for Bicep curl | 's' for squat | 'q' to Quit")

while True:
    success, img = video.read()
    if not success:
        break
    img=detector.findPose(img)
    lmList=detector.getLandmarkPositions(img)
    if current_mode=="biceps":
        img=detector.countBicepCurls(img,lmList)
        cv2.putText(img, "Mode: BICEP CURL", (400, 50),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    elif current_mode=="squat":
        img=detector.countSquats(img,lmList)
        cv2.putText(img, "Mode: SQUAT", (400, 50),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
    cv2.imshow("Fitness pose correction",img)
    key=cv2.waitKey(1)&0xFF
    if key==ord('q'):
        break
    elif key==ord('b'):
        current_mode="biceps"
    elif key==ord('s'):
        current_mode="squat"
        detector.squat_count = 0
        detector.squat_dir = 0
video.release()
cv2.destroyAllWindows()