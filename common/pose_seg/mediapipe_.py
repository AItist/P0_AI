import cv2
import mediapipe as mp
import numpy as np
import math

# MediaPipe holistic model
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# MediaPipe selfie segmentation model
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)

def _findPosition(image, pose_landmarks, draw=True):
    if not pose_landmarks:
        return None

    lmList = []
    bboxInfo = {}

    for id, lm in enumerate(pose_landmarks.landmark):
        h, w, c = image.shape
        cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
        lmList.append([id, cx, cy, cz])

    # Bounding Box
    ad = abs(lmList[12][1] - lmList[11][1]) // 2
    x1 = lmList[12][1] - ad
    x2 = lmList[11][1] + ad

    y2 = lmList[29][2] + ad
    y1 = lmList[1][2] - ad

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    bbox = (x1, y1, x2 - x1, y2 - y1)
    cx, cy = bbox[0] + (bbox[2] // 2), \
                bbox[1] + bbox[3] // 2

    bboxInfo = {"bbox": bbox, "center": (cx, cy)}

    if not draw:
        cv2.rectangle(image, bbox, (255, 0, 255), 3)
        cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    return lmList, bboxInfo

def detect_pose_seg(image, debug=False):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect pose
    results = holistic.process(image_rgb)
    
    # Detect segmentation
    segmentation_results = selfie_segmentation.process(image_rgb)
    
    # # Extract pose landmarks
    # pose_landmarks = []

    lmList, bboxInfo = _findPosition(image, results.pose_landmarks, draw=False)
    
    if bboxInfo:
        lmString = ''
        for lm in lmList:
            lmString += f'{lm[1]},{image.shape[0] - lm[2]},{lm[3]},'

    # return lmString, bboxInfo['center']

    # print(1)

    # if results.pose_landmarks:
    #     for landmark in results.pose_landmarks.landmark:
    #         pose_landmarks.append((landmark.x, landmark.y))

    # print(lmList[0], lmList[15])
    # Extract segmentation mask
    mask_image = cv2.cvtColor(segmentation_results.segmentation_mask * 255, cv2.COLOR_GRAY2BGR).astype('uint8')
    inverse_mask_image = cv2.bitwise_not(mask_image)
    white_image = 255 * np.ones_like(image)

    segmented_foreground = cv2.bitwise_and(image, mask_image)
    segmented_background = cv2.bitwise_and(white_image, inverse_mask_image)

    segmented_image = cv2.add(segmented_foreground, segmented_background)
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    # cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    
    return lmString, bboxInfo['center'], segmented_image, lmList

if __name__ == '__main__':
    # Capture video from camera
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        pose_string, pose_center, seg_img = detect_pose_seg(frame)
        
        # print(pose_landmarks)

        # Display the segmented image (for example)
        cv2.imshow('Segmented Image', seg_img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
