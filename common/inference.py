from common.enum_ import ePoses, eSegs

def slide_image(img, pose_center):
    import numpy as np
    # 이미지의 높이와 너비
    height, width, _ = img.shape
    
    # 이미지의 중심점
    center_x, center_y = width // 2, height // 2
    
    # pose_center와 이미지 중심점과의 차이 계산
    dx = pose_center[0] - center_x
    dy = pose_center[1] - center_y
    
    # 결과 이미지 초기화 (검은색으로 채움)
    result = np.ones_like(img) * 255
    
    # 이미지를 dx, dy만큼 슬라이드
    if dx > 0:
        result[:, :width-dx] = img[:, dx:]
    else:
        result[:, -dx:] = img[:, :width+dx]
        
    if dy > 0:
        result[:height-dy, :] = img[dy:, :]
    else:
        result[-dy:, :] = img[:height+dy, :]
        
    return result

def ai_model_inference(index, img, isDebug=False, RUN_POSE=True, RUN_SEG=True, poseFlag=ePoses.CVZONE, segFlag=eSegs.YOLO):
    import cv2
    from common.detect_pose import detect_pose
    from common.detect_seg import detect_seg
    from common.detect_pose_seg import detect_pose_seg

    try:
        pose_string = None
        seg_img = None

        pose_string = None
        pose_center = None
        seg_img = None

        if poseFlag == ePoses.MEDIAPIPE or segFlag == eSegs.MEDIAPIPE:
            pose_string, pose_center, seg_img = detect_pose_seg(img, poseFlag, segFlag, isDebug)
            seg_img = slide_image(seg_img, pose_center)
            seg_img = cv2.flip(seg_img, 0)
            cv2.imwrite(f'seg_{index}.jpg', seg_img)
        else:
            if RUN_SEG:
                seg_img = detect_seg(img, segFlag, isDebug)
                seg_img = cv2.flip(seg_img, 0)

            if RUN_POSE:
                pose_string, pose_center = detect_pose(img, poseFlag, isDebug)

            seg_img = slide_image(seg_img, pose_center)
        
        # if RUN_SEG:
        #     seg_img = detect_seg(img, segFlag, isDebug)
        #     seg_img = cv2.flip(seg_img, 0)

        # if RUN_POSE:
        #     pose_string, pose_center = detect_pose(img, poseFlag, isDebug)

        if isDebug:
            # print(f'{index} : {pose_string}')   
            cv2.imwrite(f'seg_{index}.jpg', seg_img)
            pass

        _pose_center = [float(0.5 - pose_center[0] / img.shape[1]), float(0.5 - pose_center[1] / img.shape[0])]

        # print('1')
        return [pose_string, seg_img, _pose_center]
    except Exception as e:
        print(f"main/ai_model_inference: Error during model inference: {e}")
        pass