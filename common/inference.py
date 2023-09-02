from common.enum_ import ePoses, eSegs
import cv2
import numpy as np

def slide_image0(img, pose_center):
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
    # result = np.ones_like((1024, 1024, 3), dtype=np.uint8) * 255
    
    # array = np.zeros((1024, 1024, 3), dtype=np.uint8)
    array = np.ones((1024, 1024, 3), dtype=np.uint8) * 255
    # print(array.shape)

    # 이미지를 dx, dy만큼 슬라이드
    if dx > 0:
        result[:, :width-dx] = img[:, dx:]
    else:
        result[:, -dx:] = img[:, :width+dx]
        
    if dy > 0:
        result[:height-dy, :] = img[dy:, :]
        result[-dy:, :] = 255
    else:
        result[-dy:, :] = img[:height+dy, :]
        result[:height-dy, :] = 255
        
    return result

def slide_image1(img, pose_center, re_width=1024, re_height=1024):
    import cv2
    import numpy as np

    # 이미지의 원래 높이와 너비
    original_height, original_width, _ = img.shape
    
    # 이미지를 (re_width, re_height, 3) 크기로 늘림
    resized_img = cv2.resize(img, (re_width, re_height))
    
    # 늘어난 이미지의 중심점
    center_x, center_y = re_width // 2, re_height // 2
    
    # pose_center 위치의 픽셀이 최종적으로 이미지의 중앙에 오도록 dx, dy를 계산
    scale_x = re_width / original_width
    scale_y = re_height / original_height
    
    dx = int(pose_center[0] * scale_x - center_x)
    dy = int(pose_center[1] * scale_y - center_y)
    
    # 결과 이미지 초기화 (흰색으로 채움)
    result = np.ones_like(resized_img) * 255
    
    # 이미지를 dx, dy만큼 슬라이드
    if dx > 0:
        result[:, :re_width-dx] = resized_img[:, dx:]
    else:
        result[:, -dx:] = resized_img[:, :re_width+dx]
        
    if dy > 0:
        result[:re_height-dy, :] = resized_img[dy:, :]
        result[-dy:, :] = 255
    else:
        result[-dy:, :] = resized_img[:re_height+dy, :]
        result[:re_height-dy, :] = 255
        
    return result



def ai_model_inference(index, img, isDebug=False, RUN_POSE=True, RUN_SEG=True, poseFlag=ePoses.CVZONE, segFlag=eSegs.YOLO):
    import cv2
    import numpy as np
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
            pose_string, pose_center, seg_img, lmList, bboxInfo = detect_pose_seg(img, poseFlag, segFlag, isDebug)
            # print(lmList[0], lmList[11], lmList[12], lmList[23], lmList[24])
            # [0]: 머리
            # [11]: 왼쪽 어깨
            # [12]: 오른쪽 어깨
            # [23]: 왼쪽 엉덩이
            # [24]: 오른쪽 엉덩이
            lst = [lmList[0], lmList[11], lmList[12], lmList[23], lmList[24]]
            lst = np.array(lst)

            lshoulder = lmList[11]
            rshoulder = lmList[12]
            lhip = lmList[23]
            rhip = lmList[24]

            neck = (int((lshoulder[1] + rshoulder[1]) / 2), int((lshoulder[2] + rshoulder[2]) / 2))
            chest = (int((lshoulder[1] + rshoulder[1] + lhip[1] + rhip[1]) / 4), 
                     int((lshoulder[2] + rshoulder[2] + lhip[2] + rhip[2]) / 4))
            middle_spine = (int((chest[0] + neck[0]) / 2), int((chest[1] + neck[1]) / 2))
            

            pose_center2 = np.sum(lst, axis=0) / 5
            pose_center2 = [int(pose_center2[1]), int(pose_center2[2])]
            # pose_center2 = np.sum(lmList[0] + lmList[11] + lmList[12] + lmList[23] + lmList[24]) / 5

            print(pose_center, pose_center2, lmList[0], bboxInfo['bbox'])

            # bbox_x_min = bboxInfo['bbox'][0]
            # bbox_x_max = bboxInfo['bbox'][0] + bboxInfo['bbox'][2]
            # bbox_y_min = bboxInfo['bbox'][1]
            # bbox_y_max = bboxInfo['bbox'][1] + bboxInfo['bbox'][3]

            # # print(bbox_x_min, bbox_x_max, bbox_y_min, bbox_y_max)

            # pose_center2_x_min_dist = pose_center2[0] - bbox_x_min
            # pose_center2_x_max_dist = bbox_x_max - pose_center2[0]
            # pose_center2_y_min_dist = pose_center2[1] - bbox_y_min
            # pose_center2_y_max_dist = bbox_y_max - pose_center2[1]
            # # print(pose_center2_x_min_dist, pose_center2_x_max_dist, pose_center2_y_min_dist, pose_center2_y_max_dist)

            # # 오른쪽 끝으로 간 상황
            # x_max = pose_center2_x_min_dist if pose_center2_x_min_dist > pose_center2_x_max_dist else pose_center2_x_max_dist
            # y_max = pose_center2_y_min_dist if pose_center2_y_min_dist > pose_center2_y_max_dist else pose_center2_y_max_dist
            # cv2.circle(seg_img, pose_center2, 5, (0, 0, 0), cv2.FILLED)

            # cv2.circle(seg_img, neck, 5, (255, 0, 0), cv2.FILLED)
            cv2.circle(seg_img, chest, 5, (0, 255, 0), cv2.FILLED)
            # cv2.circle(seg_img, middle_spine, 5, (0, 0, 255), cv2.FILLED)


            # if pose_center2_x_min_dist > pose_center2_x_max_dist:
            #     # 왼쪽 길이가 오른쪽 길이보다 길다면                
            #     # print(f'before: {pose_center2[0]}')
            #     pose_center2[0] = pose_center2[0] - pose_center2_x_min_dist + pose_center2_x_max_dist
            #     # print(f'after: {pose_center2[0]}')
            #     pass
            # else:
            #     # print(f'before: {pose_center2[0]}')
            #     pose_center2[0] = pose_center2[0] + pose_center2_x_max_dist - pose_center2_x_min_dist
            #     # print(f'after: {pose_center2[0]}')
            #     pass

            # if pose_center2_y_min_dist > pose_center2_y_max_dist:
            #     # 위쪽 길이가 아래쪽 길이보다 길다면
            #     print(f'before: {pose_center2[1]}')
            #     pose_center2[1] = pose_center2[1] - pose_center2_y_min_dist + pose_center2_y_max_dist
            #     print(f'after: {pose_center2[1]}')
            #     pass
            # else:
            #     print(f'before: {pose_center2[1]}')
            #     pose_center2[1] = pose_center2[1] + pose_center2_y_max_dist - pose_center2_y_min_dist
            #     print(f'after: {pose_center2[1]}')
            #     pass

            # seg_img = slide_image(seg_img, pose_center)
            # new_img = np.ones((1024, 1024, 3), dtype=np.uint8) * 255

            seg_img = slide_image0(seg_img, chest)
            # seg_img = slide_image1(seg_img, pose_center2)
            # seg_img = slide_image_centered_v5(seg_img, pose_center2)


            seg_img = cv2.flip(seg_img, 0)
            # print(lmList[0], lmList[15])
            cv2.imwrite(f'seg_{index}.jpg', seg_img)
        else:
            if RUN_SEG:
                seg_img = detect_seg(img, segFlag, isDebug)
                seg_img = cv2.flip(seg_img, 0)

            if RUN_POSE:
                pose_string, pose_center = detect_pose(img, poseFlag, isDebug)

            seg_img = slide_image0(seg_img, pose_center)
        
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