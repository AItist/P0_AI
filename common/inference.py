from common.enum_ import ePoses, eSegs
import cv2
import numpy as np

def adjust_pose_center(pose_center):
    original_width = 640
    original_height = 480
    new_width = 1024
    new_height = 1024

    # 원본 이미지가 새 이미지의 중앙에 위치하므로 이동 거리를 계산
    dx = (new_width - original_width) // 2
    dy = (new_height - original_height) // 2

    # pose_center 값을 보정
    adjusted_pose_center = [pose_center[0] + dx, pose_center[1] + dy]

    return adjusted_pose_center


def slide_image0(img, pose_center):
    import numpy as np
    # 이미지의 높이와 너비
    height, width, _ = img.shape
    
    # 초기 결과 이미지 (1024x1024) 생성 (배경을 흰색으로 설정)
    array = np.ones((1024, 1024, 3), dtype=np.uint8) * 255

    # 이미지를 array의 중앙에 위치시키기 위한 좌표 계산
    start_x = (1024 - width) // 2
    end_x = start_x + width
    start_y = (1024 - height) // 2
    end_y = start_y + height

    array[start_y:end_y, start_x:end_x] = img

    # pose_center를 보정
    pose_center = adjust_pose_center(pose_center)

    height, width, _ = array.shape

    # 이미지를 보정된 pose_center를 중심으로 슬라이드
    dx = pose_center[0] - width // 2
    dy = pose_center[1] - height // 2

     # 이미지를 dx, dy만큼 슬라이드
    if dx > 0:
        array[:, :width-dx] = array[:, dx:]
    else:
        array[:, -dx:] = array[:, :width+dx]
        
    if dy > 0:
        array[:height-dy, :] = array[dy:, :]
        # array[-dy:, :] = 255
    else:
        array[-dy:, :] = array[:height+dy, :]
        # array[:height-dy, :] = 255

    return array


# def slide_image1(img, pose_center, re_width=1024, re_height=1024):
#     import cv2
#     import numpy as np

#     # 이미지의 원래 높이와 너비
#     original_height, original_width, _ = img.shape
    
#     # 이미지를 (re_width, re_height, 3) 크기로 늘림
#     resized_img = cv2.resize(img, (re_width, re_height))
    
#     # 늘어난 이미지의 중심점
#     center_x, center_y = re_width // 2, re_height // 2
    
#     # pose_center 위치의 픽셀이 최종적으로 이미지의 중앙에 오도록 dx, dy를 계산
#     scale_x = re_width / original_width
#     scale_y = re_height / original_height
    
#     dx = int(pose_center[0] * scale_x - center_x)
#     dy = int(pose_center[1] * scale_y - center_y)
    
#     # 결과 이미지 초기화 (흰색으로 채움)
#     result = np.ones_like(resized_img) * 255
    
#     # 이미지를 dx, dy만큼 슬라이드
#     if dx > 0:
#         result[:, :re_width-dx] = resized_img[:, dx:]
#     else:
#         result[:, -dx:] = resized_img[:, :re_width+dx]
        
#     if dy > 0:
#         result[:re_height-dy, :] = resized_img[dy:, :]
#         result[-dy:, :] = 255
#     else:
#         result[-dy:, :] = resized_img[:re_height+dy, :]
#         result[:re_height-dy, :] = 255
        
#     return result

def correct_pose_with_chest(lmList, chest):
    corrected_list = []
    
    for lm in lmList:
        corrected_x = lm[1] - chest[0]
        corrected_y = lm[2] - chest[1]
        # Z 값은 chest의 Z 좌표 정보가 제공되지 않았기 때문에 변경하지 않습니다.
        corrected_pose = [lm[0], corrected_x, corrected_y, lm[3]]
        corrected_list.append(corrected_pose)
    
    return corrected_list

# 사용 예:
# lmList = [...]  # 포즈 데이터
# chest = (1, 2)
# corrected_poses = correct_pose_with_chest(lmList, chest)


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

            # neck = (int((lshoulder[1] + rshoulder[1]) / 2), int((lshoulder[2] + rshoulder[2]) / 2))
            chest = (int((lshoulder[1] + rshoulder[1] + lhip[1] + rhip[1]) / 4), 
                     int((lshoulder[2] + rshoulder[2] + lhip[2] + rhip[2]) / 4))
            # middle_spine = (int((chest[0] + neck[0]) / 2), int((chest[1] + neck[1]) / 2))
            under_spine = (int((chest[0] + lhip[1] + rhip[1]) / 3), int((chest[1] + lhip[2] + rhip[2]) / 3))
            
            # 이미지의 중심 위치를 보정한다.
            lmList_temp = correct_pose_with_chest(lmList, chest)

            lmString = ''
            for lm in lmList_temp:
                lmString += f'{lm[1]},{img.shape[0] - lm[2]},{lm[3]},'

            pose_center2 = np.sum(lst, axis=0) / 5
            pose_center2 = [int(pose_center2[1]), int(pose_center2[2])]

            print(pose_center, pose_center2, lmList[0], bboxInfo['bbox'], chest, under_spine)

            # cv2.circle(seg_img, chest, 5, (0, 255, 0), cv2.FILLED)
            # cv2.circle(seg_img, middle_spine, 5, (0, 0, 255), cv2.FILLED)
            # cv2.circle(seg_img, under_spine, 5, (255, 0, 0), cv2.FILLED)

            # seg_img_under_spine = slide_image0(seg_img, under_spine)
            seg_img = slide_image0(seg_img, under_spine)

            # seg_img_under_spine = cv2.flip(seg_img_under_spine, 0)
            seg_img = cv2.flip(seg_img, 0)
            
            if isDebug:
                cv2.imwrite(f'seg_{index}.jpg', seg_img)
                # cv2.imwrite(f'seg_chest.jpg', seg_img_chest)
                # cv2.imwrite(f'seg_under_spine.jpg', seg_img_under_spine)
                # cv2.imwrite(f'seg_center2.jpg', seg_img_center2)
                pass

            # seg_img = seg_img_under_spine

            _pose_center = [float(0.5 - pose_center[0] / img.shape[1]), float(0.5 - pose_center[1] / img.shape[0])]

            return [lmString, seg_img, _pose_center]
            # return [lmString, seg_img_under_spine, _pose_center]
        else:
            if RUN_SEG:
                seg_img = detect_seg(img, segFlag, isDebug)
                seg_img = cv2.flip(seg_img, 0)

            if RUN_POSE:
                pose_string, pose_center = detect_pose(img, poseFlag, isDebug)

            seg_img = slide_image0(seg_img, pose_center)
        


        if isDebug:
            # print(f'{index} : {pose_string}')   
            # cv2.imwrite(f'seg_{index}.jpg', seg_img)
            pass

        _pose_center = [float(0.5 - pose_center[0] / img.shape[1]), float(0.5 - pose_center[1] / img.shape[0])]

        # return [lmString, seg_img_under_spine, _pose_center]
        return [lmString, seg_img, _pose_center]
        # return [pose_string, seg_img, _pose_center]
    except Exception as e:
        print(f"main/ai_model_inference: Error during model inference: {e}")
        pass