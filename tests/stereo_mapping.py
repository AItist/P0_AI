# import cv2
# import numpy as np

# # 체스보드의 크기와 정사각형의 크기를 설정합니다.
# CHECKERBOARD = (6,9)
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# # ... (여기에 single_camera_calibration 함수와 스테레오 보정 코드가 있어야 함) ...
# def single_camera_calibration(images):
#     objpoints = []
#     imgpoints = []

#     objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
#     objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

#     for fname in images:
#         img = cv2.imread(fname)
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD)
#         if ret:
#             objpoints.append(objp)
#             corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
#             imgpoints.append(corners2)

#     ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#     return mtx, dist


# def compute_disparity(img1, img2, mtx1, dist1, mtx2, dist2):
#     # 스테레오 보정
#     stereo_retval, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_cam1, imgpoints_cam2, mtx1, dist1, mtx2, dist2, gray.shape[::-1])
#     R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(mtx1, dist1, mtx2, dist2, gray.shape[::-1], R, T)
#     map1, map2 = cv2.initUndistortRectifyMap(mtx1, dist1, R1, P1, gray.shape[::-1], cv2.CV_16SC2)
#     rectified1 = cv2.remap(img1, map1, map2, cv2.INTER_LINEAR)
#     map1, map2 = cv2.initUndistortRectifyMap(mtx2, dist2, R2, P2, gray.shape[::-1], cv2.CV_16SC2)
#     rectified2 = cv2.remap(img2, map1, map2, cv2.INTER_LINEAR)
#     stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
#     disparity = stereo.compute(rectified1, rectified2)
#     return disparity

# # 각 카메라의 이미지 목록 (이것은 예제이므로 실제 경로와 맞게 수정해야 합니다)
# cam1_images = ["cam1_image1.jpg", "cam1_image2.jpg"]
# cam2_images = ["cam2_image1.jpg", "cam2_image2.jpg"]
# cam3_images = ["cam3_image1.jpg", "cam3_image2.jpg"]
# cam4_images = ["cam4_image1.jpg", "cam4_image2.jpg"]

# cam1_mtx, cam1_dist = single_camera_calibration(cam1_images)
# cam2_mtx, cam2_dist = single_camera_calibration(cam2_images)
# cam3_mtx, cam3_dist = single_camera_calibration(cam3_images)
# cam4_mtx, cam4_dist = single_camera_calibration(cam4_images)

# # 각 카메라 쌍에 대해 깊이 맵을 계산합니다.
# disparity_12 = compute_disparity(cam1_image, cam2_image, cam1_mtx, cam1_dist, cam2_mtx, cam2_dist)
# disparity_23 = compute_disparity(cam2_image, cam3_image, cam2_mtx, cam2_dist, cam3_mtx, cam3_dist)
# disparity_34 = compute_disparity(cam3_image, cam4_image, cam3_mtx, cam3_dist, cam4_mtx, cam4_dist)
# disparity_14 = compute_disparity(cam1_image, cam4_image, cam1_mtx, cam1_dist, cam4_mtx, cam4_dist)

# # 각 깊이 맵 결과를 통합합니다.
# # 이 부분은 어떻게 통합할 것인지에 따라 다르게 구현될 수 있습니다. 
# # 가장 간단한 방법은 각 깊이 맵의 평균을 계산하는 것입니다.
# final_disparity = (disparity_12 + disparity_23 + disparity_34 + disparity_14) / 4
