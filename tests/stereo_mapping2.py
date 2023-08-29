import cv2
import numpy as np

CHECKERBOARD = (6,9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def calibrate_camera(images):
    objpoints = []  # 3D points in real world space 
    imgpoints = []  # 2D points in image plane

    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD)
        
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
    
    ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist

# Load images of the checkerboard captured from both cameras
cam1_images = ["cam1_img1.jpg", "cam1_img2.jpg", "cam1_img3.jpg"]
cam2_images = ["cam2_img1.jpg", "cam2_img2.jpg", "cam2_img3.jpg"]

cam1_mtx, cam1_dist = calibrate_camera(cam1_images)
cam2_mtx, cam2_dist = calibrate_camera(cam2_images)
