from common.enum_ import ePoses, eSegs

def detect_pose_seg(imgdata, ePoses=ePoses.MEDIAPIPE, eSegs=ePoses.MEDIAPIPE, debug=False):
    
    if ePoses == ePoses.MEDIAPIPE and eSegs == eSegs.MEDIAPIPE:
        import common.pose_seg.mediapipe_ as mediapipe_
        return mediapipe_.detect_pose_seg(imgdata, debug=debug)
    
    pass