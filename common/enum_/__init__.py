from enum import Enum

class ePoses(Enum):
    MEDIAPIPE = 1
    CVZONE = 2

class eSegs(Enum):
    MEDIAPIPE = 1
    YOLO = 2
    PIDNet = 3