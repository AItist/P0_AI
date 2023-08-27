import cv2
from multiprocessing import Array, Value, Lock
import time
import numpy as np

class CaptureImage:
    def __init__(self, index, stop_thread, _isDebug=False, capture_interval=1/60):
        
        self.index = index
        self.stop_thread = stop_thread
        self.isDebug = _isDebug
        self.env_capture_interval = capture_interval

        self.max_size = 480 * 640 * 3
        # self.max_size = 1024 * 1024 * 3
        self.lock = Lock()
        self.shared_array = Array('B', self.max_size)
        self.size = Value('i', 0)
        self.timestamp = Value('d', 0.0)  # Add timestamp

    def capture(self, isDebug=False):
        cap = cv2.VideoCapture(self.index)
        self.isDebug = isDebug # XXX: 이거 안먹혀서 직접 걸어버림

        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # print capture width, height and FPS
        print("capture width,height, FPS: ", cap.get(3), cap.get(4), cap.get(5))

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        try:
            while True:
                if self.stop_thread:
                    break

                ret, frame = cap.read()

                if self.isDebug:
                    cv2.circle(frame, (int(frame_width/2), int(frame_height/2)), 10, (0, 0, 255), 2)

                # Write the gray image to shared memory
                _, img_encoded = cv2.imencode('.jpg', frame)
                img_bytes = img_encoded.tobytes()

                with self.lock:
                    self.shared_array[:len(img_bytes)] = np.frombuffer(img_bytes, dtype='uint8')
                    self.size.value = len(img_bytes)
                    self.timestamp.value = time.time()  # Save the timestamp

                time.sleep(self.env_capture_interval)

            cap.release()
        except:
            cap.release()
            cv2.destroyAllWindows()
            print('KeyboardInterrupt')
