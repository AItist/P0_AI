import cv2
from multiprocessing import Queue
from queue import Empty
import numpy as np
import time

class ExtractImage:
    def __init__(self, shared_array, size, lock, timestamp, stop_thread, process_interval=0.1):
        self.env_process_interval = process_interval

        self.shared_array = shared_array
        self.size = size
        self.lock = lock
        self.timestamp = timestamp
        self.queue = Queue()  # Queue for images
        self.stop_thread = stop_thread

    def extract(self):
        try:
            while True:
                if self.stop_thread:
                    break

                with self.lock:
                    img_bytes = bytes(self.shared_array[:self.size.value])  # only read the number of bytes that were written
                    timestamp = self.timestamp.value  # Get the timestamp

                if img_bytes:
                    img_np = np.frombuffer(img_bytes, dtype='uint8')
                    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

                    # Instead of showing the image here, we put it on the queue
                    while True:  # empty the queue and put the latest processed image
                        try:
                            self.queue.get_nowait()
                        except Empty:
                            break
                    self.queue.put((img, timestamp))  # Add the timestamp with the image

                time.sleep(self.env_process_interval)
        except:
            print('KeyboardInterrupt')
            pass

