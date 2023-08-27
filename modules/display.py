import cv2
import time
# import matplot3d_2 as m3d

class DisplayImage:
    @staticmethod
    def generateAndDisplay(queue, image_queue, lock, isDebug, stop_thread=False):
        while True:
            if stop_thread:
                cv2.destroyAllWindows()
                break

            queue_have_data = not queue.empty()
            # all_queues_have_data = all([not q.empty() for q in queue])
            
            if queue_have_data:
                img, timestamp = None, None
                while not queue.empty():
                    img, timestamp = queue.get()


                if img is not None:
                    # print(f"Timestamp for image from camera {i}: {timestamp}")
                    if isDebug:
                        cv2.imshow(f'frame', img)

                    with lock:
                        image_queue.put(img)  # Add the image to the queue
                else:
                    # print(f"No image from camera {i}")
                    pass
                
                # print(len(images))

                # # Now 'images' list contains the latest image from each webcam
                # images_queue.put(images)  # Add the list to the queue

                if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit if Q is pressed
                    break
            # with lock:


            time.sleep(1/20)

    @staticmethod
    def generateAndDisplayAll(queues, img_queue, lock):
        while True:
            all_queues_have_data = all([not q.empty() for q in queues])
            
            with lock:
                if all_queues_have_data:
                    for i, queue in enumerate(queues):
                        # Get the latest image from the queue
                        img, timestamp = None, None
                        while not queue.empty():
                            img, timestamp = queue.get()  # This will always get the last item if the queue is not empty
                        
                        # print(f'frame{i}')
                        if img is not None:
                            # print(f"Timestamp for image from camera {i}: {timestamp}")
                            cv2.imshow(f'frame{i}', img)

                            img_queue.put(img)  # Add the image to the queue
                        else:
                            print(f"No image from camera {i}")
                            pass


                    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit if Q is pressed
                        break


            time.sleep(1/20)

