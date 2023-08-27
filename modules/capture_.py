
def capture(webcam_lst):
    import cv2

    camIndex = next(iter(webcam_lst))
    camKey = webcam_lst[camIndex]

    # Create a VideoCapture object
    cap = cv2.VideoCapture(camIndex)

    # Check if camera opened successfully
    if not cap.isOpened(): 
        print("Unable to read camera feed")

    # Default resolutions of the frame are obtained (system dependent)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    while(True):
        ret, frame = cap.read()

        if ret: 
            # Display the resulting frame    
            cv2.imshow('Webcam Live', frame)

            cv2.circle(frame, (int(frame_width/2), int(frame_height/2)), 10, (0, 0, 255), 2)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break 

    # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows() 
