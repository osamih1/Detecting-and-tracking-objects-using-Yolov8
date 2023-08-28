from ultralytics import YOLO
import cv2
import cvzone

# loading the model
model = YOLO("Models/yolov8n.pt")

# loading the video
cap = cv2.VideoCapture("Videos/motorbikes.mp4")

# defining the width and the height of the video
width = 1024
height = 576

# from the video
ret = True
while ret:
    # reading the frames from the video
    ret, frame = cap.read()

    if ret:
        #resizing the frame
        frame = cv2.resize(frame, (width,height))
        # detecting and tracking the objects in the frame
        results = model.track(frame, persist=True)
        # ploting the bounding boxes of the objects
        frame_ = results[0].plot()
        # visualizing the results
        cv2.imshow("Image", frame_)

        if cv2.waitKey(25) & 0xff == ord('q'):
            break

