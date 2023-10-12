from ultralytics import YOLO
import cv2

# loading the model
model = YOLO("Models/yolov8n.pt")

# capturing the video
video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)

# reading the frames from the video
width = 854
height = 480

frames_of_detections = []
ret = True
while ret:
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (width,height))
        # detecting and tracking objects
        results = model.track(frame)
        frame_ = results[0].plot()
        frames_of_detections.append(frame_)
        cv2.imshow("image", frame_)
        key = cv2.waitKey(25)
        if key & 0xFF == ord("q"):
            break

# create a new video writer object
video_writer = cv2.VideoWriter("detections.mp4", cv2.VideoWriter_fourcc(*"XVID"), 30, (width, height))

# write the frames of detections to the new video file
for frame in frames_of_detections:
    video_writer.write(frame)

# release the video writer object
video_writer.release()

print("We're done with the process of the video")