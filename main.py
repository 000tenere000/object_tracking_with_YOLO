from ultralytics import YOLO
import cv2


model = YOLO("yolov8n.pt")

video_path = "motorbikes.mp4"

cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
   
size = (frame_width, frame_height)

video_write = cv2.VideoWriter('filename.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10,size)

ret = True

while ret:

    ret,frame = cap.read()

    if ret:
        results = model.track(frame,persist=True,save=True)

        frame_ = results[0].plot()


        cv2.imshow("frame",frame_)
        video_write.write(frame_)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

