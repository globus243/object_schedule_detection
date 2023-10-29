# object_schedule_detection
This project uses Yolov8 and OpenCV to detect a given object in an RTSP stream video stream. 
The frame can be split into two parts and detections can be counted for each part.
It can also store the detections in a MariaDB database, when given a connection, or in Memory and display them as a Gantt-Diagram using plotly.

This Repo is part of the Blogpost: https://blog.timhartmann.de/2023/10/29/object-detection-is-easy/

This repo comes without the actual Yolov8 model. You can download it here: https://github.com/ultralytics/ultralytics.
You will also need to provide a video stream or a video file. I used a RTSP stream from a TP-Link C210.