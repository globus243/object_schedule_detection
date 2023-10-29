"""detection_check.py: this python script runs a video file against the yolov8n-model and saves all frames where it
was not able to detect a cat to a folder. (no_cat_found_img_path)"""

import os

import cv2
import numpy as np

no_cat_found_img_path = f'{os.path.dirname( __file__ )}/no_cat_found'

video_file_path = f'{os.path.dirname( __file__ )}/videos/various.mp4'
yolo_model_path = f'{os.path.dirname( __file__ )}/yolov8n.onnx'

cap = cv2.VideoCapture( video_file_path )
model = cv2.dnn.readNetFromONNX( yolo_model_path )
nr_of_frames = int( cap.get( cv2.CAP_PROP_FRAME_COUNT ) )
frame_counter = 0
not_found_frames = []

while cap.isOpened( ):
    frame_counter += 1

    ret, frame = cap.read( )

    if ret:
        # if frame black or white, skip
        if np.all( frame == 0 ) or np.all( frame == 255 ):
            continue

        print( f'Processing frame {frame_counter} of {nr_of_frames}' )
        # resize 1920x1080 to 640x320
        frame = cv2.resize( frame, (640, 320) )
        original_image = frame.copy( )
        [ height, width, _ ] = original_image.shape
        length = max( (height, width) )
        image = np.zeros( (length, length, 3), np.uint8 )
        image[ 0:height, 0:width ] = original_image
        scale = length / 640
        blob = cv2.dnn.blobFromImage( image, scalefactor = 1 / 255, size = (640, 640), swapRB = True )
        model.setInput( blob )
        outputs = model.forward( )
        outputs = np.array( [ cv2.transpose( outputs[ 0 ] ) ] )
        rows = outputs.shape[ 1 ]

        boxes, scores, class_ids = [ ], [ ], [ ]
        for i in range( rows ):
            classes_scores = outputs[ 0 ][ i ][ 4: ]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc( classes_scores )
            if maxScore >= 0.25 and maxClassIndex in [ 15, 16 ]:
                box = [
                    outputs[ 0 ][ i ][ 0 ] - (0.5 * outputs[ 0 ][ i ][ 2 ]),
                    outputs[ 0 ][ i ][ 1 ] - (0.5 * outputs[ 0 ][ i ][ 3 ]),
                    outputs[ 0 ][ i ][ 2 ], outputs[ 0 ][ i ][ 3 ] ]
                boxes.append( box )
                scores.append( maxScore )
                class_ids.append( maxClassIndex )

        cat_detected = any( label == "15" for label in map( str, class_ids ) )

        if not cat_detected:
            # save frame to no_cat_found_img_path
            cv2.imwrite( f'{no_cat_found_img_path}/frame_{frame_counter}.jpg', frame )
            not_found_frames.append( frame_counter )


print (not_found_frames)
print( f'Number of frames where a cat was not detected: {len(not_found_frames)}' )
