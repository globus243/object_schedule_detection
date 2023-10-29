"""x_check.py: after detection_check.py has run, x_check.py sends every frame where no cat was detected by the
n-model through the x-model and uses its much more accurate detections to creates a dataset which we will later use
to train the n-model."""
import os

import cv2
import numpy as np

no_cat_found_img_path = f'{os.path.dirname( __file__ )}/no_cat_found'
x_no_cat_found_img_path = f'{os.path.dirname( __file__ )}/x_not_found'

manual_label = f'{os.path.dirname(__file__)}/labeled'

video_file_path = f'{os.path.dirname( __file__ )}/videos/various.mp4'
yolo_model_path = f'{os.path.dirname( __file__ )}/yolov8x.onnx'

model = cv2.dnn.readNetFromONNX( yolo_model_path )
no_cat_found_img_names = os.listdir( no_cat_found_img_path )
nr_imgs = len( no_cat_found_img_names )
frame_counter = 0
not_found_frames = []


for no_cat_found_img_name in no_cat_found_img_names:
    frame_counter += 1

    frame = cv2.imread( f'{no_cat_found_img_path}/{no_cat_found_img_name}' )

    print( f'Processing frame {frame_counter} of {nr_imgs}' )
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

    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)
    cat_detected = any(label == "15" for label in map(str, class_ids))

    if cat_detected:
        print(f'cat detected in frame {frame_counter}')
        for i in range(len(boxes)):
            if i in result_boxes:
                box = boxes[i]
                # Berechne die Koordinaten für das Training-Set-Format
                x_center = (box[0] + box[2] / 2) / width
                y_center = (box[1] + box[3] / 2) / height
                w = box[2] / width
                h = box[3] / height

                # Speichere den Frame
                save_img_path = f'{manual_label}/images/{no_cat_found_img_name}'
                cv2.imwrite(save_img_path, original_image)

                # Erstelle und speichere die Label-Datei
                label_data = f"15 {x_center} {y_center} {w} {h}\n"
                save_label_path = f'{manual_label}/labels/{os.path.splitext(no_cat_found_img_name)[0]}.txt'
                with open(save_label_path, 'w') as file:
                    file.write(label_data)

    else:
        print(f'cat not detected in frame {frame_counter}')
        not_found_frames.append( frame_counter )
        cv2.imwrite( f'{x_no_cat_found_img_path}/{no_cat_found_img_name}', original_image )

print(not_found_frames)
print( f'Number of frames where cat was not detected: {len(not_found_frames)}' )
