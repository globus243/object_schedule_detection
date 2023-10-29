"""test_model.py: this python script runs a given video file against a given onnx model and displays the results in a
window."""
import argparse
import cv2
import numpy as np
from ultralytics.utils import yaml_load
from ultralytics.utils.checks import check_yaml

CLASSES = yaml_load( check_yaml( 'coco128.yaml' ) )[ 'names' ]
colors = np.random.uniform( 0, 255, size = (len( CLASSES ), 3) )


def draw_bounding_box( img, class_id, confidence, x, y, x_plus_w, y_plus_h ):
    label = f'{CLASSES[ class_id ]} ({confidence:.2f})'
    color = colors[ class_id ]
    cv2.rectangle( img, (x, y), (x_plus_w, y_plus_h), color, 2 )
    cv2.putText( img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2 )


def main( onnx_model, input_video ):
    # load model
    model = cv2.dnn.readNetFromONNX( onnx_model )
    # load video file
    cap = cv2.VideoCapture( input_video )

    # while video is opened
    while cap.isOpened( ):
        # as long as there are frames, continue
        ret, original_image = cap.read( )
        if not ret:
            break

        # the model expects the image to be 640x640
        # the following code resizes the image to 640x640 and adds black pixels to any remaining space
        [ height, width, _ ] = original_image.shape
        length = max( (height, width) )
        image = np.zeros( (length, length, 3), np.uint8 )
        image[ 0:height, 0:width ] = original_image
        scale = length / 640
        blob = cv2.dnn.blobFromImage( image, scalefactor = 1 / 255, size = (640, 640), swapRB = True )

        # actually feed the image to the model
        model.setInput( blob )
        outputs = model.forward( )

        # the model outputs an array of results that we need to process
        # and also transform the coordinates back to the original image size
        outputs = np.array( [ cv2.transpose( outputs[ 0 ] ) ] )
        rows = outputs.shape[ 1 ]
        boxes, scores, class_ids = [ ], [ ], [ ]
        for i in range( rows ):
            classes_scores = outputs[ 0 ][ i ][ 4: ]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc( classes_scores )
            if maxScore >= 0.25:
                box = [
                    outputs[ 0 ][ i ][ 0 ] - (0.5 * outputs[ 0 ][ i ][ 2 ]),
                    outputs[ 0 ][ i ][ 1 ] - (0.5 * outputs[ 0 ][ i ][ 3 ]),
                    outputs[ 0 ][ i ][ 2 ],
                    outputs[ 0 ][ i ][ 3 ]
                ]
                boxes.append( box )
                scores.append( maxScore )
                class_ids.append( maxClassIndex )

        # deduplicate multiple detections of the same object in the same location
        result_boxes = cv2.dnn.NMSBoxes( boxes, scores, 0.25, 0.45, 0.5 )

        # draw the box and label for each detection into the original image
        for i in range( len( result_boxes ) ):
            index = result_boxes[ i ]
            box = boxes[ index ]
            draw_bounding_box(
                original_image, class_ids[ index ], scores[ index ],
                round( box[ 0 ] * scale ), round( box[ 1 ] * scale ),
                round( (box[ 0 ] + box[ 2 ]) * scale ), round( (box[ 1 ] + box[ 3 ]) * scale ) )

        cv2.imshow( 'video', original_image )

        if cv2.waitKey( 1 ) & 0xFF == ord( 'q' ):
            break

    cap.release( )
    cv2.destroyAllWindows( )


if __name__ == '__main__':
    parser = argparse.ArgumentParser( )
    parser.add_argument( '--model', default = 'yolov8n.onnx', help = 'Input your onnx model.' )
    parser.add_argument( '--video', default = str( './videos/video.mp4' ), help = 'Path to input video.' )
    args = parser.parse_args( )
    main( args.model, args.video )
