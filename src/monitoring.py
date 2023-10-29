import datetime
import threading
import time
from collections import defaultdict

import cv2
import cv2.dnn
import numpy as np
from ultralytics.utils import yaml_load
from ultralytics.utils.checks import check_yaml

from camera import Camera

CLASSES = yaml_load( check_yaml( 'coco128.yaml' ) )[ 'names' ]
COLORS = np.random.uniform( 0, 255, size = (len( CLASSES ), 3) )


def draw_bounding_box( img, color, class_id, confidence, x, y, x_plus_w, y_plus_h ):
    label = f'{CLASSES[ class_id ]} ({confidence:.2f})'
    color = color if color is not None else COLORS[ class_id ]
    cv2.rectangle( img, (x, y), (x_plus_w, y_plus_h), color, 2 )
    cv2.putText( img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2 )


def compute_color(start_color, end_color, ratio):
    return (
        int(start_color[0] * (1 - ratio) + end_color[0] * ratio),
        int(start_color[1] * (1 - ratio) + end_color[1] * ratio),
        int(start_color[2] * (1 - ratio) + end_color[2] * ratio)
    )


class Monitoring:
    def __init__( self, yolo_model, stream_url, image_mid, titles, class_ids, persistence_callback = None ):
        self.yolo_model = yolo_model
        self.stream_url = stream_url
        self.threshold_ratio = image_mid
        self.titles = titles
        self.class_ids = class_ids
        self.frame_lock = threading.Lock( )
        self.output_frame = None
        self.detections_count = { titles[ 'left' ]: 0, titles[ 'right' ]: 0 }
        self.detections_log = [ ]
        self.capture_thread = threading.Thread( target = self.capture_thread_func )
        self.capture_thread.start( )
        self.last_known_position = None
        self.track_history = defaultdict( lambda: [ ] )
        self.persistence_callback = persistence_callback
        self.last_motion_detected = None
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=25, detectShadows = False )

    def capture_thread_func( self ):
        cap = Camera( self.stream_url )
        model = cv2.dnn.readNetFromONNX( self.yolo_model )
        frame_counter = 0
        last_target_timestamp = None

        while cap.is_opened( ):
            frame_counter += 1

            ret, frame = cap.read( )

            if ret:
                motion_detected = self.motion_detected(frame)
                target_detected = False
                if motion_detected:
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
                        if maxScore >= 0.75 and maxClassIndex in self.class_ids:
                            box = [
                                outputs[ 0 ][ i ][ 0 ] - (0.5 * outputs[ 0 ][ i ][ 2 ]),
                                outputs[ 0 ][ i ][ 1 ] - (0.5 * outputs[ 0 ][ i ][ 3 ]),
                                outputs[ 0 ][ i ][ 2 ], outputs[ 0 ][ i ][ 3 ] ]
                            boxes.append( box )
                            scores.append( maxScore )
                            class_ids.append( maxClassIndex )

                    result_boxes = cv2.dnn.NMSBoxes( boxes, scores, 0.25, 0.45, 0.5 )
                    target_detected = any( label in self.class_ids for label in class_ids )

                if target_detected:
                    last_target_timestamp = time.time()
                    self.last_motion_detected = datetime.datetime.now()
                    for i in range(len(boxes)):
                        if i in result_boxes:
                            box = boxes[i]
                            draw_bounding_box(
                                    frame,
                                    (0, 255, 0),
                                    class_ids[i],
                                    scores[i],
                                    round(box[0] * scale),
                                    round(box[1] * scale),
                                    round((box[0] + box[2]) * scale),
                                    round((box[1] + box[3]) * scale))
                            center_x = round((box[0] + box[2] / 2) * scale)
                            center_y = round((box[1] + box[3] / 2) * scale)
                            self.track_history[class_ids[i]].append((center_x, center_y))
                            current_position = self.titles['left'] if center_x < frame.shape[1] * self.threshold_ratio else self.titles['right']

                            # only count cat if in  at least 10 frames
                            if not len(self.track_history[class_ids[i]]) > 10:
                                continue

                            if self.last_known_position is None or self.last_known_position != current_position:
                                self.detections_count[current_position] += 1
                                self.last_known_position = current_position
                                log_line = f'{datetime.datetime.now()} - cat moved to {current_position}'
                                print(log_line)
                                self.detections_log.append(log_line)

                                if self.persistence_callback is not None:
                                    self.persistence_callback(log_line)

                else:
                    if last_target_timestamp is not None and time.time() - last_target_timestamp >= 10:
                        log_line = f'{datetime.datetime.now()} - cat left from {self.last_known_position}'
                        print(log_line)
                        self.detections_log.append(log_line)
                        self.track_history = defaultdict( lambda: [ ] )
                        self.last_known_position = None
                        last_target_timestamp = None

                        if self.persistence_callback is not None:
                            self.persistence_callback(log_line)

                for class_id, track in self.track_history.items():
                    if len(track) > 1:
                        for i in range(1, len(track)):
                            ratio = i / len(track)
                            color = compute_color((0, 0, 255), (0, 255, 0), ratio)  # from red to green
                            cv2.line(frame, tuple(track[i - 1]), tuple(track[i]), color, 2)

                threshold_x = int(frame.shape[1] * self.threshold_ratio)
                cv2.line(frame, (threshold_x, 0), (threshold_x, frame.shape[0]), (255, 255, 255), 2)
                cv2.putText(frame, self.titles['left'], (threshold_x - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(frame, self.titles['right'], (threshold_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

                toilet_counter_text = f'{self.detections_count[self.titles["left"]]}'
                eating_counter_text = f'{self.detections_count[self.titles["right"]]}'

                toilet_text_size = cv2.getTextSize(toilet_counter_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                eating_text_size = cv2.getTextSize(eating_counter_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

                toilet_text_x = threshold_x - 100 + (cv2.getTextSize(self.titles['left'], cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0] - toilet_text_size[0]) // 2
                eating_text_x = threshold_x + 10 + (cv2.getTextSize(self.titles['right'], cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0] - eating_text_size[0]) // 2

                cv2.putText(frame, toilet_counter_text, (toilet_text_x, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                cv2.putText(frame, eating_counter_text, (eating_text_x, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

                if motion_detected:
                    cv2.putText(frame, 'AI', (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                self.update_output_frame(frame)

            elif not ret:
                break

            time.sleep( 0.033 )  # 30fps

    def update_output_frame( self, frame ):
        with self.frame_lock:
            self.output_frame = frame

    def get_frame( self ):
        while True:
            time.sleep( 0.033 )  # 30fps
            with self.frame_lock:
                if self.output_frame is None:
                    continue
            _, buffer = cv2.imencode( '.jpg', self.output_frame, [ int( cv2.IMWRITE_JPEG_QUALITY ), 70 ] )
            frame = buffer.tobytes( )
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def get_status( self ):
        return self.detections_count

    def get_log( self ):
        return self.detections_log

    def motion_detected(self, frame):
        frame_without_timestamp = frame.copy()
        frame_without_timestamp[0:50, :] = 0  # blackout timestamp
        fgmask = self.background_subtractor.apply(frame_without_timestamp)

        count = cv2.countNonZero(fgmask)

        if count > 3750:
            self.last_motion_detected = datetime.datetime.now()
            return True

        if self.last_motion_detected is not None and (datetime.datetime.now() - self.last_motion_detected).seconds <= 10:
            return True

        return False
