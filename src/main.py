from datetime import datetime, timedelta
import os
import threading

import flask
import plotly.io as pio
from flask import request
from waitress import serve

from monitoring import Monitoring
from persistence import Persistence
from plot import create_figure

stream_url = os.environ.get( 'RTSP_STREAM', None )

if stream_url is None:
    print( 'RTSP_STREAM environment variable is not set' )
    exit( 1 )

app = flask.Flask( __name__ )

db_host = os.environ.get( 'MYSQL_HOST', None )
db_port = int( os.environ.get( 'MYSQL_PORT', 3306 ) )
db_database = os.environ.get( 'MYSQL_DATABASE', None )
db_user = os.environ.get( 'MYSQL_USER', None )
db_password = os.environ.get( 'MYSQL_PASSWORD', None )

persistence = None

if None not in [ db_host, db_database, db_user, db_password ]:
    persistence = Persistence.get_instance(
            host = os.environ.get( 'DB_HOST', 'mariadb' ),
            port = int( os.environ.get( 'DB_PORT', 3306 ) ),
            database = os.environ.get( 'DB_DATABASE', 'cat' ),
            user = os.environ.get( 'DB_USER', 'cat' ),
            password = os.environ.get( 'DB_PASSWORD', 'cat' ) )

monitor = Monitoring(
        "model.onnx",
        stream_url,
        0.6,
        titles = { "left": "Toilet", "right": "Eating" },
        class_ids = [ 15 ],
        persistence_callback = persistence.save_data if persistence else None )


@app.route( '/video' )
def video( ):
    return flask.Response( monitor.get_frame( ), mimetype = 'multipart/x-mixed-replace; boundary=frame' )


# noinspection DuplicatedCode
@app.route( '/status' )
def detections( ):
    start_date = request.args.get( 'start_date', None )
    end_date = request.args.get( 'end_date', None )

    if start_date is None or end_date is None:
        end_date = datetime.now( )
        start_date = end_date - timedelta( hours = 24 )
    else:
        start_date = datetime.fromtimestamp( int( start_date ) )
        end_date = datetime.fromtimestamp( int( end_date ) )

    if persistence:
        detections_log = persistence.get_data(
                from_datetime = start_date.strftime( '%Y-%m-%d %H:%M:%S' ),
                to_datetime = end_date.strftime( '%Y-%m-%d %H:%M:%S' ) )
    else:
        detections_log = monitor.detections_log

    return flask.jsonify(
            { 'detections_count': monitor.get_status( ),
              'detections_log':   detections_log } )


@app.route( '/plot' )
def stats( ):
    start_date = request.args.get( 'start_date', None )
    end_date = request.args.get( 'end_date', None )

    if start_date is None or end_date is None:
        end_date = datetime.now( )
        start_date = end_date - timedelta( hours = 24 )
    else:
        start_date = datetime.fromtimestamp( int( start_date ) )
        end_date = datetime.fromtimestamp( int( end_date ) )

    if persistence:
        data = persistence.get_data(
                from_datetime = start_date.strftime( '%Y-%m-%d %H:%M:%S' ),
                to_datetime = end_date.strftime( '%Y-%m-%d %H:%M:%S' ) )
    else:
        raw_log = monitor.get_log( )
        data = [ ]
        for i, log_entry in enumerate( raw_log ):
            timestamp, event = log_entry.split( ' - ' )
            data.append(
                    [ i,
                      datetime.strptime( timestamp, '%Y-%m-%d %H:%M:%S.%f' ),
                      event
                      ] )

    fig = create_figure( data )
    return pio.to_html( fig, full_html = True, include_plotlyjs = 'cdn' )


if __name__ == '__main__':
    capture_thread = threading.Thread( target = monitor.capture_thread_func )
    capture_thread.start( )
    serve( app, host = "0.0.0.0", port = 5000, threads = 5 )
