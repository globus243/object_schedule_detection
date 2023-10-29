import logging
from contextlib import contextmanager

import mysql.connector
from mysql.connector import Error

logger = logging.getLogger( "catLogger" )


class Persistence:
    _instance = None

    @staticmethod
    def get_instance( *args, **kwargs ):
        if Persistence._instance is None:
            Persistence._instance = Persistence( *args, **kwargs )
        return Persistence._instance

    def __init__( self, host = "mariadb", port = 3306, user = "cat", password = "cat", database = "cat" ):
        if Persistence._instance is not None:
            raise Exception( "This class is a Singleton! Use 'get_instance()' to access it." )
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self._create_tables( )

    @contextmanager
    def _get_connection( self ):
        conn = None
        try:
            conn = mysql.connector.connect(
                    host = self.host,
                    port = self.port,
                    user = self.user,
                    password = self.password,
                    database = self.database
            )
            if conn.is_connected( ):
                yield conn
            else:
                raise Exception( "Connection not established" )
        except Error as e:
            print( f"Error connecting to MySQL: {e}" )
        finally:
            if conn and conn.is_connected( ):
                conn.close( )

    def _create_tables( self ):
        with self._get_connection( ) as conn:
            cursor = conn.cursor( )
            cursor.execute(
                    '''
                    CREATE TABLE IF NOT EXISTS detections (
                        id INTEGER AUTO_INCREMENT PRIMARY KEY,
                        timestamp DATETIME(6) NOT NULL,
                        event VARCHAR(255) NOT NULL
                    );
                    '''
            )
            conn.commit( )

    def save_data( self, log_entry ):
        with self._get_connection( ) as conn:
            cursor = conn.cursor( )

            timestamp, event = log_entry.split( ' - ' )
            cursor.execute(
                    '''
                    INSERT INTO detections (timestamp, event)
                    VALUES (%s, %s)
                    ''', (timestamp, event)
            )
            conn.commit( )

    def get_data( self, from_datetime, to_datetime ):
        with self._get_connection( ) as conn:
            logger.info( f"Getting data from {from_datetime} to {to_datetime}" )
            cursor = conn.cursor( )
            cursor.execute(
                    '''
                    SELECT * FROM detections
                    WHERE timestamp BETWEEN %s AND %s
                    ORDER BY timestamp
                    ''', (from_datetime, to_datetime)
            )
            rows = cursor.fetchall( )
            logger.info( f"Got {len( rows )} rows" )
            return rows
