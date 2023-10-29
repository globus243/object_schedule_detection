import sys
import urllib.request


def healthcheck( ):
    try:
        response = urllib.request.urlopen( "http://localhost/video", timeout = 5 )
        if response.status == 200:
            return True
    except:
        None

    return False


if __name__ == "__main__":
    if healthcheck( ):
        sys.exit( 0 )
    else:
        sys.exit( 1 )
