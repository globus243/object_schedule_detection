import plotly.figure_factory as ff
import plotly.graph_objects as go  # Importieren Sie die erforderliche Bibliothek


def create_figure( rows ):
    gantt_data = [ ]
    current_activity = None
    start_time = None

    for row in rows:
        timestamp, action = row[ 1 ], row[ 2 ]  # index 1 for timestamp, index 2 for event
        timestamp = timestamp.strftime( "%Y-%m-%d %H:%M:%S.%f" )  # Convert to string with microsecond precision

        if "moved to" in action:
            if current_activity is not None:
                end_time = timestamp
                gantt_data.append(
                        dict( Task = current_activity, Start = start_time, Finish = end_time )
                )
            current_activity = f'cat on {action.split( )[ -1 ]}'
            start_time = timestamp
        elif "left from" in action and current_activity is not None:
            end_time = timestamp
            gantt_data.append(
                    dict( Task = current_activity, Start = start_time, Finish = end_time )
            )
            current_activity = None

    if not gantt_data:  # Überprüfen, ob gantt_data leer ist
        fig = go.Figure( )
        fig.update_layout(
            title = "Not enough data",
            xaxis_title = "Timestamp",
            yaxis_title = "Event",
            annotations = [
                dict(
                    x = 0.5,
                    y = 0.5,
                    xref = "paper",
                    yref = "paper",
                    text = "Not enough data to display",
                    showarrow = False,
                    font = dict(
                            size = 20
                    )
                )
            ]
        )
    else:
        fig = ff.create_gantt(
                gantt_data, show_colorbar = True, index_col = 'Task', group_tasks = True, showgrid_x = True,
                showgrid_y = True
        )
        # Add range slider
        fig.update_layout(
            title = "cat's activity",
            xaxis = dict(
                rangeselector = dict(
                    buttons = list(
                        [
                            dict(
                                count = 1,
                                label = "1d",
                                step = "day",
                                stepmode = "backward" ),
                            dict(
                                count = 7,
                                label = "1w",
                                step = "day",
                                stepmode = "backward" ),
                            dict(
                                count = 1,
                                label = "1m",
                                step = "month",
                                stepmode = "backward" ),
                            dict(
                                count = 6,
                                label = "6m",
                                step = "month",
                                stepmode = "backward" ),
                            dict(
                                count = 1,
                                label = "YTD",
                                step = "year",
                                stepmode = "todate" ),
                            dict( step = "all" )
                        ] )
                ),
                rangeslider = dict(
                        visible = True
                    ),
                type = "date"
            )
        )

    return fig


if __name__ == '__main__':
    import requests
    from datetime import datetime

    response = requests.get( "http://dh04:5000/status" )
    data = response.json( )
    # data = {
    #     "detections_log": [
    #         [
    #             1,
    #             "Tue, 03 Oct 2023 13:04:45 GMT",
    #             "cat moved to Eating"
    #         ]
    #     ],
    #     "detections_count": {
    #         "Eating": 1,
    #         "Toilet": 0
    #     }
    # }

    logs = data[ "detections_log" ]
    # convert timestamp to datetime object
    for log in logs:
        log[ 1 ] = datetime.strptime( log[ 1 ], "%a, %d %b %Y %H:%M:%S GMT" )

    create_figure( logs ).show( )
