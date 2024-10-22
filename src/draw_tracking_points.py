import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle

import numpy as np
import cv2
from datetime import datetime

import inspect

from src import params

def draw_points(team1_players: list, team2_players: list, team_players: list, ball_points: list):

    team1_players, team2_players = team2_players, team1_players

    function = eval(inspect.stack()[0][3])

    try:
        fig = function.fig
        ax = function.ax

        min_x = function.min_x
        max_x = function.max_x
        max_y = function.max_y
        min_y = function.min_y

        volleyball_field = function.volleyball_field

        volleyball_net_x = function.volleyball_net_x
        volleyball_net_y = function.volleyball_net_y

    except:

        # Init plot (only once)
        volleyball_field = params.VOLLEYBALL_FIELD
        volleyball_field = Polygon(volleyball_field, closed=True, edgecolor=(0, 0, 0), facecolor='none')

        volleyball_net_x = [point[0] for point in params.VOLLEYBALL_NET]
        volleyball_net_y = [103, 1260]

        min_x = min([point[0] for point in params.VOLLEYBALL_FIELD])
        max_x = max([point[0] for point in params.VOLLEYBALL_FIELD])
        min_y = min([point[1] for point in params.VOLLEYBALL_FIELD])
        max_y = max([point[1] for point in params.VOLLEYBALL_FIELD])

        fig, ax = plt.subplots(figsize=(10, 5))

        function.fig = fig
        function.ax = ax

        function.min_x = min_x
        function.max_x = max_x
        function.max_y = max_y
        function.min_y = min_y

        function.volleyball_field = volleyball_field

        function.volleyball_net_x = volleyball_net_x
        function.volleyball_net_y = volleyball_net_y

        function.prev_team1_players = team1_players
        function.prev_team2_players = team2_players
        function.prev_team_players = team_players
        function.prev_ball_points = ball_points

    if not len(ball_points):
        ball_points = function.prev_ball_points

    # Plot
    ax.clear()

    ax.set_xlim(0, max_x + min_x)
    ax.set_ylim(0, max_y + min_y)
    ax.set_aspect('equal')
    ax.invert_yaxis()

    plt.axis("off")
    plt.show(block=False)

    # Plot items
    ax.add_patch(volleyball_field)
    ax.plot(volleyball_net_x, volleyball_net_y, color=(0,0,0), alpha=1)

    if len(team_players):
        for player in team_players:
            for index, point in enumerate(player):

                if index != 0:
                    ax.plot((point[0], player[index-1][0]), (point[1], player[index-1][1]), color=tuple([channel / 255 for channel in params.TEAM_DEFAULT_COLOR]), alpha=1)
                
                circle = Circle(point, 10, color=tuple([channel / 255 for channel in params.TEAM_DEFAULT_COLOR]), fill=True)
                ax.add_patch(circle)
    else:
        for player in team1_players:
            for index, point in enumerate(player):

                if index != 0:
                    ax.plot((point[0], player[index-1][0]), (point[1], player[index-1][1]), color=tuple([channel / 255 for channel in params.TEAM1_COLOR]), alpha=1)
                
                circle = Circle(point, 10, color=tuple([channel / 255 for channel in params.TEAM1_COLOR]), fill=True)
                ax.add_patch(circle)

        for player in team2_players:
            for index, point in enumerate(player):

                if index != 0:
                    ax.plot((point[0], player[index-1][0]), (point[1], player[index-1][1]), color=tuple([channel / 255 for channel in params.TEAM2_COLOR]), alpha=1)
                
                circle = Circle(point, 10, color=tuple([channel / 255 for channel in params.TEAM2_COLOR]), fill=True)
                ax.add_patch(circle)

    for index, point in enumerate(ball_points):

        if index != 0:
            ax.plot((point[0], ball_points[index-1][0]), (point[1], ball_points[index-1][1]), color=tuple([channel / 255 for channel in params.BALL_COLOR]), alpha=1)
        
        circle = Circle(point, 10, color=tuple([channel / 255 for channel in params.BALL_COLOR]), fill=True)
        ax.add_patch(circle)

    function.prev_team1_players = team1_players
    function.prev_team2_players = team2_players
    function.prev_team_players = team_players
    function.prev_ball_points = ball_points

    # Show plot
    fig.canvas.draw()

    fig.savefig(f'plot/{int(datetime.now().timestamp())}.png', bbox_inches='tight')

