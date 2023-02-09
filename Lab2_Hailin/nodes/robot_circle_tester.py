#!/usr/bin/env python3
#Standard Libraries
import numpy as np
import pygame
import time
import pygame_utils
import matplotlib.image as mpimg
from skimage.draw import disk
from scipy.linalg import block_diag

def point_to_cell(p):
    return p*10

def points_to_robot_circle(points):
        #Convert a series of [x,y] points to robot map footprints for collision detection
        #Hint: The disk function is included to help you with this function

        # For each point that robot is expected draw a circle
        # Then then draw a circle with all pixel locations that robot covers
        robot_radius = 10

        robot_locations_rr = list()
        robot_locations_cc = list()

        for p in points:
            
            pixel = point_to_cell(p)
            rr, cc = disk((pixel[0], pixel[1]), robot_radius)
            robot_locations_rr.append(rr)
            robot_locations_cc.append(cc)

        # print("TO DO: Implement a method to get the pixel locations of the robot path")

        return robot_locations_rr, robot_locations_cc
        #return [], []


points = [[0,2],[3,4],[1,10]]
rr, cc = points_to_robot_circle(points)
print("rr, cc")
print(rr)
print(cc)