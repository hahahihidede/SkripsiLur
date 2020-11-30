import cv2
import numpy as np
from math import pow, sqrt
# import requests

def centroid(startX,endX,startY,endY):
    centroid_x = round((startX+endX)/2,4)
    centroid_y = round((startY+endY)/2,4)
    boundBoxHeight = round(endY-startY,4)
    return centroid_x,centroid_y,boundBoxHeight
centroid()
