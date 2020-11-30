import cv2
import numpy as np
from math import pow, sqrt
import threading
# import requests

def calcDistance(boundBoxHeight):
    distance = (calculateConstant_x * calculateConstant_y) / boundBoxHeight
    return distance

calcDistance()