import cv2
import numpy as np
import threading
from math import pow, sqrt
from drawResult import drawResult
from calcDistance import calcDistance
from centroid import centroid
from clahe import CLAHE

# import requests

def main():
    preprocessing = False
    calculateConstant_x = 300
    calculateConstant_y = 615
    personLabelID = 15.00
    debug = True
    accuracyThreshold = 0.4
    RED = (0,0,255)
    YELLOW = (0,255,255)
    GREEN = (0,255,0)
    BLACK = (0,0,0)
    write_video = False


    if __name__== "__main__":

        caffeNetwork = cv2.dnn.readNetFromCaffe("./SSD_MobileNet_prototxt.txt", "./SSD_MobileNet.caffemodel")
        cap = cv2.VideoCapture(0)
        # cap = cv2.VideoCapture("./vp.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        # output_movie = cv2.VideoWriter("./result.avi", fourcc, 24, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))


        while cap.isOpened():

            debug_frame, frame = cap.read()
            highRisk = set()
            mediumRisk = set()
            position = dict()
            detectionCoordinates = dict()

            if not debug_frame:
                # print("Video Error Gaes")
                break
                
            if preprocessing:
                frame = CLAHE(frame)
                print(frame)

            (imageHeight, imageWidth) = frame.shape[:2]
            pDetection = cv2.dnn.blobFromImage(cv2.resize(frame, (imageWidth, imageHeight)), 0.007843, (imageWidth, imageHeight), 127.5)

            caffeNetwork.setInput(pDetection)
            detections = caffeNetwork.forward()

            for i in range(detections.shape[2]):

                accuracy = detections[0, 0, i, 2]
                if accuracy > accuracyThreshold:

                    idOfClasses = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([imageWidth, imageHeight, imageWidth, imageHeight])
                    (startX, startY, endX, endY) = box.astype('int')

                    if idOfClasses == personLabelID:

                        boundBoxDefaultColor = (255,255,255)
                        cv2.rectangle(frame, (startX, startY), (endX, endY), boundBoxDefaultColor, 2)
                        detectionCoordinates[i] = (startX, startY, endX, endY)


                        centroid_x, centroid_y, boundBoxHeight = centroid(startX,endX,startY,endY)                    
                        distance = calcDistance(boundBoxHeight)

                        centroid_x_centimeters = (centroid_x * distance) / calculateConstant_y
                        centroid_y_centimeters = (centroid_y * distance) / calculateConstant_y
                        position[i] = (centroid_x_centimeters, centroid_y_centimeters, distance)


            for i in position.keys():
                for j in position.keys():
                    if i < j:
                        distanceOfboundBoxes = sqrt(pow(position[i][0]-position[j][0],2) 
                                            + pow(position[i][1]-position[j][1],2) 
                                            + pow(position[i][2]-position[j][2],2)
                                            )
                        if distanceOfboundBoxes < 150: # jarak tidak aman
                            highRisk.add(i),highRisk.add(j)
                        elif distanceOfboundBoxes < 200 > 150: # Jarak rawan
                            mediumRisk.add(i),mediumRisk.add(j) 
        

            cv2.putText(frame, "Sangat Rawan Penularan : " + str(len(highRisk)) , (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, "Rawan Penularan : " + str(len(mediumRisk)) , (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(frame, "Jumlah Terdeteksi : " + str(len(detectionCoordinates)), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            drawResult(frame, position)
            if write_video:            
                output_movie.write(frame)
            cv2.imshow('Dashboard Satpam', frame)
            waitkey = cv2.waitKey(1)
            if waitkey == ord("q"):
                break
main()
        # cap.release()
        # cv2.destroyAllWindows()

 