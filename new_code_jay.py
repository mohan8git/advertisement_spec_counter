import os
import cv2
import time
import argparse
import torch
import numpy as np
import face_recognition
from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
import dlib
import threading
import time
from timeit import default_timer as timer
import asyncio
from numba import jit,prange,cuda,njit

nv, temp, total_p, count = [], [],[], 0
OUTPUT_SIZE_WIDTH, OUTPUT_SIZE_HEIGHT = 720, 720

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

@jit( parallel = True)
def face_R(frame):
    face_locations = face_recognition.face_locations(frame)
    face_locations = np.array(face_locations)
    return face_locations

@jit(["(int32,int32,int32,int32,int32,int32)"],parallel = True)
def FindPoint(left, top, right, bottom, cx, cy):
    if left < cx < right and top < cy < bottom:
        return True
    else:
        return False

@jit(["(int32,int32,int32,int32)"], parallel = True)
def centre(top, right, bottom, left):
    y= bottom + int((top - bottom) * 0.5)
    x = left + int((right - left) * 0.5)
    return x, y
                

class VideoTracker(object):
    def __init__(self, cfg):
        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            raise UserWarning("Running in cpu mode!")

        self.detector = build_detector(cfg, use_cuda=True)
        self.deepsort = build_tracker(cfg, use_cuda=True)
        self.class_names = self.detector.class_names

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def doRecognizePerson(faceNames, fid):
        time.sleep(2)
        faceNames[fid] = "Person " + str(fid)

    # Start the window thread for the two windows we are using
    cv2.startWindowThread()

    # The color of the rectangle we draw around the face
    rectangleColor = (0, 165, 255)

    # variables holding the current frame number and the current faceid
    frameCounter, currentFaceID = 0, 0

    # Variables holding the correlation trackers and the name per faceid
    faceTrackers, faceNames = {}, {}

    def track(self):
        print("inside track")
        count = 0
        global temp
        cap = cv2.VideoCapture(0)

        # Start the window thread for the two windows we are using
        cv2.startWindowThread()

        # The color of the rectangle we draw around the face
        rectangleColor = (0, 165, 255)

        # variables holding the current frame number and the current faceid
        frameCounter, currentFaceID = 0, 0
        # Variables holding the correlation trackers and the name per faceid
        faceTrackers, faceNames, face_locations_len_previous, outputs_len_previous = {}, {}, 0, 0
        while True:
            full = timer()
            ret, frame = cap.read()
            gray_img=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
            face_locations=faceCascade.detectMultiScale(gray_img, scaleFactor=1.3,minNeighbors=4,minSize=(40,40))
            print(face_locations)
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (1080, 1080))

            frameCounter += 1
            half = timer()
            bbox_xywh, cls_conf, cls_ids = self.detector(frame)

            if (bbox_xywh is not None) & (frameCounter%2 ==0) :
                count, start = 0, timer()
                mask = cls_ids == 0
                bbox_xywh[:, 3:] *= 1.2

                cls_conf = cls_conf[mask]
                start = timer()
                outputs = self.deepsort.update(bbox_xywh, cls_conf, frame)
                outputs_length = len(outputs)
                start = timer()
                #face_locations = face_R(frame)
                print("FUNCTION maa atli waar laagi ", timer()-start)
                
                #face_locations_length = len(face_locations)
                face_locations_length=len(face_locations)
                start = timer()
                for top, right, bottom, left in face_locations:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                temp1 = []
                start = timer()
                if outputs_length > outputs_len_previous:
                    for i in range(len(face_locations)):
                        t1, t2, t3, t4 = face_locations[i]
                        temp = centre(t1, t2, t3, t4)
                        temp1.append(temp)
                    print("Centre Coordinmates",temp1)
                
                for i in range(len(outputs)):
                    if outputs[i][4] not in total_p:
                        total_p.append(outputs[i][4])

                start = timer()
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :-1]
                    identities = outputs[:, -1]
                    frame = draw_boxes(frame, bbox_xyxy, identities)
                start = timer()
                if outputs_length > outputs_len_previous:
                    print("yes,andar aavyo")
                    for i in range(len(outputs)):
                        if (outputs[i][4] not in nv):
                            print("che toh nai")
                            for j in range(len(temp1)):
                                a, b, c, d, e = outputs[i]
                                flag = FindPoint(a, b, c, d, temp1[j][0], temp1[j][1])
                                if flag == True:
                                    if e not in nv:
                                        print("navu print karyu")
                                        nv.append(e)
                                else:
                                    print("print nai thaay")
                        
                outputs_len_previous = outputs_length
            cv2.imshow("frame", frame)            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return [nv, total_p]

    def run_ml(self):
        return self.track()

if __name__ == "__main__":
    cfg = get_config()
    cfg.merge_from_file("./configs/yolov3.yaml")
    cfg.merge_from_file("./configs/deep_sort.yaml")

    #async def start():
    #   async with VideoTracker(cfg) as vdo_trk:
    #      p =  vdo_trk.track()
    #     print("Total no of person who viewed advertisement " + str(len(nv)) +  " \n Total no of persons who passed by the advertisement board " + str(len(total_p)))
    #print("before get event loop")
    #asyncio.get_event_loop().run_until_complete(start())
    vvd = VideoTracker(cfg)
    p = vvd.run_ml()
    print("Total no of person who viewed advertisement " + str(nv) +  " \n Total no of persons who passed by the advertisement board " + str(len(total_p)))
