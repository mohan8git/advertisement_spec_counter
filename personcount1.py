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
import dlib                                                                                                                             #--------------------------------------------
import threading
import time
nv = []
count=0
temp=[]
total_p=[]

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #-------------------------------------------

OUTPUT_SIZE_WIDTH = 720
OUTPUT_SIZE_HEIGHT = 720                                                                                            #--------------------------------------------

def FindPoint(left, top,right, bottom, cx, cy) :
    if (cx > left and cx < right and 
        cy > top and cy < bottom) :
        return True
    else : 
        return False


def  centre(top,right,bottom,left):
    y = bottom + int((top-bottom)*0.5)
    x = left + int((right - left)*0.5)
    return  [x,y]

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

    def doRecognizePerson(faceNames, fid):    #------------------------------------------------------------------------------
        time.sleep(2)
        faceNames[ fid ] = "Person " + str(fid)         #-----------------------------------------------------------------------------

    #Start the window thread for the two windows we are using
    cv2.startWindowThread()

    #The color of the rectangle we draw around the face
    rectangleColor = (0,165,255)

    #variables holding the current frame number and the current faceid
    frameCounter = 0
    currentFaceID = 0

    #Variables holding the correlation trackers and the name per faceid
    faceTrackers = {}
    faceNames = {}
      
    def track(self):
        count=0
        
        global temp
        #import url
        import numpy as np
        import cv2
        
        #Create two opencv named windows
        #cv2.namedWindow("base-image", cv2.WINDOW_AUTOSIZE)             #--------------------------------------------------------------
       # cv2.namedWindow("result-image", cv2.WINDOW_AUTOSIZE)           #--------------------------------------------------------------

        #Position the windows next to eachother
        #cv2.moveWindow("base-image",0,100)                                                     #--------------------------------------------------------------
        #cv2.moveWindow("result-image",400,100)                                             #---------------------------------------------------------------
        #cap = cv2.VideoCapture('http://192.168.137.61:80/')#.dtype('uint32')
        cap = cv2.VideoCapture(0)

         #Start the window thread for the two windows we are using
        cv2.startWindowThread()

        #The color of the rectangle we draw around the face
        rectangleColor = (0,165,255)

        #variables holding the current frame number and the current faceid
        frameCounter = 0
        currentFaceID = 0

        #Variables holding the correlation trackers and the name per faceid
        faceTrackers = {}
        faceNames = {}

        
        while True:
            ret,frame = cap.read()
            frame=cv2.flip(frame,1)
            frame=cv2.resize(frame,(1080,1080))
            #baseImage = cv2.resize( frame, ( 1080, 1080))   #-------------------------------------------------------
            #frame = baseImage[:, :, ::-1]
            face_locations = face_recognition.face_locations(frame)
            #print("Face location: ",face_locations)
            
            #for top, right, bottom, left in face_locations:
            #    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                
            #resultImage = baseImage.copy()          #----------------------------------------------------------------


            frameCounter += 1       #------------------------------------------------------------------------------

            
            bbox_xywh, cls_conf, cls_ids = self.detector(frame)
            #cv2.putText(frame,str(bbox_xywh),(0,15), cv2.FONT_HERSHEY_PLAIN,1,(255,255,255) ,2)
            if bbox_xywh is not None:
                #print("No of live viewers: ",len(face_locations))
                count=0
                for top, right, bottom, left in face_locations:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
##                print(bbox_xywh)  
                #print("Face location: ",face_locations)
                    
                print("No of live viewers: ",len(face_locations))
                temp1=[]
                for i in range(len(face_locations)):
                    t1,t2,t3,t4=face_locations[i]
                    temp=centre(t1,t2,t3,t4)
                    temp1.append(temp)
                    #print("Centre of face",temp)
                    #print("Face loactions "+str(face_locations)+"Person: "+str(bbox_xywh))
                    #flag=1
                mask = cls_ids==0
                bbox_xywh[:,3:] *= 1.2 # bbox dilation just in case bbox too small
##                print("Bbox: ",bbox_xywh)
                cls_conf = cls_conf[mask]
                #print("person's cord.: ",bbox_xywh)
                #print("Face location: ",face_locations)
                #print("No of live viewers: ",len(face_locations))
                # do tracking
                outputs = self.deepsort.update(bbox_xywh, cls_conf, frame)  #left,top,right,bottom
                
                #print("Outputs: ",outputs)
                #print(len(outputs))
                for i in  range(len(outputs)):
                 if outputs[i][4] not in total_p:
                     total_p.append(outputs[i][4])
                     #print(temp)
                print("Total No of people: ",len(outputs))
                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:,:-1]
                    identities = outputs[:,-1]
                    frame = draw_boxes(frame, bbox_xyxy, identities)
                
                for i in range(len(outputs)):
                    for j in range(len(temp1)):
                        a,b,c,d,e= outputs[i]
                        flag=FindPoint(a,b,c,d,temp1[j][0],temp1[j][1])
                        #print(flag)
                        if flag:
                            if e not in nv:
                                nv.append(e)
                            #print("nv",nv)
                    
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        #print(count)  
        cap.release()
        cv2.destroyAllWindows()            
        #print("No of People detected : ",len(temp))
        #print(temp)
        return [nv,total_p]

if __name__=="__main__":
    cfg = get_config()
    cfg.merge_from_file("./configs/yolov3.yaml")
    cfg.merge_from_file("./configs/deep_sort.yaml")

    with VideoTracker(cfg) as vdo_trk:
        p = vdo_trk.track()
        print("Total no of person who viewed advertisement "+str(len(nv))+" \n Total no of persons who passed by the advertisement board "+str(len(total_p)))
        
