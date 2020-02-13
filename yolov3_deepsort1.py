import os
import cv2
import time
import argparse
import torch
import numpy as np

from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
count=0
temp=[]


class VideoTracker(object):
    def __init__(self, cfg):
        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            raise UserWarning("Running in cpu mode!")


        
        self.detector = build_detector(cfg, use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names


    def __enter__(self):
        return self

    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)
        
    def track(self):
        #import url
        import numpy as np
        import cv2
        #cap = cv2.VideoCapture('http://192.168.137.61:80/')#.dtype('uint32')
        cap = cv2.VideoCapture(0)
        while True:
            ret,frame = cap.read()
            bbox_xywh, cls_conf, cls_ids = self.detector(frame)
            if bbox_xywh is not None:
                mask = cls_ids==0
                #if(mask==cls_ids==0):
                  #  count+=1
##                bbox_xywh = bbox_xywh[mask]
                bbox_xywh[:,3:] *= 1.2 # bbox dilation just in case bbox too small
                cls_conf = cls_conf[mask]

                # do tracking
                outputs = self.deepsort.update(bbox_xywh, cls_conf, frame)
                #print(len(outputs))
                for i in  range(len(outputs)):
                 if outputs[i][4] not in temp:
                     temp.append(outputs[i][4])
                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:,:-1]
                    identities = outputs[:,-1]
                    frame = draw_boxes(frame, bbox_xyxy, identities)
                    
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
          
        cap.release()
        cv2.destroyAllWindows()            
        print("No of viewers: ",len(temp))
        print(temp)

if __name__=="__main__":
    cfg = get_config()
    cfg.merge_from_file("./configs/yolov3.yaml")
    cfg.merge_from_file("./configs/deep_sort.yaml")

    with VideoTracker(cfg) as vdo_trk:
        vdo_trk.track()
