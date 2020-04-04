import cv2
import numpy as numpy
import sys
import darknet

CONFIG = "cfg/yolov3.cfg"
WEIGHTS = "weights/yolov3.weights"
DATA = "cfg/coco.data"

THRESH = 0.2

class DetectImages():
    def __init__(self):
        self.net = darknet.load_net_custom(CONFIG, WEIGHTS, 0, 1)
        self.meta = darknet.load_meta(DATA)
    
    def display_detection(self):
        cv_img = cv2.imread("/home/o.guengoer/syn/darknet/data/dog.jpg")
        detections = darknet.detect_image(self.net, self.meta, cv_img, THRESH)
        
        for detection in detections:
            label, confidence, coords = detection
            xmin = int(coords[0] - coords[2]/2)
            xmax = int(coords[0] + coords[2]/2)
            ymin = int(coords[1] - coords[3]/2)
            ymax = int(coords[1] + coords[3]/2)
            cv2.rectangle(cv_img, (xmin,ymin), (xmax,ymax), (0,255,0), 1)
            cv2.putText(cv_img, label, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX , 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow("window", cv_img)
        cv2.waitKey(0)

if __name__ == '__main__':
    di = DetectImages()
    di.display_detection()
