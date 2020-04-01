import cv2
import numpy as np
import grpc
from concurrent import futures
import sys,os

from api import darknet_detection_pb2
from api import darknet_detection_pb2_grpc

class DarknetClient():
    def __init__(self):
        self._darknet_channel = grpc.insecure_channel('[::]:50051')
        self._darknet_stub = darknet_detection_pb2_grpc.DarknetDetectionStub(self._darknet_channel)
        print("initialized darknet channel and stub")
    
    def test(self):
        cv_img = cv2.imread("/home/o.guengoer/Pictures/dog.jpg")
        _ , img_jpg = cv2.imencode('.jpg', cv_img)
        image_msg = darknet_detection_pb2.Image(data=img_jpg.tostring())
        self._darknet_stub.darknetDetection(image_msg)
        print("sent image to the service")

if __name__ == "__main__":
    dc = DarknetClient()
    dc.test()

