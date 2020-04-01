import cv2
import numpy as np
import grpc
from concurrent import futures

from api import darknet_detection_pb2
from api import darknet_detection_pb2_grpc

import darknet

CONFIG = "/home/o.guengoer/syn/darknet/cfg/yolov3.cfg"
WEIGHTS = "/home/o.guengoer/syn/darknet/yolov3.weights"
DATA = "/home/o.guengoer/syn/darknet/cfg/coco.data"
THRESH = 0.25

class DarknetDetectionServicer(darknet_detection_pb2_grpc.DarknetDetectionServicer):
    
    def __init__(self):
        self.net =  darknet.load_net_custom(CONFIG, WEIGHTS, 0, 1) # batch size = 1
        self.meta = darknet.load_meta(DATA)
        print("initialized darknet service")

    def darknetDetection(self, request, context):
        np_img = np.fromstring(request.data, np.uint8)
        cv_img = cv2.imdecode(np_img, cv2.COLOR_BGR2RGB)
        print("received image")
        print(cv_img)

        detections = darknet.detect_image(self.net, self.meta, cv_img, THRESH)
        print(detections)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    darknet_detection_pb2_grpc.add_DarknetDetectionServicer_to_server(DarknetDetectionServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            continue
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == "__main__":
    serve()
