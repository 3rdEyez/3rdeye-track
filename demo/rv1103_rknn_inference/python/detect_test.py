from utils import openvino_yunet_detector,rknnsim_yunet_detector, visualize
import os
import time
import cv2 as cv


def main():
    detector = rknnsim_yunet_detector(
        "../../../models/onnx/yunet_n_320_320.onnx", 
        "rv1103", (320, 320), conf_thresh=0.8, iou_thresh=0.45)
    img = cv.imread("../../../media/person.jpg")
    faces = detector(img)  # faces: None, or nx15 np.array
    img = visualize(img, faces)
    cv.imwrite("../../../media/person_rknn_inference_result.jpg", img)
    
    detector = openvino_yunet_detector(
        "../../../models/onnx/yunet_n_320_320.onnx",
        (320, 320), conf_thresh=0.8, iou_thresh=0.45)
    img = cv.imread("../../../media/person.jpg")
    faces = detector(img)  # faces: None, or nx15 np.array
    img = visualize(img, faces)
    cv.imwrite("../../../media/person_onnx_inference_result.jpg", img)

if __name__ == '__main__':
    main()

