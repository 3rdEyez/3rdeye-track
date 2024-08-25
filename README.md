# 3rdeye-track

This project is a sub-project of "the 3rd eye of Satori Komeiji in Touhou Project," aimed at achieving its target tracking capabilities.

SORTTracker提供一组符方法，能够方便地调用SORT跟踪算法，demo下的几个例子展示了它的使用方法。注意本项目依赖于OpenCV以及ncnn


demo使用方法：

```shell
git clone https://github.com/Fyra-BH/3rdeye-track.git
cd 3rdeye-track
mkdir build && cd build
cmake .. -D OpenCV_DIR=/opt/cpplib/opencv-4.10.0_x86_64/lib/cmake/opencv4 \
	-D ncnn_DIR=/opt/cpplib/ncnn-20240410-ubuntu-2004/lib/cmake/ncnn # 请以实际路径为准

# 单张图片推理(YOLOv8 ncnn))
./demo/ncnn_predict/ncnn_predict \
	../models/ncnn/yolov8n_ncnn_model/model.ncnn.param \
	../models/ncnn/yolov8n_ncnn_model/model.ncnn.bin \
	../media/bus.jpg

# YOLO8+SORT跟踪测试
./demo/yolov8_SORT/yolov8_sort \
	../models/ncnn/yolov8n_ncnn_model_fp16/model.ncnn.param \
	../models/ncnn/yolov8n_ncnn_model_fp16/model.ncnn.bin \
	../media/testvideo.mp4 

# yunet+SORT人脸跟踪测试
./demo/yunet_SORT/yunet_SORT \
	../models/ncnn/yunet_n_320_320_ncnn_model/model.param \
	../models/ncnn/yunet_n_320_320_ncnn_model/model.bin \
	../media/cxk.mp4 

```
