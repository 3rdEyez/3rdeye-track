###################
# Download models #
###################


if [ ! -d ../models/ncnn ]; then
    mkdir -p ../models/ncnn
fi

if [ ! -d ../models/ncnn/yolov8n-face_ncnn_model ]; then
    cd ../models/ncnn
    wget https://cdn.fyrabh.top/repo_resurces/3rdeye_track/model/yolov8n-face_ncnn_model.tar.xz
    tar -xf yolov8n-face_ncnn_model.tar.xz
    rm yolov8n-face_ncnn_model.tar.xz
fi

if [ ! -d ../models/ncnn/yolov8n_ncnn_model ]; then
    cd ../models/ncnn
    wget https://cdn.fyrabh.top/repo_resurces/3rdeye_track/model/yolov8n_ncnn_model.tar.xz
    tar -xf yolov8n_ncnn_model.tar.xz
    rm yolov8n_ncnn_model.tar.xz
fi