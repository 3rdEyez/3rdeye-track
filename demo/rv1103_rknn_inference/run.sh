#!/bin/sh

BUILD_DIR="build"

BOARD_IP=$1
if [ -z "$BOARD_IP" ]; then
    echo "Usage: $0 <board_ip>"
    exit 1
fi
echo "board ip: $BOARD_IP"

if [ ! -d $BUILD_DIR ]; then
    mkdir $BUILD_DIR
fi

cd $BUILD_DIR
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_FLAGS="-D PROFILER_ON" -D CMAKE_INSTALL_PREFIX=install ..
make install -j8
rm install/lib -r
rm install/inc -r
sshpass -p luckfox scp -r install root@$BOARD_IP:~
sshpass -p luckfox ssh root@$BOARD_IP "cd /root/install; ./rv1103_rknn_inference_demo"
sshpass -p luckfox scp root@$BOARD_IP:/root/install/result.jpg .

