#!/bin/sh

BUILD_DIR="build"

if [ ! -d $BUILD_DIR ]; then
    mkdir $BUILD_DIR
fi

cd $BUILD_DIR
cmake -D CMAKE_INSTALL_PREFIX=install ..
make install
rm install/lib -r
rm install/inc -r
sshpass -p luckfox scp -r install root@192.168.1.11:~
sshpass -p luckfox ssh root@192.168.1.11 "cd /root/install; ./rv1103_rknn_inference_demo"
sshpass -p luckfox scp root@192.168.1.11:/root/install/result.jpg .

