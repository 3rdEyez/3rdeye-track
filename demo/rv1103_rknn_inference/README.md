在使用本用例之前请先安装好arm-rockchip830-linux-uclibcgnueabihf-gcc并添加到PATH中，确保能在终端中：
```shell
$ arm-rockchip830-linux-uclibcgnueabihf-gcc -v
Using built-in specs.
COLLECT_GCC=arm-rockchip830-linux-uclibcgnueabihf-gcc
此处省略...
Thread model: posix
gcc version 8.3.0 (crosstool-NG 1.24.0) 
```
然后就可以开始编译了，因为cmake中添加了一些自动下载库的代码，所以此工程编译起来比较简单：
```shell
mkdir build
cd build
cmake ..
make
```
编译完成后，会在build目录下生成可执行文件,`rv1103_rknn_inference`，但是这里我们不直接运行，这里提供一个run.sh脚本，方便我们运行测试：
```shell
ssh root@192.168.2.107 # 该命令只执行一次，防止卡在Yes/No界面，连接上去后退出即可
./run.sh 192.168.2.107
```
执行完之后，build目录下会出现result.jpg，这就是推理结果。