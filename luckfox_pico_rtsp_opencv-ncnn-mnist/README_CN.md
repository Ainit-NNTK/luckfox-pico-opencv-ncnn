**更改CmakeList.txt中**

```
set(CMAKE_C_COMPILER "/path/to/luckfox-pico/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc")
set(CMAKE_CXX_COMPILER "/path/to//luckfox-pico/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-g++")
```

**编译**

```
mkdir build
cd build
cmake ..
make && make install
```

**生成可执行文件在luckfox_pico_rtsp_opencv-ncnn-mnist文件夹中**

```
./luckfox_pico_rtsp_opencv-ncnn-mnist/luckfox_pico_rtsp_opencv-ncnn-mnist
```

