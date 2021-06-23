#!/usr/bin/env python
"""
 Copyright (C) 2018-2019 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
from time import time
from openvino.inference_engine import IENetwork, IECore
from skimage import morphology
import math


classes_color_map = [
    (150, 150, 150),
    (58, 55, 169),
    (211, 51, 17),
    (157, 80, 44),
    (23, 95, 189),
    (210, 133, 34),
    (76, 226, 202),
    (101, 138, 127),
    (223, 91, 182),
    (80, 128, 113),
    (235, 155, 55),
    (44, 151, 243),
    (159, 80, 170),
    (239, 208, 44),
    (128, 50, 51),
    (82, 141, 193),
    (9, 107, 10),
    (223, 90, 142),
    (50, 248, 83),
    (178, 101, 130),
    (71, 30, 204)
]

def road_segmentation(classes_map):
    img_road=np.array(classes_map)
    b = np.array(img_road[:, :, 0])
    g = np.array(img_road[:, :, 1])
    r = np.array(img_road[:, :, 2])
    #道路部分
    broad=np.where(r-g>0,255,0)
    broad=np.where(r-b<=0,0,broad)
    b=broad
    g=b
    r=b

    img_road[:,:,0]=b
    img_road[:,:,1]=g
    img_road[:,:,2]=r
    return img_road

def lane_detection(classes_map):
    img_lane=np.array(classes_map)
    b = np.array(img_lane[:, :, 0])
    g = np.array(img_lane[:, :, 1])
    r = np.array(img_lane[:, :, 2])
    #车道线部分
    blane=np.where(b-g>0,255,0)
    blane=np.where(b-r<=0,0,blane)
    b=blane
    g=b
    r=b

    img_lane[:,:,0]=b
    img_lane[:,:,1]=g
    img_lane[:,:,2]=r
    return img_lane

def road_lane(classes_map):
    img_road_lane=np.array(classes_map)
    b = np.array(img_road_lane[:, :, 0])
    g = np.array(img_road_lane[:, :, 1])
    r = np.array(img_road_lane[:, :, 2])
    #车道线部分
    blane=np.where(b-g>0,255,0)
    blane=np.where(b-r<=0,0,blane)
    #道路部分
    broad=np.where(r-g>0,255,0)
    broad=np.where(r-b<=0,0,broad)
    #道路+车道线整体
    b=blane+broad
    g=b
    r=b

    img_road_lane[:,:,0]=b
    img_road_lane[:,:,1]=g
    img_road_lane[:,:,2]=r
    return img_road_lane

#指定输入的函数
def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is "
                           "acceptable. Sample will look for a suitable plugin for device specified. Default value is CPU",
                      default="CPU", type=str)
    args.add_argument("-nt", "--number_top", help="Optional. Number of top results", default=10, type=int)
    return parser

def chuli(classes_map):
    classes_map_copy = np.array(classes_map,dtype=np.uint8)

    img=np.array(classes_map,dtype=np.uint8)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    gray = img
    ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)  
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
    area = []
    # 找到最大的轮廓
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    max_idx = np.argmax(np.array(area))
    cv2.drawContours(classes_map_copy,contours[max_idx],-1,(0,0,255),3)#画出最大轮廓
    #获取轮廓凸包点，并计算轮廓最下面区域的中点，作为第二个方向点
    hull=cv2.convexHull(contours[max_idx])
    #计算质心
    M=cv2.moments(contours[max_idx])
    if(M['m00']==0):
        cx=max(hull[:,:,0])
        cy=max(hull[:,:,1])
    else:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
    
    sumx=max(hull[:,:,0])+min(hull[:,:,0])
    x=int(sumx/2)
    cv2.arrowedLine(classes_map_copy,(int(x),511),(cx,cy),(0,0,255),thickness=3)#绘制箭头(画在原始图像上，img->img_gray)
    return classes_map_copy#输出原始图像：img->img_gray



def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    model_xml = "../model/road-segmentation-adas-0001.xml"
    model_bin = "../model/road-segmentation-adas-0001.bin"

    log.info("Creating Inference Engine")
    #创建推理机
    ie = IECore()
    # Read IR
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)

    assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"

    log.info("Preparing input blobs")
    #获取输入输出层的名称(next是获取名称，iter是获取输入输出层)
    #net.inputs:{'data': <openvino.inference_engine.ie_api.InputInfo object at 0x000001C9845A1EB8>}
    input_blob = next(iter(net.inputs))#"data"
    out_blob = next(iter(net.outputs))#"L0317_ReWeight_SoftMax"

    net.batch_size = 1
    # Loading model to the plugin
    #def load_network (self, IENetwork, network, str, device_name, config=None, int, num_requests=1)
    #调用 IECore 的 load_network 方法创建一个 ExecutableNetwork， 这个 ExecutableNetwork 将负责执行推理
    exec_net = ie.load_network(network=net, device_name=args.device)

    img_dirs = ["../data/images"]

    seg_dirs=["../results/seg"]
    road_dirs = ["../results/road"]
    lane_dirs=["../results/lane"]
    road_lane_dirs=["../results/road_lane"]
    line_dirs=["../results/direction"]
    img_num = len(img_dirs)
    
    sum_time=0
    for i in range (img_num):
        image_dir = img_dirs[i]
        seg_dir=seg_dirs[i]
        road_dir=road_dirs[i]
        lane_dir=lane_dirs[i]
        road_lane_dir=road_lane_dirs[i]
        line_dir=line_dirs[i]
        images=os.listdir(image_dir)
        
        for label in images:
            print(label)
            image_name=image_dir+"/"+label
            image=cv2.imread(image_name)

            seg_name=seg_dir+"/"+label
            road_name=road_dir+"/"+label
            lane_name=lane_dir+"/"+label
            road_lane_name=road_lane_dir+"/"+label
            line_name=line_dir+"/"+label

            # Read and pre-process input images
            #net.inputs：{'data': <openvino.inference_engine.ie_api.InputInfo object at 0x000001C9845A1EB8>}
            n, c, h, w = net.inputs[input_blob].shape
            #输入图像，执行推断
            #将图片转换为固定大小
            if image.shape[:-1] != (h, w):
                log.warning("Image is resized from {} to {}".format(image.shape[:-1], (h, w)))
                image = cv2.resize(image, (w, h))

            image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            # c, h, w -> 1, c, h, w
            image = image[np.newaxis]


            inf_start=time()#计算一帧处理时间
            # Start sync inference
            log.info("Starting inference")
            #res是推理结果,包含索引号out_blob和四维矩阵，只有一个索引out_blob
            #def infer (self, inputs=None)
            #调用 ExecutableNetwork 类的 infer 函数来执行推理工作
            res = exec_net.infer(inputs={input_blob: image})#指定输入的连通域(图像)
            #inputs={input_blob: image}为由输入名访问的输入连通域
            # Processing output blob(均要输出4D))
            log.info("Processing output blob")
            res = res[out_blob]#网络输出，与out_blob对应的推理结果,相当于去除索引号out_blob,直接得到四维矩阵
            if len(res.shape) == 3:
                res = np.expand_dims(res, axis=1)
            if len(res.shape) == 4:
                _, _, out_h, out_w = res.shape
            else:
                raise Exception("Unexpected output blob shape {}. Only 4D and 3D output blobs are supported".format(res.shape))
            #利用numpy库提高速度
            classes_color_map_np = np.array(classes_color_map)
            for batch, data in enumerate(res):
                #初始分割结果
                classes_map = np.zeros(shape=(out_h, out_w, 3), dtype=np.int)
                index = np.argmax(data, axis=0)
                classes_map = classes_color_map_np[np.where(index<20, index, np.full(index.shape,20))]
                cv2.imwrite(seg_name, classes_map)

                #道路分割
                img_road=road_segmentation(classes_map)
                cv2.imwrite(road_name, img_road)
                #车道线检测
                img_lane=lane_detection(classes_map)
                cv2.imwrite(lane_name, img_lane)
                #道路+车道线
                img_road_lane=road_lane(classes_map)
                cv2.imwrite(road_lane_name, img_road_lane)
                #方向图
                img_line=chuli(img_road_lane)
                cv2.imwrite(line_name, img_line)
                #所用时间
                inf_end = time()
                det_time = inf_end - inf_start
                print(det_time)

                sum_time=sum_time+det_time
        avgtime=sum_time/len(images)
        print(avgtime)
        print("finished predicting ", image_dir)
if __name__ == '__main__':
    sys.exit(main() or 0)
