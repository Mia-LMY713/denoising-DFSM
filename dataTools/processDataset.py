import glob
import cv2
from pathlib import Path
import ntpath
import time
import argparse
import sys
from PIL import Image
import numpy as np
import os
from utilities.customUtils import *
from utilities.aestheticUtils import *
from etaprogress.progress import *
from dataTools.sampler import *


class datasetSampler:

    def __init__(self, config, source, target, gridSze, numberOfSamples=None, linRGBPath=None):

        # 配置基础路径
        self.gtPath = formatDirPath(source)
        self.targetPath = formatDirPath(target)
        self.numberOfSamples = numberOfSamples
        print("number of data samples to be processed", self.numberOfSamples)
        self.interval = int(config['interval'])
        self.barLen = int(config['barLen'])
        self.gridSze = int(gridSze)
        self.patchSize = 128

        # 整合 Linear RGB 路径逻辑
        self.linRGBPath = formatDirPath(linRGBPath) if linRGBPath else None
        
        # 创建用于保存处理后样本的目录
        if self.linRGBPath:
            # 如果是 Linear RGB 模式，成对创建 GT 目标目录与采样目标目录
            self.gtTargetPath = self.targetPath + "gtPatch/"
            self.sampledTargetPath = self.targetPath + "sampledPatch/"
            createDir(self.gtTargetPath)
            createDir(self.sampledTargetPath)
        else:
            # 普通单图处理模式
            createDir(self.targetPath)
        
        # 列出源目录中的所有图像
        self.sourceDataSamples = imageList(formatDirPath(self.gtPath))

    def startResumeProcess(self):
        startTime = time.time()
        printProgressBar(0, len(self.samplesInTargetDirectory), prefix='Loading process', suffix='completed', length=self.barLen)

        for s, i in enumerate(self.samplesInTargetDirectory):
            targetFile = self.gtPath + extractFileName(i)
            try:
                self.sourceDataSamples.remove(targetFile)
            except:
                pass
            printProgressBar(s, len(self.samplesInTargetDirectory), prefix='Loading process', suffix=' completed', length=self.barLen)
            sys.stdout.flush()
        hours, minutes, seconds = timer(startTime, time.time())     
        print("\nTime elapsed to resume process [{:0>2}:{:0>2}:{:0>2}]".format(hours, minutes, seconds))

        return self.sourceDataSamples

    def samplingImages(self):
        if not self.numberOfSamples:
            self.numberOfSamples = len(self.sourceDataSamples)
        i = 0
        timerFlag = 0 
        startTime = time.time()
        bar = ProgressBar(self.numberOfSamples, max_width=self.barLen)

        for sample in self.sourceDataSamples:
            if timerFlag == 0:
                loopTime = time.time()
                timerFlag = 1

            try:
                filename = extractFileName(sample)
                patchSize = (self.patchSize, self.patchSize) 
                
                # 1. 读取并缩放 Ground Truth 图像
                image = Image.open(sample)
                image = np.asarray(image.resize(patchSize))

                # 2. 根据模式执行不同的采样与保存逻辑
                if self.linRGBPath:
                    # 【Linear RGB 模式】
                    # 获取对应的 Linear RGB 图像路径
                    linSamplePath = self.linRGBPath + filename
                    linImage = Image.open(linSamplePath)
                    linImage = np.asarray(linImage.resize(patchSize))

                    # 对 Linear RGB 图像进行结构化下采样
                    if self.gridSze == 1:
                        sampledImg = bayerSampler(linImage)
                    elif self.gridSze == 2:
                        sampledImg = quadBayerSampler(linImage)
                    else:
                        sampledImg = dynamicBayerSampler(linImage, self.gridSze)
                    
                    # 保存成对的图像
                    gtImageOut = Image.fromarray(image)
                    gtImageOut.save(self.gtTargetPath + filename)

                    sampledImageOut = Image.fromarray(sampledImg)
                    sampledImageOut.save(self.sampledTargetPath + filename)
                else:
                    # 【标准单图模式】：直接对原图（GT）进行采样
                    if self.gridSze == 1:
                        image = bayerSampler(image)
                    elif self.gridSze == 2:
                        image = quadBayerSampler(image)
                    else:
                        image = dynamicBayerSampler(image, self.gridSze)
                    
                    image = Image.fromarray(image)
                    image.save(self.targetPath + filename)
                
                i += 1

            except Exception as e:
                # 发生异常时打印错误，并避免中断主循环
                print(f"\nError processing sample {sample}: {e}")
                if os.path.exists(sample):
                    os.remove(sample)

            if i % 1000 == 0:
                bar.numerator = i
                print("Image Sampled:", bar, end='\r')
            sys.stdout.flush()

            if i == self.numberOfSamples:
                print("Successfully sampled target {} of images!".format(self.numberOfSamples))
                break

        hours, minutes, seconds = timer(startTime, time.time())     
        print("Processed [{}] images! Total time elapsed [{:0>2}:{:0>2}:{:0>2}].".format(i, hours, minutes, seconds))

    def resumeSampling(self, numberOfSamples=None):
        print("Resuming Process....")
        
        # 根据当前模式检测对应的输出目标文件夹
        checkPath = self.sampledTargetPath if self.linRGBPath else self.targetPath
        self.samplesInTargetDirectory = imageList(formatDirPath(checkPath))
        
        if self.samplesInTargetDirectory:
            print("[{}] image samples have been found in the target directory!".format(len(self.samplesInTargetDirectory)))
            if int(len(self.samplesInTargetDirectory)) >= int(len(self.sourceDataSamples)):
                print("All target images are already been processed! Thus, the process did not resume!")   
            else:
                if self.numberOfSamples:
                    if self.numberOfSamples - int(len(self.samplesInTargetDirectory)) > 0 and self.numberOfSamples < int(len(self.sourceDataSamples)):
                        self.numberOfSamples = self.numberOfSamples - int(len(self.samplesInTargetDirectory))
                    else:
                        print("Invalid amount of target samples have been given!")
                        sys.exit()

                # 排除已处理的文件
                self.sourceDataSamples = self.startResumeProcess()
                # 开始采样
                self.samplingImages()
        else:
            print("Target directory is empty! Unable to resume process. Process is starting from the beginning...")
            self.samplingImages()