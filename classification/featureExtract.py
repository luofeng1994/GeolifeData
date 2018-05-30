# -*- coding: utf-8 -*-
import time
import os
from math import radians, cos, sin, asin, sqrt
import utm
import numpy as np
import csv

def csvWrite(file_path, title, data):
    # with open(file_path, 'wb') as csvfile:
    csvfile = file(file_path, 'wb')
    writer = csv.writer(csvfile)
    writer.writerow(title)
    writer.writerows(data)
    csvfile.close()

def haversine(lon1, lat1, lon2, lat2):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """ 
    Calculate the great circle distance between two points  
    on the earth (specified in decimal degrees) 
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r * 1000

def extract(trajetory):
    def add(x,y):
        return x+y

    trajetory1 = trajetory[:-1]
    trajetory2 = trajetory[1:]
    trajetory3 = [[trajetory1[i], trajetory2[i]] for i in range(len(trajetory1))]
    distanceList = [haversine(item[0].lon, item[0].lat, item[1].lon, item[1].lat) for item in trajetory3]
    accelerationList = [(item[1].velocity-item[0].velocity)/(item[1].timestamp-item[0].timestamp + 1) for item in trajetory3]
    timeList = [item[1].timestamp - item[0].timestamp for item in trajetory3]

    distance = sum(distanceList)
    timeRange = sum(timeList)
    avelocity = distance / (timeRange + 1)
    evelocity = np.array([item.velocity for item in trajetory]).mean()
    dvelocity = np.array([item.velocity for item in trajetory]).var()
    velocitySort = set([item.velocity for item in trajetory])
    velocitySort = list(velocitySort)
    velocitySort.sort()
    maxv3, maxv2, maxv1 = velocitySort[-3:]
    minv1, minv2, minv3 = velocitySort[:3]
    return [distance, avelocity, evelocity, dvelocity, maxv1, maxv2, maxv3, minv1, minv2, minv3]

def featureExtract(rootdir):
    files = os.listdir(rootdir)
    data = []
    for file in files:
        file = os.path.join(rootdir, file)
        if os.path.isfile(file):
            trajetory = []
            lable = file.split('/')[-1].split('-')[0]
            # print lable
            with open(file) as f:
                for line in f:
                    line = line.strip()
                    lat, lon, time, _, _, x, y, velocity, _ = line.split(' ')
                    point = Point(float(lat), float(lon), float(time), float(velocity))
                    trajetory.append(point)
            if len(trajetory) < 10:
                continue
            try:
                features = extract(trajetory)
                features.append(lable)
                features = tuple(features)
                data.append(features)
            except:
                print file
                continue

    csvWrite('./2017-09-27 18-05-32 splitByNumSeqs featureData/test.csv', ['distance', 'averageVelocity', 'expectationVelocity', 'velocityVariance', 'maxVelocity1', 'maxVelocity2', 'minVelocity3', 'minVelocity1', 'minVelocity2', 'minVelocity3', 'lable'],data)
