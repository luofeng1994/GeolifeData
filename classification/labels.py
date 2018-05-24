# -*- coding: utf-8 -*-
from __future__ import division
import time
import os
from math import radians, cos, sin, asin, sqrt
import utm
import shutil
import random
import math
from featureExtract import extract
from featureExtract import csvWrite
from itertools import chain

class Point():
    def __init__(self, lat=0, lon=0, timestamp=0, velocity=0):
        self.lat = lat
        self.lon = lon
        self.timestamp = timestamp
        self.velocity = velocity
    def setVelocity(self, velocity):
        self.velocity = velocity

def getDate():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()).split(' ')[0]
def getTime():
    return time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())

def isVaildDate(date):
    try:
        if ":" in date:
            time.strptime(date, "%Y/%m/%d %H:%M:%S")
        else:
            time.strptime(date, "%Y/%m/%d")
        return True
    except:
        return False
def getTimeStamp(timeFormat):
    return time.mktime(time.strptime(timeFormat, "%Y/%m/%d %H:%M:%S"))

def getVelocityVector(lon1, lat1, lon2, lat2):
    y1, x1, _, _ = utm.from_latlon(float(lat1), float(lon1))
    y2, x2, _, _ = utm.from_latlon(float(lat2), float(lon2))
    return [x2-x1, y2-y1]

def getPolarB(x, y):
    r = math.sqrt(x * x + y * y)
    if x == 0:
        if y >= 0:
            return math.pi / 2
        else:
            return 3 * math.pi / 2
    if y == 0:
        if x >= 0:
            return 0
        else:
            return math.pi
    b = math.asin(abs(x)/r)
    if x < 0 and y > 0:
        return b
    elif x > 0 and y > 0:
        return 2 * math.pi - b
    elif x < 0 and y < 0:
        return math.pi - b
    elif x > 0 and y < 0:
        return math.pi + b
    else:
        return 0


def addVelocityVector(list):
    listAddVelocityVector=[]
    for n in range(1,len(list)):
        lat_, lon_, second_ = list[n-1][0], list[n-1][1], list[n-1][2]
        lat, lon, second = list[n][0], list[n][1], list[n][2]
        distance = haversine(float(lon_), float(lat_), float(lon), float(lat))
        velocity = distance/(float(second)-float(second_))
        velocityVector = getVelocityVector(lon_,lat_,lon,lat)
        velocityVector = [c/(float(second)-float(second_)) for c in velocityVector]
        list[n].insert(5,velocityVector[0])
        list[n].insert(6,velocityVector[1])
        list[n].insert(7,velocity)
        listAddVelocityVector.append(list[n])
    return listAddVelocityVector

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
def writeList1(filePath, listToWrite):
    # print listToWrite
    dir = os.path.split(filePath)[0]
    if not os.path.isdir(dir):
        os.makedirs(dir)
    f = open(filePath, 'w')
    for i in listToWrite:
        f.write(i)
    f.close()
def writeList(filePath, listToWrite):
    # print listToWrite
    dir = os.path.split(filePath)[0]
    if not os.path.isdir(dir):
        os.makedirs(dir)
    if os.path.isfile(filePath):
        os.remove(filePath)
    f = open(filePath, 'w')
    for i in listToWrite:
        if isinstance(i, list):
            k = ','.join([str(j) for j in i])
            f.write(k + "\n")
        elif isinstance(i, basestring):
            f.write(i + "\n")
    f.close()


def makeTrainData():
    # rootdir = './trajectory_data'
    # save_dir = './Data_Geolife'
    rootdir = u'/media/luofeng/新加卷1/实验室/项目/轨迹预测与分类/gps data/measured'
    save_dir= './Data_measured'
    train_dir = 'all_cl'
    test_dir = 'all_init'
    dirs = os.listdir(rootdir)
    for dir in dirs:
        dir_ = os.path.join(rootdir, dir)
        if os.path.isdir(dir_) and (dir == train_dir or dir == test_dir):
            feature_data = []
            type_dirs = os.listdir(dir_)
            for type_dir in type_dirs:
                type_dir_ = os.path.join(dir_, type_dir)
                if os.path.isdir(type_dir_):
                    type = type_dir
                    files = os.listdir(type_dir_)
                    for file in files:
                        file_name = file.split('.')[0]
                        file = os.path.join(type_dir_, file)
                        print file
                        point_list = []
                        velocity_list = []
                        velocity_polar_B_list = []
                        velocity_x_list = []
                        velocity_y_list = []
                        acceleration_list = []
                        acceleration_polar_B_list = []
                        acceleration_x_list = []
                        acceleration_y_list = []
                        with open(file, 'r') as f:
                            for line in f:
                                line = line.strip('\n').strip()
                                if not len(line.split(',')) == 7:
                                    continue
                                lat, lon, aaa, bbb, numberOfDays, date, time = line.split(',')
                                if not len(date)==0 and not len(time)==0:
                                    timeStamp = getTimeStamp(date + ' ' + time)
                                else:
                                    timeStamp = float(numberOfDays) * 24 * 60 * 60
                                point_list.append(Point(float(lat), float(lon), timeStamp))
                        for i in range(1, len(point_list)):
                            point_1 = point_list[i-1]
                            point_2 = point_list[i]
                            distance = haversine(point_1.lon, point_1.lat, point_2.lon, point_2.lat)
                            velocity = distance/(point_2.timestamp - point_1.timestamp + 1)
                            velocityVector = getVelocityVector(point_1.lon, point_1.lat, point_2.lon, point_2.lat)
                            velocityVector = [c / (point_2.timestamp - point_1.timestamp + 1) for c in velocityVector]
                            velocity_polar_B = getPolarB(velocityVector[0], velocityVector[1])
                            point_list[i].setVelocity(velocity)
                            velocity_list.append(velocity)
                            velocity_polar_B_list.append(velocity_polar_B)
                            velocity_x_list.append(velocityVector[0])
                            velocity_y_list.append(velocityVector[1])
                        for i in range(1, len(velocity_list)):
                            velocity_1 = velocity_list[i-1]
                            velocity_2 = velocity_list[i]
                            velocity_x_1 = velocity_x_list[i-1]
                            velocity_x_2 = velocity_x_list[i]
                            velocity_y_1 = velocity_y_list[i-1]
                            velocity_y_2 = velocity_y_list[i]
                            acceleration = (velocity_2-velocity_1)/(point_list[i+1].timestamp - point_list[i].timestamp + 1)
                            acceleration_x = (velocity_x_2-velocity_x_1)/(point_list[i+1].timestamp - point_list[i].timestamp + 1)
                            acceleration_y = (velocity_y_2-velocity_y_1)/(point_list[i+1].timestamp - point_list[i].timestamp + 1)
                            acceleration_polar_B = getPolarB(acceleration_x, acceleration_y)
                            acceleration_list.append(acceleration)
                            acceleration_x_list.append(acceleration_x)
                            acceleration_y_list.append(acceleration_y)
                            acceleration_polar_B_list.append(acceleration_polar_B)
                        out = []
                        for i in range(len(acceleration_list)):
                            out.append([velocity_list[i+1], velocity_polar_B_list[i+1], velocity_x_list[i+1], velocity_y_list[i+1], acceleration_list[i], acceleration_polar_B_list[i], acceleration_x_list[i], acceleration_y_list[i]])
                        save_path = os.path.join(save_dir, dir, type, file_name+'.txt')
                        writeList(save_path, out)
                        print 'extracting machine learning features...'
                        point_list.pop(0)
                        try:
                            features = extract(point_list)
                            features.append(type)
                            features = tuple(features)
                            feature_data.append(features)
                        except:
                            continue
            csvWrite(os.path.join(save_dir, dir, '{}.csv'.format(dir)),
                     ['distance', 'averageVelocity', 'expectationVelocity', 'velocityVariance', 'maxVelocity1',
                      'maxVelocity2', 'minVelocity3', 'minVelocity1', 'minVelocity2', 'minVelocity3', 'label'], feature_data)




def separateTrainAndTestTrajectory():
    rootdir = './2018-05-10 12-09-39'
    train_dir = './trajectory_data/train_tmp/'
    test_dir = './trajectory_data/test_tmp/'
    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)
    if not os.path.isdir(test_dir):
        os.mkdir(test_dir)
    files = os.listdir(rootdir)
    random.shuffle(files)
    random.shuffle(files)
    index = int(0.8*(len(files)))
    train_files = files[:index]
    test_files = files[index:len(files)]
    for file in train_files:
        label = file.split('-')[0]
        file = os.path.join(rootdir, file)
        dst_dir = os.path.join(train_dir, label)
        if not os.path.isdir(dst_dir):
            os.mkdir(dst_dir)
        shutil.move(file, dst_dir)
    for file in test_files:
        label = file.split('-')[0]
        file = os.path.join(rootdir, file)
        dst_dir = os.path.join(test_dir, label)
        if not os.path.isdir(dst_dir):
            os.mkdir(dst_dir)
        shutil.move(file, dst_dir)

def cutByWindow():
    rootdir = './trajectory_data/test_tmp/'
    dstdir = './trajectory_data/test/'
    type_num = dict()
    for label in os.listdir(rootdir):
        label_dir = os.path.join(rootdir, label)
        for file in os.listdir(label_dir):
            file = os.path.join(label_dir, file)
            data = []
            with open(file, 'r') as f:
                for line in f:
                    line = line.strip('\n')
                    data.append(line)
            if len(data) >= 5:
                max_length = 302
                if len(data) <= max_length:
                    if type_num.has_key(label):
                        type_num[label] = type_num[label] + 1;
                    else:
                        type_num[label] = 1
                    filePath = dstdir + label + '/' + str(type_num[label]) + '.plt'
                    writeList(filePath, data)
                else:
                    step = 50
                    for j in range(0, len(data) - max_length + step, step):
                        data_part = data[j:min(j+max_length, len(data))]
                        if type_num.has_key(label):
                            type_num[label] = type_num[label] + 1;
                        else:
                            type_num[label] = 1
                        filePath = dstdir + label + '/' + str(type_num[label]) + '.plt'
                        writeList(filePath, data_part)

class Separator:
    startTime = ''
    def __init__(self, dir):
        self.dir = dir
        self.num = 0
    def getLabelForTimestampFormDict(self, timeStamp):
        for label, timeList in self.labelDict.items():
            for range in timeList:
                if timeStamp>=range[0] and timeStamp<range[1]:
                    return label
                # print label, range
            # print label, timeList
        return 'null'

    def separateForOneLog(self, fileName):
        print ('start separate for file %s'%(fileName))
        labeledList = []
        labelForLastPoint = ''
        with open(fileName) as file:
            for i in range(6):
                line = next(file)
            raw_data = []
            for line in file:
                line = line.strip('\n').strip()
                raw_data.append(line)
            if len(raw_data) == 1:
                self.num += 1
                filePath = os.path.join(self.saveDir, str(self.num)+'.plt')
                writeList(filePath, raw_data)
            else:
                start = 0
                end = 0
                for i in range(len(raw_data)-1):
                    line1 = raw_data[i]
                    line2 = raw_data[i+1]
                    _, _, _, _, _, date, time = line1.split(',')
                    timeStamp1 = getTimeStamp(date + ' ' + time)
                    _, _, _, _, _, date, time = line2.split(',')
                    timeStamp2 = getTimeStamp(date + ' ' + time)
                    if timeStamp2-timeStamp1>60*60.0:
                        start = end
                        end = i+1
                        self.num += 1
                        filePath = os.path.join(self.saveDir, str(self.num) + '.plt')
                        writeList(filePath, raw_data[start:end])
                start = end
                end = len(raw_data)+1
                self.num += 1
                filePath = os.path.join(self.saveDir, str(self.num) + '.plt')
                writeList(filePath, raw_data[start:end])



    def separateForOneDir(self):
        trajectoryDir = os.path.join(self.dir, 'trajectory')
        self.saveDir = self.dir + '/trajectory_separated'
        if os.path.isdir(self.saveDir):
            shutil.rmtree(self.dir + '/trajectory_separated')
        for file in os.listdir(trajectoryDir):
            filePath = os.path.join(trajectoryDir, file)
            if os.path.isfile(filePath):
                self.separateForOneLog(filePath)
        # trajectoryFile = trajectoryDir + '23-29.plt'
        # self.labelForOneLog(trajectoryFile)



class Labeling:
    type_num = {}
    startTime = ''
    def __init__(self, dir):
        self.labelDict = {}
        self.dir = dir

    def getLabelForTimestampFormDict(self, timeStamp):
        for label, timeList in self.labelDict.items():
            for range in timeList:
                if timeStamp>=range[0] and timeStamp<range[1]:
                    return label
                # print label, range
            # print label, timeList
        return 'null'

    def labelForOneLog(self, fileName):
        print ('start label for file %s'%(fileName))
        labeledList = []
        labelForLastPoint = ''
        with open(fileName) as file:
            # for i in range(6):
            #     line = next(file)
            for line in file:
                line = line.strip('\n').strip()
                lat, lon, aaa, bbb, ccc, date, time = line.split(',')
                if isVaildDate(date + ' ' + time):
                    timeStamp = getTimeStamp(date + ' ' + time)
                    label = self.getLabelForTimestampFormDict(timeStamp)
                    #将taxi和car合并
                    if label == 'taxi':
                        label = 'car'
                    list_tmp = [lat, lon, aaa, bbb, ccc, date, time]
                    if label != 'null' and label == labelForLastPoint:
                        labeledList.append(list_tmp)
                    elif label != 'null' and label != labelForLastPoint:
                        if len(labeledList):
                            if Labeling.type_num.has_key(labelForLastPoint):
                                Labeling.type_num[labelForLastPoint] = Labeling.type_num[labelForLastPoint] + 1;
                            else:
                                Labeling.type_num[labelForLastPoint] = 1
                            filePath = './' + Labeling.startTime + '/' + labelForLastPoint + '-' + str(
                                Labeling.type_num[labelForLastPoint]) + '.plt'
                            writeList(filePath, labeledList)
                        # if len(labeledList) >= 5:
                        #     max_length = 300
                        #     if len(labeledList) <= max_length:
                        #         if Labeling.type_num.has_key(labelForLastPoint):
                        #             Labeling.type_num[labelForLastPoint] = Labeling.type_num[labelForLastPoint] + 1;
                        #         else:
                        #             Labeling.type_num[labelForLastPoint] = 1
                        #         filePath = './' + Labeling.startTime + '/' + labelForLastPoint + '-' + str(Labeling.type_num[labelForLastPoint]) + '.plt'
                        #         # labeledList = addVelocityVector(labeledList)
                        #         writeList(filePath, labeledList)
                        #     else:
                        #         step = 50
                        #         for j in range(0, len(labeledList) - max_length + step, step):
                        #             labeledList_part = labeledList[j:min(j+max_length, len(labeledList))]
                        #             if Labeling.type_num.has_key(labelForLastPoint):
                        #                 Labeling.type_num[labelForLastPoint] = Labeling.type_num[labelForLastPoint] + 1;
                        #             else:
                        #                 Labeling.type_num[labelForLastPoint] = 1
                        #             filePath = './' + Labeling.startTime + '/' + labelForLastPoint + '-' + str(
                        #                 Labeling.type_num[labelForLastPoint]) + '.plt'
                        #             writeList(filePath, labeledList_part)
                        labeledList = []
                        labeledList.append(list_tmp)
                        labelForLastPoint = label
    def getLabelDict(self):
        with open(self.dir + '/labels.txt') as file:
            next(file) #从第二行开始
            for line in file:
                line = line.strip('\n')
                line = line.strip()
                if len(line.split('\t'))==4:
                    date, startime, endtime, label = line.split('\t')
                    label = label.strip()
                    starttimeFormat = date + ' ' + startime
                    endtimeFormat = date + ' ' + endtime
                    if isVaildDate(starttimeFormat) and isVaildDate(endtimeFormat):
                        startStamp = getTimeStamp(starttimeFormat)
                        endStamp = getTimeStamp(endtimeFormat)
                        if self.labelDict.has_key(label):
                            self.labelDict[label].append([startStamp, endStamp])
                        else:
                            self.labelDict[label] = []
                            self.labelDict[label].append([startStamp, endStamp])
        # print self.labelDict
    def labelForOneDir(self):
        trajectoryDir = os.path.join(self.dir, 'trajectory_separated')
        for file in os.listdir(trajectoryDir):
            filePath = os.path.join(trajectoryDir, file)
            if os.path.isfile(filePath):
                self.labelForOneLog(filePath)
        # trajectoryFile = trajectoryDir + '23-29.plt'
        # self.labelForOneLog(trajectoryFile)

def test():
    dir = unicode('E:/实验室/项目/轨迹预测与分类/gps data/Geolife Trajectories with transportation mode labels/006/', 'utf8')
    Labeling.startTime = getTime()
    labeling = Labeling(dir)
    labeling.getLabelDict()
    labeling.labelForOneDir()
def makeLabel():
    rootdir = unicode('/media/luofeng/新加卷1/实验室/项目/轨迹预测与分类/gps data/Geolife Trajectories with transportation mode labels', 'utf8')
    Labeling.startTime = getTime()
    files = os.listdir(rootdir)
    for file in files:
        m = os.path.join(rootdir, file)
        print m
        labeling = Labeling(m)
        labeling.getLabelDict()
        labeling.labelForOneDir()

def separateByTime():
    rootdir = unicode('E:/实验室/项目/轨迹预测与分类/gps data/Geolife Trajectories with transportation mode labels/', 'utf8')
    files = os.listdir(rootdir)
    for file in files:
        m = os.path.join(rootdir, file)
        print m
        separator = Separator(m)
        separator.separateForOneDir()

if __name__ == '__main__':
    # separateByTime()
    # makeLabel()
    # separateTrainAndTestTrajectory()
    # cutByWindow()
    makeTrainData()
