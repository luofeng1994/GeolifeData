# -*- coding: utf-8 -*-
import time
import os
from math import radians, cos, sin, asin, sqrt
import utm
from itertools import chain
import utm
import random
import math
import cPickle as pickle
import sys
from baiduSearch import search
reload(sys)
sys.setdefaultencoding('utf8')

def remove(file_path):
    if os.path.isfile(file_path):
        os.remove(file_path)
def concat(x, y):
    return str(x) + ' ' + str(y)

def save(list_to_save, save_path):
    file_dir = os.path.split(save_path)[0]
    if not os.path.isdir(file_dir):
        os.mkdir(file_dir)
    with open(save_path, 'a') as f:
        for item in list_to_save:
            if isinstance(item, Point):
                f.write(item.string+ "\n")
            elif isinstance(item, StayPoint):
                output = '{} {} {} {} {}'.format(item.lat, item.lon, item.arriveTimestamp, item.leaveTimestamp, item.timeInterval)
                f.write(output + "\n")
            elif isinstance(item, basestring):
                f.write(item + "\n")
            elif isinstance(item, list):
                output = reduce(concat, item)
                f.write(output + '\n')
            else:
                pass

def uniq(list):
    k = -1
    list_out = []
    for i in list:
        if i != k:
            list_out.append(i)
            k = i
    return list_out

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

def getTimestamp(date):
    # 转换成时间数组
    timeArray = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    # 转换成时间戳
    timestamp = time.mktime(timeArray)
    return timestamp

class StayPoint():
    def __init__(self, lat, lon, arriveTimestamp, leaveTimestamp):
        self.lat = lat
        self.lon = lon
        self.arriveTimestamp = arriveTimestamp
        self.leaveTimestamp = leaveTimestamp
        self.timeInterval = leaveTimestamp - arriveTimestamp
class Point:
    def __init__(self, lat, lon, timestamp, string):
        self.lat = lat
        self.lon = lon
        self.timestamp = timestamp
        self.string = string

def computeMeanCoord(pointsList):
    lats = [point.lat for point in pointsList]
    lons = [point.lon for point in pointsList]
    lat = sum(lats)/len(pointsList)
    lon = sum(lons)/len(pointsList)
    return lat, lon
def getPointsList(file_path):
    prefix = []
    pointsList = []
    with open(file_path, 'r') as file:
        for _ in range(6):
            prefix.append(next(file).strip())
        for line in file:
            line = line.strip()
            lat, lon, _, _, _, date, hms = line.split(',')
            timestamp = getTimestamp('{} {}'.format(date, hms))
            point = Point(float(lat.strip()), float(lon.strip()), timestamp, line)
            pointsList.append(point)
    return prefix, pointsList

def filter(file_path, save_path):
    print '***********filter file: {}*************'.format(file_path)
    prefix, pointsList = getPointsList(file_path)
    threadhold = 30  #83.33m/s = 300km/h
    length = 0
    index = 0
    #实现论文中的异常值过滤算法
    while(True):
        if index < len(pointsList)-1:
            # print index, len(pointsList)
            point = pointsList[index]
            while(True):
                if index + 1 < len(pointsList):
                    point_next = pointsList[index + 1]
                    if not point_next.timestamp == point.timestamp:
                        velocity = haversine(point.lon, point.lat, point_next.lon, point_next.lat) / (
                                            point_next.timestamp - point.timestamp)
                    else:
                        velocity = haversine(point.lon, point.lat, point_next.lon, point_next.lat) / 5.0
                    # print velocity
                    if velocity > threadhold:
                        pointsList.pop(index + 1)
                        # print 'remove index {}, velocity is {}, in file: '.format(index + 1, velocity), file_path
                    else:
                        break
                else:
                    break
            index += 1
        else:
            break
    save(prefix, save_path)
    save(pointsList, save_path)

class StayPointExtracterRS():
    def __init__(self, save_path):
        self.stayPoints = set()
        self.delta = 100 #100m
        self.threScore = 0.4 #coah
        self.minPts = 5
        self.save_path = save_path
    def extract(self, file_path, stayPoint_path):
        print '***********filter file: {}*************'.format(file_path)
        self.save_path_per_file = stayPoint_path
        stayPointList = []
        _, self.pointsList = getPointsList(file_path)
        self.N = len(self.pointsList)
        self.initialize()
        I = [i for i in range(self.N)]
        while len(I) > 0:
            i = I.pop(0)
            # print 'i:{}   k:{}'.format(i, self.k)
            if self.m[i] == 0:
                T = list(self.Neps[i])
                if len(T) < self.minPts:
                    self.m[i] = -1
                else:
                    self.m[i] = self.k
                    while len(T) > 0:
                        j = T.pop(0)
                        if self.m[j] == 0 or self.m[j] == -1:
                            self.m[j] = self.k
                            if len(self.Neps[j]) > self.minPts:
                                T = list(set(T) | self.Neps[j])
                    self.k += 1
        self.reverse()
        self.m_reverse_coordinate = {}
        for k in self.m_reverse.keys():
            v = self.m_reverse[k]
            indexList = list(v)
            tmpPointList = [self.pointsList[i] for i in indexList]
            lat, lon = computeMeanCoord(tmpPointList)
            self.m_reverse_coordinate[k] = [lat, lon]
        self.makeStayPointTrajectory()
        aaa = 1

    def makeStayPointTrajectory(self):
        stayPointTrajectory = []
        tmpList = self.m.values()
        k = 0
        stayPointIndexTmp = []
        cc = 0
        rangeListTmp = []
        start = 0
        end = 0
        for i in tmpList:
            if i != k:
                end = cc - 1
                if cc == 0:
                    pass
                else:
                    rangeListTmp.append([start, end])
                start = cc
                stayPointIndexTmp.append(i)
                k = i
            if cc == len(tmpList) - 1:
                end = cc
                rangeListTmp.append([start, end])
            cc += 1
        stayPointIndex = []
        rangeList = []
        for i in range(len(stayPointIndexTmp)):
            if stayPointIndexTmp[i] != -1:
                stayPointIndex.append(stayPointIndexTmp[i])
                rangeList.append(rangeListTmp[i])

        for i in range(len(stayPointIndex)):
            lat = self.m_reverse_coordinate[stayPointIndex[i]][0]
            lon = self.m_reverse_coordinate[stayPointIndex[i]][1]
            arriveTime = self.pointsList[rangeList[i][0]].timestamp
            leaveTime = self.pointsList[rangeList[i][-1]].timestamp
            stayPoint = StayPoint(lat, lon, arriveTime, leaveTime)
            stayPointTrajectory.append(stayPoint)
        if len(stayPointTrajectory) > 0:
            save(stayPointTrajectory, self.save_path_per_file)
        self.stayPoints |= set(stayPointTrajectory)

    def initialize(self):
        self.k = 1
        self.m = {i:0 for i in range(self.N)}
        self.Neps = {i:set([i]) for i in range(self.N)}
        self.initlizeNeps()
    def initlizeNeps(self):
        for i in range(self.N - 1):
            # print i
            for j in range(i + 1, self.N):
                point_i = self.pointsList[i]
                point_j = self.pointsList[j]
                RSscore = self.getRSscore(point_i, point_j)
                if RSscore >= self.threScore:
                    self.Neps[i].add(j)
                    self.Neps[j].add(i)
    def getRSscore(self, point1, point2):
        distance = self.getDistance(point1, point2)
        duration = point2.timestamp - point1.timestamp
        RSscore = math.exp(-(distance/self.delta)-(distance/duration))
        return RSscore
    def reverse(self):
        self.m_reverse = {}
        for key in self.m.keys():
            value = self.m[key]
            if self.m_reverse.has_key(value):
                self.m_reverse[value].add(key)
            else:
                tmp = set([key])
                self.m_reverse[value] = tmp
    def getDistance(self, a, b):
        return haversine(a.lon, a.lat, b.lon, b.lat)
    def dump(self):
        stayPoints = {c: i for c, i in enumerate(self.stayPoints)}
        pickle.dump(stayPoints, open(self.save_path, 'wb'))
class StayPointExtracter():
    def __init__(self, save_path, distanceThreahold, timeThreadhold):
        self.stayPoints = set()
        self.distanceThreahold = distanceThreahold  #100m
        self.timeThreadhold = timeThreadhold  #10minutes
        self.save_path = save_path
    def extract(self, file_path, stayPoint_path):
        print '***********filter file: {}*************'.format(file_path)
        stayPointList = []
        _, pointsList = getPointsList(file_path)
        i = 0
        length = len(pointsList)
        while i < length:
            j = i+1
            while j < length:
                # print 'i = {}, j = {}'.format(i,j)
                point_i = pointsList[i]
                point_j = pointsList[j]
                distance = haversine(point_i.lon, point_i.lat, point_j.lon, point_j.lat)
                if distance >= self.distanceThreahold:
                    timeInterval = point_j.timestamp - point_i.timestamp
                    if timeInterval >= self.timeThreadhold:
                        lat, lon = computeMeanCoord(pointsList[i:j+1])
                        stayPoint = StayPoint(lat, lon, point_i.timestamp, point_j.timestamp)
                        stayPointList.append(stayPoint)
                    i = j
                    break
                j += 1
            if j == length:
                break
        if len(stayPointList) > 0:
            save(stayPointList, stayPoint_path)
        self.stayPoints |= set(stayPointList)
    def dump(self):
        stayPoints = {c: i for c, i in enumerate(self.stayPoints)}
        pickle.dump(stayPoints, open(self.save_path, 'wb'))

class ClusterTrajectoryMaker():
    def __init__(self, dbscanner_path, cluster_trajectory_save_path_total):
        self.dbscanner_path = dbscanner_path
        self.cluster_trajectory_save_path_total = cluster_trajectory_save_path_total
        self.dbscanner = pickle.load(open(self.dbscanner_path, 'r'))
        self.cluster_trajectory = []
        self.indexDict = {}
        for key in self.dbscanner.points:
            lat = self.dbscanner.points[key].lat
            lon = self.dbscanner.points[key].lon
            key_ = str(lat)+str(lon)
            self.indexDict[key_] = key

    def makeClusterTrajectoryFile(self, dir, save_file):
        if os.path.isdir(dir):
            cluster_trajectories = []
            stayPointsFiles = os.listdir(dir)
            for stayPointsFile in stayPointsFiles:
                stayPointsFile = os.path.join(dir, stayPointsFile)
                if os.path.isfile(stayPointsFile):
                    trajectory_str = ''
                    with open(stayPointsFile, 'r') as f:
                        for line in f:
                            # print line
                            line = line.strip()
                            lat, lon, arriveTime, leaveTime, _ = line.split(' ')
                            key_ = str(lat) + str(lon)
                            point_index = self.indexDict[key_]
                            cluster = self.dbscanner.m[point_index]
                            if cluster != -1:
                                trajectory_str += str(cluster) + ' '
                    if len(trajectory_str.strip())>0:
                        cluster_trajectories.append(trajectory_str)
                        self.cluster_trajectory.append(trajectory_str)
            save(cluster_trajectories, save_file)
    def save(self):
        save(self.cluster_trajectory, self.cluster_trajectory_save_path_total)
        cluster_trajectory_uniq = []
        for item in self.cluster_trajectory:
            item = item.strip()
            list = item.split(' ')
            list = [int(i) for i in list]
            list = uniq(list)
            def f(x, y):
                return str(x) + ' '+ str(y)
            string = reduce(f, list)
            cluster_trajectory_uniq.append(str(string))
        cluster_trajectory_uniq_save_path_total = self.cluster_trajectory_save_path_total.replace('.txt', '_uniq.txt')
        save(cluster_trajectory_uniq, cluster_trajectory_uniq_save_path_total)

def createVectorTrainingCorpus(file_path, windows, save_path):
    def separate(list, windows):
        res = []
        for i in range(len(list)-windows+1):
            res.append(list[i:i+windows])
        return res
    with open(file_path, 'r') as file:
        list_window = []
        for line in file:
            line = line.strip()
            list = [int(i) for i in line.split(' ')]
            if len(list) == 2:
                for i in range(windows - 2):
                    list.insert(0, 0)
            if len(list)>= windows:
                list_separate = separate(list, windows)
                list_window.extend(list_separate)
        save(list_window, save_path)

def createPredectTrainingCorpus1(file_path, num_steps_list, train_save_path_format, test_save_path_format):
    def feed(list, length):
        if len(list)<length:
            for i in range(length-len(list)):
                list.insert(0,-1)
        return list
    data_label = []
    trajectories = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            trajectories.append(line)
    random.shuffle(trajectories)
    index = int(len(trajectories)*0.8)
    train_trajectories = trajectories[0:index]
    test_trajectories = trajectories[index:len(trajectories)]


    for num_steps in num_steps_list:
        train_save_path = train_save_path_format.format(num_steps)
        test_save_path = test_save_path_format.format(num_steps)
        remove(train_save_path)
        remove(test_save_path)
        train_data_label = []
        for line in train_trajectories:
            list = [int(i) for i in line.split(' ')]
            if len(list) > 2:
                for i in range(len(list) - 1, 1, -1):
                    labels = list[max(0, i - num_steps + 1):i + 1]
                    labels = feed(labels, num_steps)
                    data = list[max(0, i - num_steps):i]
                    data = feed(data, num_steps)
                    result = reduce(concat, data) + '|' + reduce(concat, labels)
                    print result
                    train_data_label.append(result)
        test_data_label = []
        for line in test_trajectories:
            list = [int(i) for i in line.split(' ')]
            if len(list) > 2:
                for i in range(len(list) - 1, 1, -1):
                    labels = list[max(0, i - num_steps + 1):i + 1]
                    labels = feed(labels, num_steps)
                    data = list[max(0, i - num_steps):i]
                    data = feed(data, num_steps)
                    result = reduce(concat, data) + '|' + reduce(concat, labels)
                    print result
                    test_data_label.append(result)

        save(train_data_label, train_save_path)
        save(test_data_label, test_save_path)

def createPredectTrainingCorpus2(file_path, num_steps, train_save_path, test_save_path):
    def feed(list, length):
        if len(list)<length:
            for i in range(length-len(list)):
                list.insert(0,-1)
        return list
    data_label = []
    trajectories = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            trajectories.append(line)

    remove(train_save_path)
    remove(test_save_path)
    data_label = []
    for line in trajectories:
        list = [int(i) for i in line.split(' ')]
        if len(list) > 2:
            for i in range(len(list) - 1, 1, -1):
                labels = list[max(0, i - num_steps + 1):i + 1]
                labels = feed(labels, num_steps)
                data = list[max(0, i - num_steps):i]
                data = feed(data, num_steps)
                result = reduce(concat, data) + '|' + reduce(concat, labels)
                print result
                data_label.append(result)

    random.shuffle(data_label)
    index = int(len(data_label)*0.8)
    train_data_label = data_label[0:index]
    test_data_label = data_label[index:len(data_label)]
    save(train_data_label, train_save_path)
    save(test_data_label, test_save_path)

#不带-1的路径
def createPredectTrainingCorpusFull(file_path, num_steps, train_save_path, test_save_path):
    def feed(list, length):
        if len(list)<length:
            for i in range(length-len(list)):
                list.insert(0,-1)
        return list
    trajectories = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            trajectories.append(line)

    remove(train_save_path)
    remove(test_save_path)
    data_label = []
    for line in trajectories:
        list = [int(i) for i in line.split(' ')]
        if len(list) > num_steps:
            for i in range(len(list)-num_steps):
                labels = list[i+1:i+num_steps+1]
                data = list[i:i+num_steps]
                result = reduce(concat, data) + '|' + reduce(concat, labels)
                print result
                data_label.append(result)

    random.shuffle(data_label)
    index = int(len(data_label)*0.8)
    train_data_label = data_label[0:index]
    test_data_label = data_label[index:len(data_label)]
    save(train_data_label, train_save_path)
    save(test_data_label, test_save_path)

if __name__ == '__main__':
    dbscanner = pickle.load(open('./utils/dbscaner_100m_5minutes_50eps_5minPts.pkl', 'r'))
    pointList = []
    for key in dbscanner.m_reverse.keys():
        out = []
        for i in dbscanner.m_reverse[key]:
            lat = dbscanner.points[i].lat
            lon = dbscanner.points[i].lon
            point = Point(lat, lon, 39755.9735300926, 'asdfasdf')
            # string = '{},{},0,0,0,0,0'.format(lat, lon)
            out.append(point)
        save(out, './classes_100_5_50_5/class_{}.plt'.format(key))
        lat, lon = computeMeanCoord(out)
        pointList.append([key, lat, lon])
    for item in pointList:
        key = item[0]
        lat = item[1]
        lon = item[2]
        business = search(lat, lon)
        print '{}:{}'.format(key, business)