# -*- coding: utf-8 -*-
import time
import os
from math import radians, cos, sin, asin, sqrt
import utm
from itertools import chain
import random
import cPickle as pickle
from preprocessing import filter
from preprocessing import Point
from preprocessing import computeMeanCoord
from preprocessing import remove
from preprocessing import StayPointExtracter
from preprocessing import StayPointExtracterRS
from preprocessing import ClusterTrajectoryMaker
from preprocessing import createVectorTrainingCorpus
from preprocessing import createPredectTrainingCorpus1
from preprocessing import createPredectTrainingCorpus2
from preprocessing import createPredectTrainingCorpusFull
from preprocessing import save
from baiduSearch import search
from cluster import DBSCAN


def separateClassesAndPlaceName():
    dbscanner_path = './utils/dbscaner_200m_10minutes_50eps_5minPts.pkl'
    classes_save_dir = './classes/classes_200_10_50_5/'
    place_name_save_path = './utils/dbscaner_200m_10minutes_50eps_5minPts.txt'

    dbscanner = pickle.load(open(dbscanner_path, 'r'))
    pointList = []
    for key in dbscanner.m_reverse.keys():
        print key
        out = []
        kk = []
        for i in dbscanner.m_reverse[key]:
            lat = dbscanner.points[i].lat
            lon = dbscanner.points[i].lon
            point = Point(lat, lon, 39755.9735300926, 'asdfasdf')
            string = '{},{},0,0,0,0,0'.format(lat, lon)
            out.append(point)
            kk.append(string)
        save_path = '{}class_{}.plt'.format(classes_save_dir,key)
        remove(save_path)
        save(kk, save_path)
        lat, lon = computeMeanCoord(out)
        pointList.append([key, lat, lon])

    placeNameList = []
    for item in pointList:
        key = item[0]
        lat = item[1]
        lon = item[2]
        business = search(lat, lon)
        result = '{}:{}'.format(key, business)
        placeNameList.append(result)
        print '{}:{}'.format(key, business)
    save(placeNameList, place_name_save_path)


def filterOutlier():
    rootdir = unicode('E:\实验室\项目\轨迹预测与分类\gps data\Geolife Trajectories 1.3\Data', 'utf8')
    person_dirs = os.listdir(rootdir)
    for person_dir in person_dirs:
        person_dir = os.path.join(rootdir, person_dir)
        if os.path.isdir(person_dir):
            person_trajectory_dir = os.path.join(person_dir, 'Trajectory')
            person_trajectory_files = os.listdir(person_trajectory_dir)
            for person_trajectory_file in person_trajectory_files:
                person_trajectory_file = os.path.join(person_trajectory_dir, person_trajectory_file)
                if os.path.isfile(person_trajectory_file):
                    filterd_path = person_trajectory_file.replace('Trajectory', 'Trajectory_filtered')
                    filterd_path, name = os.path.split(filterd_path)
                    name = name.split('.')[0] + '_filtered.plt'
                    filterd_path = os.path.join(filterd_path, name)
                    if os.path.isfile(filterd_path):
                        os.remove(filterd_path) #先把上一次的删掉
                    filter(person_trajectory_file, filterd_path)
#太耗时了
def extractStayPointRS():
    rootdir = unicode('E:\实验室\项目\轨迹预测与分类\gps data\Geolife Trajectories 1.3\Data', 'utf8')
    stayPoint_save_path_total = './utils/stayPointRS_100m_5minPts_0.8thred.pkl'
    stayPoint_trajectory_save_dir_for_each_person = 'StayPointsRS_100m_5minPts_0.8thred'
    stayPointExtrater = StayPointExtracterRS(stayPoint_save_path_total)
    person_dirs = os.listdir(rootdir)
    for person_dir in person_dirs:
        person_dir = os.path.join(rootdir, person_dir)
        if os.path.isdir(person_dir):
            person_trajectory_dir = os.path.join(person_dir, 'Trajectory_filtered')
            person_trajectory_files = os.listdir(person_trajectory_dir)
            for person_trajectory_file in person_trajectory_files:
                person_trajectory_file = os.path.join(person_trajectory_dir, person_trajectory_file)
                if os.path.isfile(person_trajectory_file):
                    stayPoint_path = person_trajectory_file.replace('Trajectory_filtered', stayPoint_trajectory_save_dir_for_each_person)
                    stayPoint_path = stayPoint_path.replace('_filtered', '_stayPoints')
                    if os.path.isfile(stayPoint_path):
                        os.remove(stayPoint_path) #先把上一次的删掉
                    stayPointExtrater.extract(person_trajectory_file, stayPoint_path)
    stayPointExtrater.dump()

def extractStayPoint():
    rootdir = unicode('E:\实验室\项目\轨迹预测与分类\gps data\Geolife Trajectories 1.3\Data', 'utf8')
    stayPoint_save_path_total = './utils/stayPoint_200m_10minutes.pkl'
    stayPoint_trajectory_save_dir_for_each_person = 'StayPoints_200m_10minutes'
    stayPointExtrater = StayPointExtracter(stayPoint_save_path_total, 200, 10*60)
    person_dirs = os.listdir(rootdir)
    for person_dir in person_dirs:
        person_dir = os.path.join(rootdir, person_dir)
        if os.path.isdir(person_dir):
            person_trajectory_dir = os.path.join(person_dir, 'Trajectory_filtered')
            person_trajectory_files = os.listdir(person_trajectory_dir)
            for person_trajectory_file in person_trajectory_files:
                person_trajectory_file = os.path.join(person_trajectory_dir, person_trajectory_file)
                if os.path.isfile(person_trajectory_file):
                    stayPoint_path = person_trajectory_file.replace('Trajectory_filtered', stayPoint_trajectory_save_dir_for_each_person)
                    stayPoint_path = stayPoint_path.replace('_filtered', '_stayPoints')
                    if os.path.isfile(stayPoint_path):
                        os.remove(stayPoint_path) #先把上一次的删掉
                    stayPointExtrater.extract(person_trajectory_file, stayPoint_path)
    stayPointExtrater.dump()

def cluster():
    stayPoints = pickle.load(open('./utils/stayPoint_100m_5minutes.pkl', 'r'))
    dbscaner_result_save_path = './utils/dbscaner_100m_5minutes_50eps_5minPts.pkl'
    dbscaner = DBSCAN(50, 5)
    dbscaner.cluster(stayPoints)
    dbscaner_path = dbscaner_result_save_path
    if os.path.isfile(dbscaner_path):
        os.remove(dbscaner_path)
    pickle.dump(dbscaner, open(dbscaner_path, 'wb'))

def makeClusterTrajectory():
    rootdir = unicode('E:\实验室\项目\轨迹预测与分类\gps data\Geolife Trajectories 1.3\Data', 'utf8')
    dbscanner_path = './utils/dbscaner_100m_5minutes_50eps_5minPts.pkl'
    stayPoint_trajectory_dir_for_each_person = 'StayPoints_100m_5minutes'
    cluster_trajectory_save_path_total = './utils/clusterTrajectory_100m_5minutes_50eps_5minPts.txt'
    person_dirs = os.listdir(rootdir)
    clusterTrajectoryMaker = ClusterTrajectoryMaker(dbscanner_path, cluster_trajectory_save_path_total)
    for person_dir in person_dirs:
        person_dir = os.path.join(rootdir, person_dir)
        if os.path.isdir(person_dir):
            stayPoint_dir = stayPoint_trajectory_dir_for_each_person
            person_staypoint_dir = os.path.join(person_dir, stayPoint_dir)
            cluster_trajectory_save_path_for_each_person = os.path.join(person_dir, os.path.split(dbscanner_path)[-1].split('.')[0] + '_cluster_trajectory.txt')
            if os.path.isfile(cluster_trajectory_save_path_for_each_person):
                os.remove(cluster_trajectory_save_path_for_each_person)
            clusterTrajectoryMaker.makeClusterTrajectoryFile(person_staypoint_dir, cluster_trajectory_save_path_for_each_person)
    clusterTrajectoryMaker.save()

def makeVectorTrainingCorpus():
    read_path = './utils/clusterTrajectory_200m_10minutes_50eps_5minPts_uniq.txt'
    windows = 2
    save_path = './vector_training_corpus/clusterTrajectory_200m_10minutes_50eps_5minPts_uniq_{}windows.txt'.format(windows)
    createVectorTrainingCorpus(read_path, windows, save_path)

def makePredictTrainingCorpus1():
    read_path = './utils/clusterTrajectory_100m_5minutes_50eps_5minPts_uniq.txt'
    numseqs = [2,3,4,5]
    train_save_path = './predict_training_corpus/clusterTrajectory_100m_5minutes_50eps_5minPts_uniq_{}numseqs_train.txt'
    test_save_path = './predict_training_corpus/clusterTrajectory_100m_5minutes_50eps_5minPts_uniq_{}numseqs_test.txt'
    createPredectTrainingCorpus1(read_path, numseqs, train_save_path, test_save_path)

def makePredictTrainingCorpus2():
    read_path = './utils/clusterTrajectory_100m_10minutes_50eps_5minPts_uniq.txt'
    numseqs = 3
    train_save_path = './predict_training_corpus/clusterTrajectory_100m_10minutes_50eps_5minPts_uniq_{}numseqs_train.txt'.format(numseqs)
    test_save_path = './predict_training_corpus/clusterTrajectory_100m_10minutes_50eps_5minPts_uniq_{}numseqs_test.txt'.format(numseqs)
    createPredectTrainingCorpusFull(read_path, numseqs, train_save_path, test_save_path)

def main():
    # filterOutlier() #过滤异常值
    # extractStayPoint() #从路径中提取stayPoints
    # extractStayPointRS() #从路径中提取stayPoints,利用论文中的方法
    # cluster()　#stayPoints聚类，提取重要位置
    # makeClusterTrajectory() #将路径换成由重要位置构成的路径
    # makeVectorTrainingCorpus() #制造预料，训练vector
    makePredictTrainingCorpus2()
    # separateClassesAndPlaceName()
    aaa = 1


if __name__ == '__main__':
    main()