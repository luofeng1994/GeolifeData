#coding=utf-8
import time
import xml.dom.minidom
import os
import math
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
            k = ' '.join([str(j) for j in i])
            f.write(k + "\n")
        elif isinstance(i, basestring):
            f.write(i + "\n")
    f.close()
def getDate():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()).split(' ')[0]
def getTimestamp():
    return int(time.time())
def formatToTimeStamp(timeFormat):
    return time.mktime(time.strptime(timeFormat, "%Y/%m/%d %H:%M:%S"))

def parseXml(file):
    begin_timestamp = getTimestamp()
    prefix = os.path.split(file)[-1].split('.')[0]
    trajetories = {}
    # 打开xml文档
    dom = xml.dom.minidom.parse(file)

    # 得到文档元素对象
    root = dom.documentElement
    timesteps = root.getElementsByTagName('timestep')
    for timestep in timesteps:
        time = timestep.getAttribute('time')
        print time
        items = timestep.getElementsByTagName('vehicle')
        type = 'vehicle'
        if len(items)==0:
            items = timestep.getElementsByTagName('person')
            type = 'person'
        for item in items:
            id = item.getAttribute('id')
            longitude = item.getAttribute('x')
            longitude_cut = '0.{}'.format(longitude.split('.')[-1])
            latitude = item.getAttribute('y')
            latitude_cut = '0.{}'.format(latitude.split('.')[-1])
            angle = item.getAttribute('angle')
            speed = item.getAttribute('speed')
            speed_x = float(speed)*math.cos(float(str(angle)))
            speed_y = float(speed)*math.sin(float(str(angle)))
            key = '{}_{}{}'.format(prefix, type, id)
            if key in trajetories.keys():
                trajetories.get(key).append([longitude, latitude, longitude_cut, latitude_cut, angle, speed, speed_x, speed_y])
            else:
                trajetories[key] = [[longitude, latitude, longitude_cut, latitude_cut, angle, speed, speed_x, speed_y]]
    cur_dir = os.path.abspath('..')
    for key, trajetory in trajetories.items():
        path = '{}/{}/{}/{}.plt'.format(cur_dir, 'preprocess', 'trajetories', key)
        writeList(path, trajetory)


    aaa = 1

if __name__ == '__main__':
    files = os.listdir('./data')
    for file in files:
        file_path = os.path.join('./data', file)
        parseXml(file_path)
    aaa = 1