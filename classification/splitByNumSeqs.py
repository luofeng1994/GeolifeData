# -*- coding: utf-8 -*-
import time
import os
from math import radians, cos, sin, asin, sqrt
import utm
from itertools import chain
from labels import writeList1 as writeList

type_num = {}

def increaseDic(dic, key):
    if key in dic:
        dic[key] = dic[key] + 1
    else:
        dic[key] = 1
def splitPerFile(file_path, num_seqs):
    type = file_path.split('/')[-1].split('-')[0]
    file_data = []
    with open(file_path) as file:
        file_data = [line for line in file]
        # print file_data
    for i in range(0,len(file_data)-num_seqs, num_seqs):
        print i
        increaseDic(type_num, type)
        file_name = './2017-09-27 18-05-32 splitByNumSeqs/train/' + type + '-' + str(type_num[type]) + '.plt'
        writeList(file_name, file_data[i:i+num_seqs])

def split(rootdir, num_seqs):
    files = os.listdir(rootdir)
    for file in files:
        file = os.path.join(rootdir, file)
        if os.path.isfile(file):
            splitPerFile(file, num_seqs)

if __name__ == '__main__':
    num_seqs = 50
    split('./2017-09-27 18-05-32/train/', num_seqs)