# -*- coding: utf-8 -*-
import time
import os
from math import radians, cos, sin, asin, sqrt
import utm
from itertools import chain
from preprocessing import filter
import cPickle as pickle
from preprocessing import StayPointExtracter
from preprocessing import haversine

class DBSCAN():
    def __init__(self, Eps, MinSamplie):
        self.eps = Eps
        self.minPts = MinSamplie
    def cluster(self, points):
        self.points = points
        self.N = len(points)
        self.initialize()

        I = [i for i in range(self.N)]
        while len(I) > 0:
            i = I.pop(0)
            print 'i:{}   k:{}'.format(i, self.k)
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
                        # print '         j:{}   T:{}'.format(j, T)
                    self.k += 1

        self.reverse()
    def initialize(self):
        self.k = 1
        self.m = {i:0 for i in range(self.N)}
        self.Neps = {i:set([i]) for i in range(self.N)}
        self.initlizeNeps()
    def initlizeNeps(self):
        for i in range(self.N - 1):
            if i%100 == 0:
                print 'initlizeNeps: {}'.format(i)
            for j in range(i + 1, self.N):
                stayPoint_i = self.points[i]
                stayPoint_j = self.points[j]
                distance = self.getDistance(stayPoint_i, stayPoint_j)
                if distance <= self.eps:
                    self.Neps[i].add(j)
                    self.Neps[j].add(i)
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
