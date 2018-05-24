import urllib2
import re
import json
# response = urllib2.urlopen('http://api.map.baidu.com/geocoder/v2/?coordtype=wgs84ll&callback=renderReverse&location=39.983424,116.322987&output=json&pois=1&ak=H0nGiz18dfTjLuSRytP5uUTKYaZTPNVK')
# out = response.read()
# jsonStr = re.findall(r"renderReverse&&renderReverse\((.+?)\)$",out)[0]
# print jsonStr

def search(lat, lon):
    try:
        url = 'http://api.map.baidu.com/geocoder/v2/?coordtype=wgs84ll&callback=renderReverse&location={},{}&output=json&pois=1&ak=H0nGiz18dfTjLuSRytP5uUTKYaZTPNVK'
        url = url.format(lat, lon)
        response = urllib2.urlopen(url)
        out = response.read()
        jsonStr = re.findall(r"renderReverse&&renderReverse\((.+?)\)$",out)[0]
        jsonObj = json.loads(jsonStr)
        business = jsonObj['result']['business']
        return business
    except:
        return 'null'

if __name__ == '__main__':
    print search(39.983424,116.322987)