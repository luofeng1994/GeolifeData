import time
def getTimestamp():
    return int(time.time())
def timestampToFormat(timestamp):
    time_local = time.localtime(timestamp)
    return time.strftime("%Y/%m/%d %H:%M:%S", time_local)
def formatToTimeStamp(timeFormat):
    return time.mktime(time.strptime(timeFormat, "%Y/%m/%d %H:%M:%S"))
print getTimestamp()
print timestampToFormat(getTimestamp())
print formatToTimeStamp(timestampToFormat(getTimestamp()))