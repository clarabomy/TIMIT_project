#!/bin/python
from pip._internal import main as pipmain
import os

#Auto install the ffmpeg-python (with pip > 10)
try:
    import ffmpeg
except:
    print("Package ffmpeg-python not installed")
    print("Installing")
    pipmain(["install", "--user", "ffmpeg-python"])

files = open("TIMIT/allfilelist.txt","r")
lines = files.readlines()
file_list = [x[:-1]+".wav" for x in lines]
for file in file_list:
    file = file.upper()
    stream = ffmpeg.input("TIMIT/" + file)
    folder = "/".join(file.split("/")[:-1])
    if not os.path.exists("TIMIT_wav/" + folder):
        os.makedirs("TIMIT_wav/"  + folder)
    stream = ffmpeg.output(stream, "TIMIT_wav/" + file)
    ffmpeg.run(stream)
    print(file)
