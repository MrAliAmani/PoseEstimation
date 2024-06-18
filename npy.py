import numpy as np

path = input('file path:')

fileName = input('output file name:')

frameCount = input('file frameCount:')

file = np.load(path+"\\"+fileName+".npy", "r")

f = open(path+"\\"+fileName+".txt", "w")

for i in range(0, int(frameCount), 1):
    for j in range(0, 17, 1):
        f.write(str(file[i][j])+"\t")
    f.write("\n")

