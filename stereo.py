import cv2 as cv
import json
import string
import array

PixeltoMeters=0.0002645833
focal_length=0.13890625                     #525 pixels=0.13890625meters
T=0.28

path = input('file path: ')
vision1=input('vision1 video:')
vision2=input('vision2 video:')
fileNumber=input('last file number:')
fileName=input('output file name:')

number=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
xl=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0]
xr=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0]
disparity=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1]
z=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0]
y=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0]

imagepath = r'C:\Users\ali\Pictures\Camera Roll\1.jpg'

scale=5000

SkeletonConnectionMap = [10, 11,
                         9, 10,
                         12, 13,
                         13, 14,
                         8, 9,
                         8, 12,
                         5, 6,
                         6, 7,
                         2, 3,
                         3, 4,
                         1, 2,
                         1, 5]

color = (0, 255, 0) 

thickness = 9

window_name = 'Image'

f = open(fileName+".txt", "w")

for num in range(0,int(fileNumber)+1,1):
    n=num
    for i in range(0,12,1):
        number[i]=n%10
        n=n//10
        i=i+1

    with open(path+"\\"+vision1+"_"+str(number[11])+str(number[10])+str(number[9])+str(number[8])+str(number[7])+str(number[6])+str(number[5])+str(number[4])+str(number[3])+str(number[2])+str(number[1])+str(number[0])+"_keypoints.json","r") as jsf:
        data1 = jsf.read()
        jsf.close()

    with open(path+"\\"+vision2+"_"+str(number[11])+str(number[10])+str(number[9])+str(number[8])+str(number[7])+str(number[6])+str(number[5])+str(number[4])+str(number[3])+str(number[2])+str(number[1])+str(number[0])+"_keypoints.json","r") as jsf:
        data2 = jsf.read()
        jsf.close()

    keypoints1=json.loads(data1)["people"][0]["pose_keypoints_2d"]
    keypoints2=json.loads(data2)["people"][0]["pose_keypoints_2d"]

    for i in range(0,24,1):
        xl[i]=keypoints1[i*3]*PixeltoMeters
        xr[i]=keypoints2[i*3]*PixeltoMeters

        y[i]=keypoints1[i*3+1]*PixeltoMeters

        disparity[i]=(xr[i]-xl[i])

        f.write(str(xl[i])+"\t"+str(y[i])+"\t")

        if disparity[i] != 0:
            z[i]=focal_length*T/disparity[i]
            f.write(str(z[i])+"\n")
        else:
            f.write("disparity=0\n")
        
    f.write("\n")
    
    image = cv.imread(imagepath) 

    for i in range(0,23,2):

            start_point = (int(xl[ SkeletonConnectionMap[i] ] * scale ), int(y[ SkeletonConnectionMap[i] ] * scale) ) 
  
            end_point = (int(xl[SkeletonConnectionMap[i+1] ] * scale), int(y[SkeletonConnectionMap[i+1] ] * scale) ) 

            image = cv.line(image, start_point, end_point, color, thickness) 
    
            cv.imshow(window_name, image) 

            cv.waitKey(100)

f.close()
        

