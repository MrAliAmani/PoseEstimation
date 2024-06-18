import numpy as np
import cv2 as cv

print(cv.__version__)

cam = cv.VideoCapture(0)
cam1 = cv.VideoCapture(1)

if (cam.isOpened() == False): 
  print("Unable to read camera feed")
if (cam1.isOpened() == False): 
  print("Unable to read camera feed")

frame_width = int(cam.get(3))
frame_height = int(cam.get(4))
frame_width1 = int(cam1.get(3))
frame_height1 = int(cam1.get(4))

out = cv.VideoWriter('outpy.mp4',cv.VideoWriter_fourcc('M', 'P', '4', 'V'), 30, (frame_width,frame_height))
out1 = cv.VideoWriter('outpy1.mp4',cv.VideoWriter_fourcc('M', 'P', '4', 'V'), 30, (frame_width1,frame_height1))

while(True):
  ret, frame = cam.read()
  ret1, frame1 = cam1.read()
  if ret == True: 
    out.write(frame)
    # Press Q on keyboard to stop recording
    if cv.waitKey(1) & 0xFF == ord('q'):
      break
  else:
    break 
  if ret1 == True: 
    out1.write(frame1)
    # Press Q on keyboard to stop recording
    if cv.waitKey(1) & 0xFF == ord('q'):
      break
  else:
    break 

cam.release()
out.release()
cam1.release()
out1.release()