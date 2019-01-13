import cv2
import numpy as np
path='../data/video/4_1.mp4'
savepath1='../data/video/4_1_l.avi'
savepath2='../data/video/4_1_r.avi'
stream = cv2.VideoCapture(path)
fourcc=cv2.VideoWriter_fourcc(*'XVID')
fps=10
frameSize=(960,1080)
video1 = cv2.VideoWriter(savepath1, fourcc, fps, frameSize)
video2 = cv2.VideoWriter(savepath2, fourcc, fps, frameSize)
_,a=stream.read()
count=1
while(_):
	if(count==3):
		# print(np.shape(a[:,:960,:]))
		video1.write(a[:,:960,:])
		video2.write(a[:,960:,:])
		count=0
	_,a=stream.read()
	count+=1
	

