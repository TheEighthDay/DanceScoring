import cv2
import numpy as np
# im=cv2.imread("../data/image/1/a/0_0.jpg")
# print(im.shape)
def let_people_full_of_screen(im):
	height_start=0
	for i in range(im.shape[0]):
		if(np.sum(im[i])!=0):
			height_start=i
			break
	# print(height_start)
	height_end=0
	for i in range(im.shape[0]-1,-1,-1):
		if(np.sum(im[i])!=0):
			height_end=i
			break
	# print(height_end)


	width_start=0
	for i in range(im.shape[1]):
		if(np.sum(im[:,i])!=0):
			width_start=i
			break
	# print(width_start)
	width_end=0
	for i in range(im.shape[1]-1,-1,-1):
		if(np.sum(im[:,i])!=0):
			width_end=i
			break
	# print(width_end)

	c=cv2.resize(im[height_start:height_end,width_start:width_end],(300,300))
	return c
