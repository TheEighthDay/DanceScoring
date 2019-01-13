import cv2
import os

img_files=['data/image/1/a','data/image/1/b','data/image/2/c','data/image/2/d','data/image/2/e','data/image/3/f',
'data/image/3/g','data/image/3/h','data/image/4/i','data/image/4/j','data/image/4/k','data/image/5/l','data/image/5/m',
'data/image/6/n','data/image/6/o','data/image/7/p','data/image/7/q','data/image/7/r','data/image/8/s','data/image/8/t',
'data/image/8/u','data/image/9/v','data/image/9/w','data/image/9/x','data/image/10/y','data/image/10/z',
'data/image/11/aa','data/image/11/bb']
seconds=[103,103,216,216,216,78,78,78,224,224,224,119,119,19,19,120,120,120,177,177,177,127,127,127,198,198,258,258]

for w in range(len(img_files)):
	print(img_files[w])
	img_list=os.listdir("../"+img_files[w])

	tmp=-1

	for img in img_list:
		m=int(img.split("_")[0])
		n=int(img.split("_")[1].split(".")[0])

		if (tmp+1)%10 != n:
			print(m,n)
		tmp=n







