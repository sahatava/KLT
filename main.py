from PIL import Image, ImageDraw
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as LA
from scipy import signal
import random
import math
import time
import scipy.ndimage
import cv2


##########################################################################################################################################
def HessiancornerDetection(I, threshold1, threshold2):
	# params for ShiTomasi corner detection
	feature_params = dict( maxCorners = 100,
		        qualityLevel = 0.3,
		        minDistance = 7,
		        blockSize = 7 )
	
	# the start time in running the program is t1
	t1 = time.time()
	# make the Hessian matrix 
	n, m = I.shape
	Ix = zeros((n,m))
	Iy = zeros((n,m))	
	Ixx = zeros((n,m))
	Ixy = zeros((n,m))
	Iyy = zeros((n,m))

	# find the derivatives which are needed for making the Hessian matrix
	for j in range(1,m-1):
		Ix[:,j] = (I[:,j+1] - I[:,j-1])
	for j in range(1,m-1):
		Ixx[:,j] = (Ix[:,j+1] - Ix[:,j-1])
	for i in range(1,n-1):
		Ixy[i,:] = (Ix[i+1,:] - Ix[i-1,:])
	for i in range(1,n-1):
		Iy[i,:] = (I[i+1,:] - I[i-1,:])
	for i in range(1,n-1):
		Iyy[i,:] = (Iy[i+1,:] - Iy[i-1,:])

	# find the hessian matrix for each pixel
	landa1Image = zeros((n,m))
	landa2Image = zeros((n,m))
	landa1Image2 = zeros((n,m))
	landa2Image2 = zeros((n,m))
	x =[]
	y =[]
	figure()
	gray()
	imshow(I)
	title('Hessian corner detection')
	p0 = cv2.goodFeaturesToTrack(I1, mask = None, **feature_params)
	for i in range(n):
		for j in range(m):
			H = [[Ixx[i,j], Ixy[i,j]] ,[Ixy[i,j], Iyy[i,j]]]
			# find the eigen values of each hessian matrix
        		landa1Image[i,j] = ((LA.eig(H)[0][0])).real 
			landa2Image[i,j] = ((LA.eig(H)[0][1])).real
			# if both the eigen values are high that point will be selcted as a corner
  			if (landa1Image[i,j] > threshold1 and landa2Image[i,j] > threshold2):
				x.append(j)
	 			y.append(i)
	p0 = cv2.goodFeaturesToTrack(I1, mask = None, **feature_params)
	return p0

#################################################################################################
def LK(I1,I2,p0):
	lk_params = dict( winSize  = (15,15),
		   maxLevel = 3,
		   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
	# find the size of the images
	n , m = I1.shape
	i =1
	j = 1
	# find the derivitive Ix and Iy and It to compose the equations to find the optical flow
	Ix = signal.convolve(I1, [[-0.25,0.25],[-0.25,0.25]], mode='same')     
	Iy = signal.convolve(I1, [[-0.25,-0.25],[0.25,0.25]], mode='same') 
	It = signal.convolve(I1, [[0.25,0.25],[0.25,0.25]], mode='same') + signal.convolve(I2, [[-0.25,-0.25],[-0.25,-0.25]], mode='same')

	# define two initial matrix for velocity in x and y directions 
	U = np.empty((n,m))
	V = np.empty((n,m)) 

	# define the size of the window to extract the equations
	halfWindow = 3

	# for each pixel find all the information in the all pixels in the window, and set the equations
	 
	finalIx = Ix[i-halfWindow:i+halfWindow,j-halfWindow:j+halfWindow]
	finalIy = Iy[i-halfWindow:i+halfWindow,j-halfWindow:j+halfWindow]
	finalIt = It[i-halfWindow:i+halfWindow,j-halfWindow:j+halfWindow]

	a = finalIx.flatten()
	finalIx = a.reshape(len(a),1)
	 
	a = finalIy.flatten()
	finalIy = a.reshape(len(a),1)

	a = finalIt.flatten()
	finalIt = a.reshape(len(a),1)
	p1, st, err = cv2.calcOpticalFlowPyrLK(I1, I2, p0, None, **lk_params)
	A = np.concatenate((finalIx, finalIy), axis=1)
	velocity = np.dot( np.dot( LA.pinv(np.dot(A.T, A)),A.T), finalIt )
	U[i,j] = velocity[0]
	V[i,j] = velocity[1]
	 
 	return p1, st, err

#################################################################################################
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
		        qualityLevel = 0.3,
		        minDistance = 7,
		        blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
		   maxLevel = 3,
		   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

color = np.random.randint(0,255,(100,3))
#################################################################################################
def track(I1,I2,framename, bbox):
	
	#p0 = HessiancornerDetection(I1, 250, 250)
	p0 = cv2.goodFeaturesToTrack(I1, mask = None, **feature_params)
	# calculate optical flow
        p1, st, err = LK(I1,I2,p0)
 



	# Create a mask image for drawing purposes

	mask = np.zeros_like(I1)

	# Select good points
	good_new = p1[st==1]
	good_old = p0[st==1]
	n0,m0 = good_old.shape


	 
	#plt.imshow(I1,cmap = cm.gray)
	 

	print good_old
 
	count = 0 
	u = 0
	v = 0
	for i in range(n0):
		if (bbox[0] - 70  < good_old[i][0] < bbox[2] + 70) and (bbox[1]- 70 < good_old[i][1] < bbox[3]+ 70):
			count +=1
			v = (good_new[i][0] - good_old[i][0])
			u = (good_new[i][1] - good_old[i][1])
	#u = -1.5*int((float(u))) #vertival 
	#v = 1*int((float(v)))
	print u,v
	#plot new circle in second image
	im = Image.open(str(framename)+".jpg")
 
	# updating the bbox
	bbox =  (bbox[0] + v, bbox[1] +u,bbox[2] + v, bbox[3] + u)
	draw = ImageDraw.Draw(im)
	draw.ellipse(bbox, fill=None)
	del draw


	im.save("output"+str(framename)+".png")
	 
	plt.show()
	return bbox
#############################################################################################################3

I1 = array(Image.open('1.jpg').convert('L'))  
I2 = array(Image.open('2.jpg').convert('L')) 

# first iteration
########p0 = HessiancornerDetection(I1, 250, 250)
p0 = cv2.goodFeaturesToTrack(I1, mask = None, **feature_params)
# calculate optical flow
p1, st, err = LK(I1,I2,p0)
 
# Create a mask image for drawing purposes
mask = np.zeros_like(I1)
# Select good points
good_new = p1[st==1]
good_old = p0[st==1]
n0,m0 = good_old.shape
 
im = Image.open("1.jpg")

x, y =  im.size
eX, eY = 50, 100 #Size of Bounding Box for ellipse

bbox =  (310 - eX/2, 330 - eY/2, 310 + eX/2, 330 + eY/2)
draw = ImageDraw.Draw(im)
draw.ellipse(bbox, fill=None)
del draw

im.save("output1.png")
im2 = Image.open("output1.png")
plt.subplot(121)
plt.imshow(im2)
#im.show()
count = 0 
u = 0
v = 0
for i in range(n0):
	if (310 - eX/2 < good_old[i][0] < 310 + eX/2) and (330 - eY/2 < good_old[i][1] < 330 + eY/2):
		count +=1
		v = (good_new[i][0] - good_old[i][0])
		u = (good_new[i][1] - good_old[i][1])
u = 8*(float(u))
v = (float(v))
print u,v
#plot new circle in second image
im = Image.open("2.jpg")

x, y =  im.size
eX, eY = 50 , 100 #Size of Bounding Box for ellipse
bbox0 =  (310 + v  - eX/2, 320 + u - eY/2, 310 + v + eX/2, 320+ u + eY/2)
draw = ImageDraw.Draw(im)
draw.ellipse(bbox, fill=None)
del draw


im.save("output2.png")
im3 = Image.open("output2.png")
plt.subplot(122)
plt.imshow(im3)	

plt.show()

####################################
 
for framename in range(2,64):
	I1 = array(Image.open(str(framename)+".jpg").convert('L'))  
	I2 = array(Image.open(str(framename+1)+".jpg").convert('L'))

        if framename == 2:
		bbox = track(I1,I2,framename+1,bbox0)

        else:
	        bbox = track(I1,I2,framename+1,bbox)


	#im2 = Image.open("output"+str(framename)+".png")
	#plt.subplot(121)
	#plt.imshow(im2)

	#im3 = Image.open("output"+str(framename+1)+".png")
	#plt.subplot(122)
	#plt.imshow(im3)
	print framename

####################################
# convert the results to a video



fourcc=cv2.VideoWriter_fourcc('I','Y','U','V')
 
out = cv2.VideoWriter('out_video.avi', fourcc, 24, (704, 240))

#c = cv2.VideoCapture('in_video.avi')

img1 = cv2.imread('img1.jpg')
img2 = cv2.imread('img2.jpg')
out.write(img1)  #write frame to the output video
out.write(img2)

out.release()
cv2.destroyAllWindows()
#c.release()




































 
