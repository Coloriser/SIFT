import cv2 as cv
import numpy as np
import os

PATH = ".\\test2\\"
PI = 3.14159265
E = 2.71828182

def SIFT_OPENCV():
	for filename in os.listdir(PATH):
		print filename
		img = cv.imread(PATH+filename, cv.IMREAD_GRAYSCALE)

		sift = cv.SIFT()

		kp = sift.detect(img, None)

		img = cv.drawKeypoints(img, kp, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

		cv.imshow("SIFT", img)

		cv.waitKey()
		cv.destroyAllWindows()


def normalize(img):
	normImg = np.ndarray(shape=img.shape, dtype=np.float64)

	max = img.max()
	if max > 0:
		normImg = img/float(max)
		normImg *= 255.
	else:
		return img.copy()

	return normImg.astype(np.uint8)

def scale(img, factor=2):
	assert(len(img.shape) == 2)
	rows, cols = img.shape
	scaledImg = np.ndarray((rows*factor, cols*factor), np.float64)

	for y in range(0, scaledImg.shape[0]):
		for x in range(0, scaledImg.shape[1]):
			scaledImg[y][x] = img[y/factor][x/factor]

	return scaledImg

def downscale(img):
	assert(len(img.shape) == 2)
	rows, cols = img.shape
	scaledImg = np.ndarray((rows/2, cols/2), np.float64)

	for y in range(0, scaledImg.shape[0]):
		for x in range(0, scaledImg.shape[1]):
			scaledImg[y][x] = img[2*y][2*x]

	return scaledImg

def getGaussianKernel(sigma, kernelHeight=51, kernelWidth=51):
	assert(kernelHeight % 2 == 1 and kernelWidth % 2 == 1)

	yOffset = (kernelHeight - 1) / 2
	xOffset = (kernelWidth - 1) / 2

	kernel = np.ndarray((kernelHeight, kernelWidth), np.float64)

	for y in range(-yOffset, yOffset+1, 1):
		for x in range(-xOffset, xOffset+1, 1):
			kernel[y+yOffset][x+xOffset] = (1. / (2.*PI*sigma**2)) * E**(-(x**2 + y**2) / (2 * sigma**2))

	# normalize kernel
	kernel /= kernel.sum()
	return kernel


def calcGaussianPyramid(org_img):
	img = org_img.copy()
	bluredImg = img.copy()

	sigma = 1.6

	octaveCount = 7
	sigmaCount = 4
	
	gp = np.ndarray(shape=(octaveCount,), dtype=np.ndarray)

	# vgl. https://courses.cs.washington.edu/courses/cse576/11sp/notes/SIFT_white2011.pdf
	for o in range(0, octaveCount):
		gp[o] = np.ndarray(shape=(sigmaCount+1, img.shape[0], img.shape[1]), dtype=np.float64)
		gp[o][0] = bluredImg.copy()
		for s in range(1, sigmaCount + 1):
			k = 2**(float(s)/float(sigmaCount))
			kernel = getGaussianKernel(k*sigma)
			bluredImg = cv.filter2D(img, -1, kernel)
			gp[o][s] = bluredImg.copy()

		if (o < octaveCount-1):
			#sigma *= 2
			img = downscale(img)
			bluredImg = downscale(bluredImg)

	return gp

def calcDifference(img0, img1, threshold = 0):
	assert(img0.shape == img1.shape)
	#return cv.absdiff(img0, img1);
	#return abs(img1-img0)
	diffImg = np.ndarray(img0.shape, np.float64)

	for y in range(diffImg.shape[0]):
		for x in range(diffImg.shape[1]):
			difference = abs(img1[y][x] - img0[y][x])
			if difference > threshold:
				diffImg[y][x] = difference
			else:
				diffImg[y][x] = 0

	return diffImg


def calcDoG(gp):
	#octaveCount = gp.shape[0]
	#sigmaCount = gp[0].shape[0]

	DoG = np.ndarray(shape=gp.shape, dtype=np.ndarray)

	for o in range(DoG.shape[0]):
		DoG[o] = np.ndarray(shape=(gp[o].shape[0]-1, gp[o].shape[1], gp[o].shape[2]), dtype=np.float64)
		for s in range(DoG[o].shape[0]):
			DoG[o][s] = calcDifference(gp[o][s], gp[o][s+1])

	return DoG

def getNeighbourhood(octave, s, y, x, radius=1):
	neighbourhood = octave[s-radius:s+radius+1, y-radius:y+radius+1, x-radius:x+radius+1]
	#neighbourhood[1, 1, 1] = neighbourhood[0, 0, 0]

	return neighbourhood

def calcExtrema(DoG, threshold=1, radius=1):
	keypoints = np.ndarray(shape=DoG.shape, dtype=np.ndarray)

	sigma = 1.6
	sigmaCount = DoG[0].shape[0]

	for o in range(DoG.shape[0]):
		keypoints[o] = np.ndarray(shape=(DoG[o].shape[0]-(2*radius),), dtype=list)
		for s in range(radius, DoG[o].shape[0]-radius):
			keypoints[o][s-radius] = []
			k = 2**(float(s)/float(sigmaCount))

			# -1 --> ignore borders
			# needs some serious speed improvement!
			for y in range(radius, DoG[o].shape[1]-radius):
				for x in range(radius, DoG[o].shape[2]-radius):
					value = DoG[o][s, y, x]
					neighbourhood = getNeighbourhood(DoG[o], s, y, x, radius=radius).flatten()
					neighbourhood.sort()
					min2 = neighbourhood[1]
					max2 = neighbourhood[-2]
					if value < min2 or (value > threshold and value > max2):
						scale = 2**o
						keypoints[o][s-radius].append((scale * y + scale/2, scale * x + scale/2, scale * k*sigma)) #y-pos, x-pos and scale

	return keypoints

def drawKeypoints(img, kp):
	if (len(img.shape) < 3 or img.shape[2] == 1):
		kpImg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
	else:
		kpImg = img.copy()
	for y, x, scale in kp:
		cv.circle(kpImg, (x, y), int(scale), (0, 0, 255))

	return kpImg

def SIFT():
	for filename in os.listdir(PATH):
		print filename
		img = cv.imread(PATH+filename, cv.IMREAD_GRAYSCALE)

		gp = calcGaussianPyramid(img)
		DoG = calcDoG(gp)

		radius = 1
		keypoints = calcExtrema(DoG, radius=radius)
		kpImg = img.copy()
		for o in range(keypoints.shape[0]):
			for s in range(radius, DoG[o].shape[0]-radius):
				kp = keypoints[o][s-radius]
				kpImg = drawKeypoints(kpImg, kp)

				cv.imshow("SIFT", drawKeypoints(normalize(scale(DoG[o][s], 2**o)), kp))

				cv.waitKey()
				cv.destroyAllWindows()
		
		cv.imshow("SIFT", kpImg)

		cv.waitKey()
		cv.destroyAllWindows()




if __name__ == "__main__":
	SIFT_OPENCV()
	SIFT()
	#SIFT_OPENCV()
