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

		img = cv.drawKeypoints(img, kp)

		cv.imshow("SIFT", img)

		cv.waitKey()
		cv.destroyAllWindows()

def downscale(img):
	assert(len(img.shape) == 2)
	rows, cols = img.shape
	scaledImg = np.ndarray((rows/2, cols/2), np.uint8)

	for y in range(0, scaledImg.shape[0]):
		for x in range(0, scaledImg.shape[1]):
			scaledImg[y][x] = img[2*y][2*x]

	return scaledImg

def getGaussianKernel(sigma=1.6, kernelHeight=51, kernelWidth=51):
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
	gp = []

	sigma = 1.6

	octaveCount = 4
	sigmaCount = 3

	# vgl. https://courses.cs.washington.edu/courses/cse576/11sp/notes/SIFT_white2011.pdf
	for o in range(0, octaveCount):
		gp.append(bluredImg.copy())
		for s in range(1, sigmaCount + 1):
			k = 2**(float(s)/float(sigmaCount))
			kernel = getGaussianKernel(k*sigma)
			bluredImg = cv.filter2D(img, -1, kernel)
			gp.append(bluredImg.copy())

		if (o < octaveCount-1):
			#sigma *= 2
			img = downscale(img)
			bluredImg = downscale(bluredImg)

	return gp

def calcDifference(img1, img2, threshold = 0):
	#return cv.absdiff(img1, img2);
	#return abs(img2-img1)
	diffImg = np.ndarray(img1.shape, np.uint8)

	for y in range(diffImg.shape[0]):
		for x in range(diffImg.shape[1]):
			difference = abs(int(img2[y][x]) - int(img1[y][x]))
			if difference > threshold:
				diffImg[y][x] = difference
			else:
				diffImg[y][x] = 0

	return diffImg


def calcDoG(gp):
	DoG = []

	octaveCount = 3
	sigmaCount = 3

	for i in range(len(gp)-1):
		gauss1 = gp[i]
		gauss2 = gp[i+1]

		if (gauss1.shape == gauss2.shape):
			DoG.append(calcDifference(gauss1, gauss2))

	return DoG

def normalize(img):
	normImg = np.ndarray(img.shape, np.float64)

	max = img.max()
	if max > 0:
		normImg = img/float(max)

	return normImg


def SIFT():
	for filename in os.listdir(PATH):
		print filename
		img = cv.imread(PATH+filename, cv.IMREAD_GRAYSCALE)

		gp = calcGaussianPyramid(img)
		DoG = calcDoG(gp)

		for img in DoG:
			cv.imshow("SIFT", normalize(img))

			cv.waitKey()
			cv.destroyAllWindows()




if __name__ == "__main__":
	SIFT()
	#SIFT_OPENCV()
