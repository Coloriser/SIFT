import cv2 as cv
import numpy as np
import os




def main():
	path = ".\\tests\\"
	for filename in os.listdir(path):
		print filename
		test_img = cv.imread(path+filename, cv.IMREAD_GRAYSCALE)

		img = cv.SIFT(test_img)

		cv.imshow("SIFT", img)

		cv.waitKey()
		cv.destroyAllWindows()

if __name__ == "__main__":
	main()
