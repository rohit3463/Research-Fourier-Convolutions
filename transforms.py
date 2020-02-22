import numpy as np
import cv2


def resize(img, outShape):
	'''
		sample :img = resize(img, (1920,1080))
	'''
	assert isinstance(outShape, (list, tuple)) and len(outShape)==2
	return cv2.resize(img, outShape) 


def flip(img, axis = 0):
	return cv2.flip(img, axis)

def rotate(img, degree = 1):
	'''
		Args : degree : 1=90' clockwise, 2=180' , 3=90' counter-clockwise
	'''
	if(degree == 1):
		ro = cv2.ROTATE_90_CLOCKWISE
	elif (degree == 2):
		ro = cv2.ROTATE_180
	else : 
		ro = cv2.cv2.ROTATE_90_COUNTERCLOCKWISE
	return cv2.rotate(img, ro)

def gaussianBlur(img, kernel = (5,5)):
	return cv2.GaussianBlur(img, kernel,cv2.BORDER_DEFAULT) 


def sharpen(img, kernel=None):
	if(kernel == None):
		kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
	return cv2.filter2D(img, -1, kernel)


if __name__ == "__main__":
	# cap = cv2.VideoCapture(0)
	# _,img = cap.read()
	# cap.release()
	img = cv2.imread("contact-us-1208462_1280.png")
	print(img.shape)
	img = resize(img, (128, 39))
	print(img.shape)
	cv2.imwrite("one.jpg",img)

