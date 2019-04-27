import glob
import cv2

filenames = [img for img in glob.glob('/home/shayan/CIFAR/Train/' + '*.png')]

filenames.sort()

print(len(filenames))