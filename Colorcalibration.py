#%% packages
import cv2
import numpy as np
from sklearn.cross_decomposition import PLSRegression
import pandas as pd
import colour
import glob

#%% load color calibration image 
img = cv2.imread('undistorted_images/UNDISTORTED_calibration_color_w33_4056x3040_0.jpg')

#%% make mask
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
low_treshold = (0,90,150)
Hi_treshold = (200,255,255)

mask = cv2.inRange(hsv_img, low_treshold,Hi_treshold)
result = cv2.bitwise_and(hsv_img, hsv_img, mask=mask)


#%% remove noise in mask
ret, thresh = cv2.threshold(mask,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
kernel = np.ones((3,3),np.uint8)
dilated_mask = cv2.dilate(mask,kernel,iterations=3)
mask = cv2.erode(dilated_mask,kernel,iterations=3)
cv2.imwrite('mask.png',mask)

#%% find contours of mask
im = cv2.imread('mask.png') #I have to load the mask in again otherise I get an error
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
image, contours, hierarchy= cv2.findContours(imgray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# remove small unwanted contours
contours = [x for x in contours if x.size > 20]

final = np.zeros(img.shape,np.uint8)
mask = np.zeros(imgray.shape,np.uint8)



#%% find mean values of each color
red_colors = []
yellow_colors = []
green_colors = []
blue_colors = []
cyan_colors = []
magenta_colors = []
extracted_colors = []

#%%% red
for i in (0,2,4,34,32,30):
    mask[...]=0
    cv2.drawContours(mask,contours,i,255,-1)
    cv2.drawContours(final,contours,i,cv2.mean(img,mask),-1)
    mean = cv2.mean(final,mask = mask)
    mean = mean[:3]
    red_colors.append(mean)
extracted_colors.append(np.mean(red_colors, axis = 0))

#%%% yellow
for i in (1,3,5,35,33,31):
    mask[...]=0
    cv2.drawContours(mask,contours,i,255,-1)
    cv2.drawContours(final,contours,i,cv2.mean(img,mask),-1)
    mean = cv2.mean(final,mask = mask)
    mean = mean[:3]
    yellow_colors.append(mean)
extracted_colors.append(np.mean(yellow_colors, axis = 0))

#%%% green
for i in (6,8,10,25,27,24):
    mask[...]=0
    cv2.drawContours(mask,contours,i,255,-1)
    cv2.drawContours(final,contours,i,cv2.mean(img,mask),-1)
    mean = cv2.mean(final,mask = mask)
    mean = mean[:3]
    green_colors.append(mean)
extracted_colors.append(np.mean(green_colors, axis = 0))

#%%% blue
for i in (7,9,11,29,26,28):
    mask[...]=0
    cv2.drawContours(mask,contours,i,255,-1)
    cv2.drawContours(final,contours,i,cv2.mean(img,mask),-1)
    mean = cv2.mean(final,mask = mask)
    mean = mean[:3]
    blue_colors.append(mean)
extracted_colors.append(np.mean(blue_colors, axis = 0))

#%%% cyan
for i in (12,14,16,19,21,23):
    mask[...]=0
    cv2.drawContours(mask,contours,i,255,-1)
    cv2.drawContours(final,contours,i,cv2.mean(img,mask),-1)
    mean = cv2.mean(final,mask = mask)
    mean = mean[:3]
    cyan_colors.append(mean)
extracted_colors.append(np.mean(cyan_colors, axis = 0))

#%%% magenta
for i in (13,15,17,18,20,22):
    mask[...]=0
    cv2.drawContours(mask,contours,i,255,-1)
    cv2.drawContours(final,contours,i,cv2.mean(img,mask),-1)
    mean = cv2.mean(final,mask = mask)
    mean = mean[:3]
    magenta_colors.append(mean)
extracted_colors.append(np.mean(magenta_colors, axis = 0))

#%% reference colors from Lab values
reference_colors = [[75.072,75.443,187.330],[47.579,207.441,230.770],[72.677,175.494,125.194],[152.752,99.458,21.251],[186.418,183.985,63.106],[135.170,67.013,156.345]]

#%% match colors
pls=PLSRegression(n_components=3)
pls.fit(extracted_colors,reference_colors)
pls.score(extracted_colors,reference_colors)


#%% load names for correcting the images
keys = glob.glob('undistorted_images/*.jpg')

#%% calibrate images

for i in keys:
    
    img = cv2.imread(i)
    calibrated_img = morph = img.copy()
    for im in calibrated_img:
        #[:]=pls.predict(im[:])
        im[:]=colour.colour_correction(im[:],extracted_colors,reference_colors,'Finlayson 2015')
    i = i.split('\\')[-1]
    cv2.imwrite('Colorcalib_images/' + i,calibrated_img)