# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 17:37:52 2021

@author: woutv
"""

#import packages
import pandas as pd
import cv2

weeks = ['20200706_w26','20200709_w27','20200716_w28','20200723_w29','20200730_w30','20200806_w31',
         #'20200813_w32','20200820_w33','20200827_w34','20200903_w35',
         #'20200910_w36',
         #'20200917_w37','20200924_w38','20201001_w39']
for week in weeks:

    #load in data
    all_plates = pd.read_excel(week + '/undistort-main/PlateLabels.xlsx')
    grouped = all_plates.groupby(all_plates.platename) 
    keys = grouped.groups.keys()
    font = cv2.FONT_HERSHEY_SIMPLEX
    print(week)
    #label images
    for i in keys:
        df = grouped.get_group(i)
        img = cv2.imread(week + '/undistort-main/undistorted_images/' + i)
        
        for x, row in df.iterrows():
            if int(df.boundingrectleft[x])-(150 - int(df.insect_width[x]))/2 < 0:
                left = int(0)
            elif int(df.boundingrectleft[x])-(150 - int(df.insect_width[x]))/2 > df.plate_width[x]-150:
                left = int(df.plate_width[x]-150)
            else:
                left = int(int(df.boundingrectleft[x])-(150 - int(df.insect_width[x]))/2)
            right = int(left + 150)
            if int(df.boundingrecttop[x])-(150 - int(df.insect_height[x]))/2 < 0:
                top = int(0)
            elif int(df.boundingrecttop[x])-(150 - int(df.insect_height[x]))/2 > df.plate_height[x]-150:
                top = int(df.plate_height[x]-150)
            else:
                top = int(int(df.boundingrecttop[x])-(150 - int(df.insect_height[x]))/2)       
            bottom = int(top + 150)
            crop = img[top:bottom,left:right]
            print('saving:' + 'insects/' + df.Class[x] + '/' + df.Class[x] + '_' + str(x) + '_' + i)
            cv2.imwrite('C:/Users/woutv/Documents/unief/2020-2021/projectwerk/Insectlabelling/Neural_Net/unsorted_raw/' + df.Class[x] + '/' + df.Class[x] + '_' + str(x) + '_' + i, crop)