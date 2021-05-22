#import packages
import pandas as pd
import cv2



#load in data
all_plates = pd.read_excel('PlateLabels.xlsx')
grouped = all_plates.groupby(all_plates.platename) 
keys = grouped.groups.keys()
font = cv2.FONT_HERSHEY_SIMPLEX

#label images
for i in keys:
    df = grouped.get_group(i)
    img = cv2.imread('undistorted_images/' + i)
    
    for x, row in df.iterrows():
        img = cv2.rectangle(img,
                            (int(df.boundingrectleft[x]),int(df.boundingrecttop[x])),
                            (int(df.boundingrectright[x]),int(df.boundingrectbottom[x])),
                            (255,0,0),2)
        img = cv2.putText(img,
                          str(df.Class[x]),
                          (int(df.boundingrectleft[x]),int(df.boundingrecttop[x]-20)),
                          font,1,(255,0,0),2)
        
    cv2.imwrite('labeled_images/' + i,img)
