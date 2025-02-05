from google.colab import files
import zipfile
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix

files.upload()
zipfile.ZipFile('Q3_Dataset.zip' , 'r').extractall()


file_list = os.listdir('/content/Q3_Dataset')
file_list


chelsi,manchester= [],[]

for string in file_list:
  if 'c' in string.lower():
    chelsi.append(string)
  else:
    manchester.append(string)

print(f'number of manchester is {len(manchester)}'
f'\nnumber of chelsi is {len(chelsi)}')



img1 = cv2.imread('/content/Q3_Dataset/M11.jpg')
img2 = cv2.imread('/content/Q3_Dataset/c10.jpg')

img1_rgb = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

fig , ax = plt.subplots(1,2)
ax[0].imshow(img1_rgb)
ax[1].imshow(img2_rgb);



red_rgb,blue_rgb =np.array([255,0,0]) ,np.array([0,0,255])
chelsi_per,manchester_per=[],[]

for string in file_list:

  img=('/content/Q3_Dataset/'+string)

  img=cv2.imread(img)
  img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

  img_mean = np.mean(img,axis=(0,1))

  img_r = ((img_mean[0]-red_rgb[0])**2+(img_mean[1]-red_rgb[1])**2+(img_mean[2]-red_rgb[2])**2)**(1/2)
  img_b = ((img_mean[0]-blue_rgb[0])**2+(img_mean[1]-blue_rgb[1])**2+(img_mean[2]-blue_rgb[2])**2)**(1/2)

  if img_r >= img_b :
    chelsi_per.append(string)
  else:
    manchester_per.append(string)

lable =[]
for string in file_list:
  if string in chelsi:
    lable.append('chelsi')
  else:
    lable.append('manchester')

pred=[]
for string in file_list:
  if string in chelsi_per:
    pred.append('chelsi')
  else:
    pred.append('manchester')


confusion_matrix(lable,pred)

accuracy=((confusion_matrix(lable,pred)[0,0])+(confusion_matrix(lable,pred)[1,1]))/(confusion_matrix(lable,pred).sum())
accuracy

precision = confusion_matrix(lable,pred)[0,0]/(confusion_matrix(lable,pred)[0,0]+confusion_matrix(lable,pred)[0,1])
precision

recall = confusion_matrix(lable,pred)[0,0]/(confusion_matrix(lable,pred)[0,0]+confusion_matrix(lable,pred)[1,0])
recall