import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils 

def display(image,caption = ''):
    plt.figure(figsize = (5,10))
    plt.title(caption)
    plt.imshow(image)
    plt.show()

def extract_film(image):
  gray = cv2.cvtColor(image.copy(),cv2.COLOR_BGR2GRAY)
  gray = cv2.equalizeHist(gray)
  thresh = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)[1]
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
  morph = cv2.erode(thresh,kernel, iterations = 2)

  cnts = cv2.findContours(morph.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)
  cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

  temp = image.copy()
  croped = image.copy()
  h,w,c = croped.shape
  croped[:,:] = 0
  x0 = 0
  y0 = 0
  h0 = 0
  w0 = 0
  
  for c in cnts:
          (x,y,w,h) = cv2.boundingRect(c)      

          if w>8000 and h>8000:
              cv2.rectangle(temp,(x,y),(x+w,y+h),(0,255,0),15)
              croped = image[y:y+h,x:x+w]
              x0 = x
              y0 = y
              h0 = h
              w0 = w
            
  values = [x0,y0,h0,w0]
  return croped, values

def refine_film(image, threshold_value):
  gray_crop = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  gray_crop = cv2.equalizeHist(gray_crop)

  thresh_crop = cv2.threshold(gray_crop,threshold_value, 255,cv2.THRESH_BINARY)[1]

  crop = thresh_crop.copy()
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
  eroded = cv2.erode(crop,kernel,1)

  w,h = eroded.shape
  eroded[int(h*0.06):int(h*0.94),int(w*0.06):int(w*0.94)] = 255

  cnts = cv2.findContours(eroded, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)
  cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

  temp = eroded.copy()
  width,height = eroded.shape
  x1,y1, w1, h1 = cv2.boundingRect(cnts[0])

  i=0
  for c in cnts:
    if i == 0:
      i = i+1
      continue
    (x,y,w,h) = cv2.boundingRect(c)
    if (w < 500 or height < 500) and (abs(x-x1) < 700 or abs(w1-x) < 700):
      eroded[y:y+h,x:x+w] = 0

  kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50,50))
  dilated = cv2.dilate(eroded,kernel1,1)
  
  return dilated


if __name__ == '__main__':
  threshold_value = 3 # Use value 3 or 10
  path = '/content/drive/MyDrive/Scene/'
  img = cv2.imread(path+'film4.jpg')
  image = img.copy()
  display(image, 'Original Image')

  croped, values = extract_film(image)
  display(croped,'Film Extracted')

  refined = refine_film(croped, threshold_value)
  display(refined, 'Refined Mask')
  final_mask = img.copy()
  final_mask = cv2.cvtColor(final_mask, cv2.COLOR_BGR2GRAY)
  final_mask[:,:] = 0
  final_mask[values[1]:values[1]+values[2],values[0]:values[0]+values[3]] = refined
  display(final_mask, 'Film mask')




