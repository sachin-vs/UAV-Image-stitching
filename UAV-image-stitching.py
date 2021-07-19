# -*- coding: utf-8 -*-
#Import library
#import libraries
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
def warpImages(img1, img2, H):
  rows1, cols1 = img1.shape[:2]
  rows2, cols2 = img2.shape[:2]

  list_of_points_1 = np.float32([[0,0], [0, rows1],[cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2) #coordinates of a reference image
  temp_points = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2) #coordinates of second image

  # When we have established a homography we need to warp perspective
  # Change field of view
  list_of_points_2 = cv2.perspectiveTransform(temp_points, H)#calculate the transformation matrix

  list_of_points = np.concatenate((list_of_points_1,list_of_points_2), axis=0)

  [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
  [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
  
  translation_dist = [-x_min,-y_min]
  
  H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

  output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max-x_min, y_max-y_min))
  output_img[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = img1

  return output_img
#folfer containing images from drones, sorted by name 
import glob
path = sorted(glob.glob("*.jpg"))
img_list = []
for img in path:
    n = cv2.imread(img)
    img_list.append(n)
"""Functions for stitching"""

#Use ORB detector to extract keypoints
orb = cv2.ORB_create(nfeatures=2000)
while True:
  img1=img_list.pop(0)
  img2=img_list.pop(0)
# Find the key points and descriptors with ORB
  keypoints1, descriptors1 = orb.detectAndCompute(img1, None)#descriptors are arrays of numbers that define the keypoints
  keypoints2, descriptors2 = orb.detectAndCompute(img2, None)


# Create a BFMatcher object to match descriptors
# It will find all of the matching keypoints on two images
  bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)#NORM_HAMMING specifies the distance as a measurement of similarity between two descriptors

# Find matching points
  matches = bf.knnMatch(descriptors1, descriptors2,k=2)

  all_matches = []
  for m, n in matches:
    all_matches.append(m)
# Finding the best matches
  good = []
  for m, n in matches:
    if m.distance < 0.6 * n.distance:#Threshold
        good.append(m)

# Set minimum match condition
  MIN_MATCH_COUNT = 5

  if len(good) > MIN_MATCH_COUNT:
    
    # Convert keypoints to an argument for findHomography
    src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    # Establish a homography
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    
    result = warpImages(img2, img1, M)
    
    img_list.insert(0,result)
    
    if len(img_list)==1:
      break
result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB )  
plt.imshow(result)
plt.show()

