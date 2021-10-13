# ORB feature based image stitching for UAV images
Image stitching is the process of stitching multiple images with overlapping areas to produce a panoramic or high resolution image. Unmanned Aerial Vehicle (UAV) can be widely used to study the terrain. Each time a UAV sweeps through an area it takes images on different sections of the regions. An image stitching software is used to stitch these images into a single image based on overlapping regions. 
This is a python script for doing the same. Here I used ORB features to identify each feature in an image and using that the images are stitched together. ORB is a pre-built function available on openCV.
The data for the program are cropped regions of a UAV image, stored in a folder. The path of the folder is given as input. Program iterate through the folder and joins image based on common feature.
# Input
![data](https://user-images.githubusercontent.com/85213549/137188913-721a9724-74d3-4cab-81c5-26c8e2401393.png)
# Output
![mossaic](https://user-images.githubusercontent.com/85213549/137188955-5c698ea6-98af-4998-b8d3-e577aa67a96e.png)
# How to run the code
1. Clone the repository in your machine
2. Install requirements.txt using pip
3. Add your data on same folder
4. Run UAV-image-stitching.py
