# diff_spotter

A popular game consists of finding the differences between 2 images. In this project we develop an application to do it automatically, using computer vision tools in Python (OpenCV and Scikit-image).

## Main algorithm

The main algorithm is about finding the differences between two perfectly aligned images and displaying the result.

The steps are relatively simple:

- align the images (registration),
- do some blurring (of the typical size of a difference) to reduce the noise,
- compute a difference image,
- identify the difference areas on binary image,
- find the connected components
- display the location of the differences.