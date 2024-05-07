import cv2 
from src import utils
import numpy as np
from typing import Union

def blend_image(mat: Union[cv2.typing.MatLike, cv2.cuda.GpuMat, cv2.UMat, list[cv2.typing.MatLike], list[cv2.cuda.GpuMat], list[cv2.UMat]], intersection: int, offset: int = 20, intensity: int = 2) -> np.ndarray:
    image1 = mat
    image2 = mat
    
    rec_image1 = image1.copy()[:,:intersection-offset]
    image1 = image1[:, intersection-offset:intersection]

    rec_image2 = image2.copy()[:,intersection+offset:]
    image2 = image2[:, intersection:intersection+offset]

    # Generate Gaussian pyramid for image1
    image1_copy = image1.copy()
    gp_image1 = [image1_copy]
    for i in range(intensity):
        image1_copy = cv2.pyrDown(image1_copy)
        gp_image1.append(image1_copy)

    # Generate Gaussian pyramid for image2
    image2_copy = image2.copy()
    gp_image2 = [image2_copy]
    for i in range(intensity):
        image2_copy = cv2.pyrDown(image2_copy)
        gp_image2.append(image2_copy)

    # Generate Laplacian Pyramid for image1
    image1_copy = gp_image1[intensity-1]
    lp_image1 = [image1_copy]
    for i in range(intensity-1, 0, -1):
        gaussian_expanded = cv2.pyrUp(gp_image1[i])
        gaussian_expanded = cv2.resize(gaussian_expanded, (gp_image1[i-1].shape[1], gp_image1[i-1].shape[0]))
        laplacian = cv2.subtract(gp_image1[i-1], gaussian_expanded)
        lp_image1.append(laplacian)

    # Generate Laplacian Pyramid for image2
    image2_copy = gp_image2[intensity-1]
    lp_image2 = [image2_copy]
    for i in range(intensity-1, 0, -1):
        gaussian_expanded = cv2.pyrUp(gp_image2[i])
        gaussian_expanded = cv2.resize(gaussian_expanded, (gp_image2[i-1].shape[1], gp_image2[i-1].shape[0]))
        laplacian = cv2.subtract(gp_image2[i-1], gaussian_expanded)
        lp_image2.append(laplacian)

    # Now add left and right halves of images in each level
    image1_image2_pyramid = []
    n = 0
    for image1_lap, image2_lap in zip(lp_image1, lp_image2):
        n += 1
        rows, cols, ch = image1_lap.shape
        laplacian = np.hstack((image1_lap, image2_lap))
        image1_image2_pyramid.append(laplacian)

    # now reconstruct
    image1_image2_reconstruct = image1_image2_pyramid[0]
    for i in range(1, intensity):
        image1_image2_reconstruct = cv2.pyrUp(image1_image2_reconstruct)
        image1_image2_reconstruct = cv2.resize(image1_image2_reconstruct, (image1_image2_pyramid[i].shape[1], image1_image2_pyramid[i].shape[0]))   
        image1_image2_reconstruct = cv2.add(image1_image2_pyramid[i], image1_image2_reconstruct)

    return np.hstack((rec_image1, image1_image2_reconstruct, rec_image2))