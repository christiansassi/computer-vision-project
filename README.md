# Top view stitching and tracking (tracking and geometry)

<div>
    <img src="https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54" alt="Python"/>
    <img src="https://tinyurl.com/cvyolo11" alt="Yolo 11"/>
</div>

# Table of contents

-   [Project Overview](#project-overview)
-   [Code Overview](#code-overview)
    - [Top-View Court Stitching](#top-view-court-stitching)
    - [Object Detection on Top-View Images](#object-detection-on-top-view-images)
    - [Object Tracking](#object-tracking)
    - [Ball Detection and Tracking](#ball-detection-and-tracking)
-   [Getting Started](#getting-started)
-   [Contacts](#contacts)

# Project Overview
This project focuses on processing video camera images from the Sanbapolis facility in Trento. The objectives are:

- **Top-View Court Stitching**: the facility has three distinct views—top, center, and bottom—each captured by four cameras. The goal is to first stitch the images from cameras within the same view. After reconstructing each view, the next step is to stitch the top, center, and bottom views together to create a seamless top-view of the entire court.
- **Object Detection on Top-View Images**: we applied several detection algorithms to the stitched top-view images, testing various techniques from our coursework, including frame subtraction, background subtraction, adaptive background subtraction, and Gaussian averaging. After evaluation, we selected background subtraction as the most effective method for our purpose.
- **Object Tracking**: for tracking detected objects (bounding boxes), we implemented particle filtering, one of the techniques studied during the course. Since it performed well, we opted not to explore additional methods further.
- **Ball Detection and Tracking**: for ball detection and tracking, we used the YOLO (You Only Look Once) algorithm, which proved suitable for this task.

# Code Overview

## Top-View Court Stitching

## Object Detection on Top-View Images
Several detection algorithms were applied to the stitched top-view images, testing various techniques from coursework, including frame subtraction, background subtraction, adaptive background subtraction, and Gaussian averaging. After evaluation, background subtraction was selected as the most effective method.

The first step involves applying a threshold to the image to extract the most relevant areas. During this phase, dilation is applied to account for stitching errors that sometimes cause players to be incorrectly displayed as separate objects. The dilation helps merge these separated segments into a single object. Additionally, small areas are discarded:

<p align="center" text-align="center"> 
    <img width="75%" src="assets/motion_detection/motion_detection_1.png"> 
    <br> 
    <span><i>Thresholded image</i></span> 
</p>

Next, contours are filtered based on the volleyball court area. The court's boundaries are defined, and objects that intercept this area by 25% or more are retained. This approach helps discard irrelevant objects, such as people outside the court (e.g., coaches) who may briefly step into the frame:

<p align="center" text-align="center"> 
    <img width="75%" src="assets/motion_detection/motion_detection_2.png"> 
    <br> 
    <span><i>Volleyball field mask</i></span> 
</p>

By combining these two techniques, the following result was achieved:

<p align="center" text-align="center"> 
    <img width="75%" src="assets/motion_detection/motion_detection_3.png"> 
    <br> 
    <span><i>Motion detection</i></span> 
</p>

## Object Tracking

For tracking detected objects (bounding boxes), particle filtering was implemented, a technique studied during the course. As this method performed well, further exploration of additional techniques was deemed unnecessary.

For each detected bounding box, a new particle system was initialized. Initially, the particles in each system exhibited chaotic behavior due to the randomness at the start:

<p align="center" text-align="center"> 
    <img width="75%" src="assets/motion_tracking/motion_tracking_1.png"> 
    <br> 
    <span><i>Initial particle system</i></span> 
</p>

At each iteration, the particle systems were compared with the updated bounding boxes to determine if a particle system still had an associated bounding box (i.e., the object is still detected) or if a new system was required (i.e., the object is no longer detected, or a new object has appeared).

To associate a particle system with its corresponding bounding box, the distance between the centroid of the particle system and the bounding box was evaluated. A particle system was associated with a bounding box if it had the smallest distance to that bounding box. Otherwise, if no suitable particle system was found, a new one was created.

Through repeated iterations, the randomness within each particle system diminished:

<p align="center" text-align="center"> 
    <img width="75%" src="assets/motion_tracking/motion_tracking_2.png"> 
    <br> 
    <span><i>Particle system after some iterations</i></span> 
</p>

Finally, the particle systems were used to predict the possible direction of a moving object. It is important to note that for small movements, the direction arrow may appear slow and less certain. Additionally, if an object makes a sudden, fast movement, the particle system may require a few iterations to adapt, potentially resulting in incorrect predictions during those iterations.

<p align="center" text-align="center"> 
    <img width="75%" src="assets/motion_tracking/motion_tracking_3.png"> 
    <br> 
    <span><i>Motion tracking</i></span> 
</p>

## Ball Detection and Tracking

For ball detection and tracking, the YOLO (You Only Look Once) algorithm was employed, as it proved well-suited for this task. Due to the ball’s high velocity, it often appeared distorted in some frames, making it difficult to detect using traditional techniques.

The first step involved creating a dataset specifically for this task. Approximately 1,000 images were manually extracted from the videos, selecting both the players and the ball (when present). YOLO v11 was then applied to this dataset, producing the following result:

# Getting Started

1. Set up the workspace:

    ```bash
    git clone https://github.com/christiansassi/computer-vision-project
    cd computer-vision-project
    pip install -r requirements.txt
    ```

2. Run [main.py](main.py) script:

    ```bash
    python3 main.py
    ```

> [!WARNING]
> Due to privacy reasons, the video files cannot be shared.

<p align="center" text-align="center">
  <img width="75%" src="assets/demo/demo.gif">
  <br>
  <span><i>Demo</i></span>
</p>

# Contacts

Pietro Bologna - [pietro.bologna@studenti.unitn.it](mailto:pietro.bologna@studenti.unitn.it)

Christian Sassi - [christian.sassi@studenti.unitn.it](mailto:christian.sassi@studenti.unitn.it)

<picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/extras/dark.png">
    <img alt="https://www.unitn.it/" src="assets/extras/light.png" width="300px">
</picture>
