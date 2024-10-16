# Top view stitching and tracking (tracking and geometry)

<div>
    <img src="https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54" alt="Python"/>
    <img src="https://img.shields.io/badge/yolo-11-00FFFF" alt="Yolo 11"/>
</div>

# Table of contents

-   [Project Overview](#project-overview)
-   [Code Overview](#code-overview)
-   [Getting Started](#getting-started)
-   [Contacts](#contacts)

# Project Overview
This project focuses on processing video camera images from the Sanbapolis facility in Trento. The objectives are:

- **Top-View Court Stitching**: the facility has three distinct views—top, center, and bottom—each captured by four cameras. The goal is to first stitch the images from cameras within the same view. After reconstructing each view, the next step is to stitch the top, center, and bottom views together to create a seamless top-view of the entire court.
- **Object Detection on Top-View Images**: we applied several detection algorithms to the stitched top-view images, testing various techniques from our coursework, including frame subtraction, background subtraction, adaptive background subtraction, and Gaussian averaging. After evaluation, we selected background subtraction as the most effective method for our purpose.
- **Object Tracking**: for tracking detected objects (bounding boxes), we implemented particle filtering, one of the techniques studied during the course. Since it performed well, we opted not to explore additional methods further.
- **Ball Detection and Tracking**: for ball detection and tracking, we used the YOLO (You Only Look Once) algorithm, which proved suitable for this task.

# Code Overview

# Getting Started

1. Set up the workspace:

    ```bash
    git clone https://github.com/christiansassi/computer-vision-project
    cd computer-vision-project
    pip install -r requirements.txt
    ```

2. To execute the project, simply run the [main.py](main.py) script:

    ```bash
    python3 main.py
    ```

> [!WARNING]
> Due to privacy reasons, the video files cannot be shared.

# Contacts

Pietro Bologna - [pietro.bologna@studenti.unitn.it](mailto:pietro.bologna@studenti.unitn.it)

Christian Sassi - [christian.sassi@studenti.unitn.it](mailto:christian.sassi@studenti.unitn.it)

<picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/extras/dark.png">
    <img alt="https://www.unitn.it/" src="assets/extras/light.png" width="300px">
</picture>
