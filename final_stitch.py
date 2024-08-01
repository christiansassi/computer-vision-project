import cv2
import os
from typing import Union
import numpy as np

from src import utils
from src import stitch_image

def crop_image(image):
    height, width = image.shape[:2]
    center_x = width // 2
    center_y = height // 2
    image = image[center_y - 878:center_y + 879, center_x - 878:center_x + 879]

    return image

def rotate_and_crop (imgs):
    rotate_and_crop_images = []
    # Iterate over all the files in the folder
    for i, img in enumerate(imgs):
        if i == 2:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img = crop_image(img)
        else:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img = crop_image(img)
        rotate_and_crop_images.append(img)

    img_3 = imgs[2].copy()
    rotate_and_crop_images.append(crop_image(cv2.rotate(img_3, cv2.ROTATE_90_COUNTERCLOCKWISE)))
    
    return rotate_and_crop_images

def bb(left_frame: Union[cv2.typing.MatLike, cv2.cuda.GpuMat, cv2.UMat], right_frame: Union[cv2.typing.MatLike, cv2.cuda.GpuMat, cv2.UMat], left_min: int = None, left_max: int = None, right_min: int = None, right_max: int = None) -> tuple[np.ndarray, np.ndarray]:
    # Replace pixels outside the range with black pixels
    left_frame[:, :left_min] = 0
    left_frame[:, left_max:] = 0
    right_frame[:, :right_min] = 0
    right_frame[:, right_max:] = 0
    return left_frame, right_frame

def stitch_all(left_frame = None, right_frame = None, lf = None, rf = None, angle = (-90, 90), value = 0.99, left_kp = None, right_kp = None, left_shift_dx = 0, left_shift_dy = 0, remove_offset = 0):
    _, matches = stitch_image.stitch_images(left_frame=lf, right_frame=rf, value=value, angle=angle, method=cv2.RANSAC, user_left_kp=left_kp, user_right_kp=right_kp)
    stitched_image, _ = stitch_image.stitch_images(left_frame=left_frame, right_frame=right_frame, angle=angle, method=cv2.RANSAC, value=value, clear_cache=False, f_matches=False, left_shift_dx=left_shift_dx, left_shift_dy=left_shift_dy, remove_offset=remove_offset)
    
    return stitched_image, matches

def main():
    imgs_path = 'videos/blend'
    imgs = []

    for filename in os.listdir(imgs_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(imgs_path, filename)
            img = cv2.imread(image_path)
            imgs.append(img)
    
    cropped_images = rotate_and_crop(imgs) # [bottom, top, center_for_top, center_for_bottom]

    #! Stitch the images top and center
    left_kp = [(1724, 373), (1704, 606), (1580, 460), (1704, 1126), (1746, 1208), (1680, 1352), (1714, 1462), (1188, 1464)]
    right_kp = [(1037, 388), (966, 666), (754, 552), (1002, 1224), (1104, 1306), (1006, 1454), (1098, 1560), (402, 1570)]
    angle = (-8, -5)
    value = 0.95
    
    lf = cropped_images[2].copy()
    rf = cropped_images[1].copy()
    lf, rf = bb(left_frame=lf, right_frame=rf, left_min=950, left_max=lf.shape[1], right_min=100, right_max=1100)

    top_img, matches = stitch_all(left_frame = cropped_images[2], right_frame = cropped_images[1], lf = lf, rf = rf, 
                         angle = angle, value = value, left_kp = left_kp, right_kp = right_kp,
                         left_shift_dx = 15, left_shift_dy = 15, remove_offset = 780)
    
    utils.show_img(top_img, "Stitched Image", ratio=1.5)
    # cv2.imwrite('videos/final/top_center.jpg', top_img)

    #! Stitch the images bottom and center
    left_kp = None 
    right_kp = None
    angle = (-15, -5)
    value = 0.95
    
    lf = cropped_images[3].copy()
    rf = cropped_images[0].copy()
    lf, rf = bb(left_frame=lf, right_frame=rf, left_min=1200, left_max=lf.shape[1], right_min=100, right_max=800)
    
    bottom_img, matches = stitch_all(left_frame = cropped_images[3], right_frame = cropped_images[0], lf = lf, rf = rf, 
                        angle = angle, value = value, left_kp = left_kp, right_kp = right_kp,
                        left_shift_dx = 0, left_shift_dy = 0, remove_offset = 550)

    utils.show_img(bottom_img, "Stitched Image", ratio=1.5)
    # cv2.imwrite('videos/final/bottom_center.jpg', bottom_img)

    #! Final stitching
    left_kp = None 
    right_kp = None
    angle = (-40, 0)
    value = 0.95

    top_img = crop_image(cv2.rotate(top_img, cv2.ROTATE_180))
    bottom_img = crop_image(bottom_img)

    lf = top_img.copy()
    rf = bottom_img.copy()

    lf, rf = bb(left_frame=lf, right_frame=rf, left_min=1000, left_max=1400, right_min=350, right_max=770)
    
    final_img, matches = stitch_all(left_frame = top_img, right_frame = bottom_img, lf = lf, rf = rf, 
                        angle = angle, value = value, left_kp = left_kp, right_kp = right_kp,
                        left_shift_dx = 0, left_shift_dy = 0, remove_offset = 400)
    
    # utils.show_img(matches, "Match", ratio=1.5)
    utils.show_img(final_img, "Final Stitched Image", ratio=1.5)
    cv2.imwrite('videos/final/final_image.jpg', final_img)

if __name__ == "__main__":
    main()