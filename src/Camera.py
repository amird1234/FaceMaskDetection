import time
import os
import cv2 as cv
import argparse
import shutil
import glob

from tkinter import *
import torch
from PIL import Image, ImageTk
from FinalProject import FinalProject
from FaceMaskClassificationUtils import imshow
import numpy as np


import typing
from typing import Union
ImageType = typing.Union[np.ndarray, Image.Image]


def pil2cv(pil_image) -> np.ndarray:
    """ Convert from pillow image to opencv """
    # convert PIL to OpenCV
    pil_image = pil_image.convert('RGB')
    cv2_image = np.array(pil_image)
    # Convert RGB to BGR
    cv2_image = cv2_image[:, :, ::-1].copy()
    return cv2_image


def ispil(im):
    return isinstance(im, Image.Image)


def color_bgr2gray(image: ImageType):
    """ change color image to gray
    Returns:
        opencv-image
    """
    if ispil(image):
        image = pil2cv(image)

    if len(image.shape) == 2:
        return image
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def write_video(folder_path):
    import cv2

    # choose codec according to format needed
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('video.avi', fourcc, 1, (640, 480))

    for j in range(0,5):
        img = cv2.imread(str(i) + '.png')
        video.write(img)

    cv2.destroyAllWindows()
    video.release()

class VideoClassifier:
    def __init__(self, human_model_path, mask_model_path):
        self.fp = FinalProject(human_model_path, mask_model_path)

    def classify(self, image_path):
        with torch.no_grad():
            class_predictions, orig_img = self.fp.single_image_classify(image_path)
            if class_predictions is None or orig_img is None:
                return
            return orig_img

    def mainloop(self, video_path, skip):
        cap = cv.VideoCapture(video_path)
        i = 0
        folder_path = os.path.join(os.path.abspath(''), "Frames")

        if skip is False:
            # delete frames
            try:
                print("deleting {}".format(folder_path))
                shutil.rmtree(folder_path)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))

            # collect frames
            os.mkdir(folder_path)
            while cap.isOpened():
                ret, frame = cap.read()
                # if frame is read correctly ret is True
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                gray = frame  # cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                path = os.path.join(folder_path, str(i) + ".jpg")
                cv.imwrite(path, gray)
                i = i+1
                # hold the camera from capturing forever
                if i > 300:
                    break
            cap.release()
            print(i)

            im_files = [f.split(".")[0] for f in os.listdir(folder_path)]
            num_frames = len(im_files)
            im_files.sort(key=int)

            # classify frames and save them
            for i in range(num_frames):
                print(i)
                path = os.path.join(folder_path, str(i) + ".jpg")
                orig_img = self.classify(path)
                if orig_img is not None:
                    orig_img.save(path)
        else:
            im_files = [f.split(".")[0] for f in os.listdir(folder_path)]
            num_frames = len(im_files)
            im_files.sort(key=int)

        # show frames
        window_name = 'frame'
        for i in range(num_frames):
            path = os.path.join(folder_path, str(i) + ".jpg")
            x = cv.imread(path)
            cv.imshow(window_name, x)
            if cv.waitKey(1) == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-H', '--human_model_path', help='Human Model Path', required=True)
    parser.add_argument('-M', '--mask_model_path', help='Face Mask Model path', required=True)
    parser.add_argument('-F', '--video_path', help='Video path', required=True)
    parser.add_argument('--skip_capturing', dest='skip', action='store_true')
    parser.set_defaults(skip=False)
    args = parser.parse_args()
    print(args)
    vc = VideoClassifier(args.human_model_path, args.mask_model_path)
    vc.mainloop(args.video_path, args.skip_capturing)
