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


def write_video(folder_path, num_frames):
    import cv2

    # choose codec according to format needed
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(os.path.join(folder_path, 'output.mp4'), fourcc, 20.0, (640, 480))

    for i in range(num_frames):
        path = os.path.join(folder_path, str(i) + ".jpg")
        x = cv.imread(path)
        video.write(x)

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
        in_folder_path = os.path.join(os.path.abspath(''), "Frames")
        out_folder_path = os.path.join(os.path.abspath(''), "outFrames")

        if skip is False:
            # delete frames
            try:
                print("deleting {}".format(out_folder_path))
                shutil.rmtree(out_folder_path)
                shutil.rmtree(in_folder_path)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))

            # collect frames
            os.mkdir(out_folder_path)
            os.mkdir(in_folder_path)
            while cap.isOpened():
                ret, frame = cap.read()
                # if frame is read correctly ret is True
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                gray = frame  # cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                path = os.path.join(in_folder_path, str(i) + ".jpg")
                print(path)
                cv.imwrite(path, gray)
                i = i+1
                # hold the camera from capturing forever
                if i > 300:
                    print("stopped capturing images from comera")
                    break
            cap.release()

            im_files = [f.split(".")[0] for f in os.listdir(in_folder_path)]
            num_frames = len(im_files)
            im_files.sort(key=int)

            # classify frames and save them
            for i in range(num_frames):
                print("classifying frame #{}".format(i))
                path = os.path.join(in_folder_path, str(i) + ".jpg")
                orig_img = self.classify(path)
                out_path = os.path.join(out_folder_path, str(i) + ".jpg")
                if orig_img is not None:
                    orig_img.save(out_path)
        else:
            im_files = [f.split(".")[0] for f in os.listdir(out_folder_path)]
            num_frames = len(im_files)
            print("analyzing {} frames in out dir".format(num_frames))
            im_files.sort(key=int)

        print("num_frames {}".format(num_frames))
        # show frames
        window_name = 'frame'
        for i in range(num_frames):
            path = os.path.join(out_folder_path, str(i) + ".jpg")
            print("showing frame #{}".format(i))
            x = cv.imread(path)
            cv.imshow(window_name, x)
            if cv.waitKey(1) == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()
        # write_video(out_folder_path, num_frames)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-H', '--human_model_path', help='Human Model Path', required=True)
    parser.add_argument('-M', '--mask_model_path', help='Face Mask Model path', required=True)
    parser.add_argument('-F', '--video_path', help='Video path', required=True)
    parser.add_argument('--skip', dest='skip', action='store_true')
    parser.set_defaults(skip=False)
    args = parser.parse_args()
    print(args)
    vc = VideoClassifier(args.human_model_path, args.mask_model_path)
    vc.mainloop(args.video_path, args.skip)
