import argparse
import tkinter as tk
from tkinter import filedialog
from tkinter import *
import torch
from PIL import Image, ImageTk
from FinalProject import FinalProject
from torchvision import transforms
from FaceMaskClassificationUtils import imshow

class Gui:
    def __init__(self, human_model_path, mask_model_path):
        self.fp = FinalProject(human_model_path, mask_model_path)
        self.current_label = None
        self.image_path_to_classify = None
        # Creates the window from the imported Tkinter module
        self.window = tk.Tk()
        # Creates the size of the window
        self.window.geometry("1300x800")
        # Adds a title to the Windows GUI for the window
        self.window.title("Mask Classifier")
        title = tk.Label(self.window, foreground="black", text="Mask Classifier", font="30")
        title.place(x=240, y=20)
        upload_image_button = tk.Button(self.window,
                                        text="Upload Image",
                                        width=20,
                                        height=2,
                                        bg="light yellow",
                                        fg="black",
                                        command=self.upload_image)
        upload_image_button.place(x=50, y=70)

    def place_image(self, i, width, height, x, y):
        i = i.resize((width, height), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(i)
        panel = tk.Label(self.window, image=img)
        panel.photo = img
        panel.grid(column=240, row=90)
        panel.place(x=x, y=y)

    def upload_image(self):
        if self.current_label is not None:
            self.current_label.destroy()
            self.current_label = None
        image_path = filedialog.askopenfilename()
        i = Image.open(image_path)
        self.place_image(i, 700, 700, 300, 90)

        self.image_path_to_classify = image_path
        classify_button = tk.Button(self.window,
                                    text="Classify",
                                    width=20,
                                    height=2,
                                    bg="light yellow",
                                    fg="black",
                                    command=self.classify)
        classify_button.place(x=50, y=400)

    def classify(self):
        with torch.no_grad():
            class_predictions, orig_img = self.fp.single_image_classify(self.image_path_to_classify)
            self.place_image(orig_img, 700, 700, 300, 90)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-H', '--human_model_path', help='Human Model Path', required=True)
    parser.add_argument('-M', '--mask_model_path', help='Face Mask Model path', required=True)
    args = parser.parse_args()
    print(args)
    gc = Gui(args.human_model_path, args.mask_model_path)
    gc.window.mainloop()
