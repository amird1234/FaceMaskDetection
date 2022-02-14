import argparse
import tkinter as tk
from tkinter import filedialog
from tkinter import *
import torch
from PIL import Image, ImageTk
from FinalProject import FinalProject


class Gui:
    def __init__(self, human_model_path, mask_model_path):
        self.fp = FinalProject(human_model_path, mask_model_path)
        self.current_label = None
        self.image_path_to_classify = None
        # Creates the window from the imported Tkinter module
        self.window = tk.Tk()
        # Creates the size of the window
        self.window.geometry("700x600")
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

    def upload_image(self):
        if self.current_label is not None:
            self.current_label.destroy()
            self.current_label = None
        image_path = filedialog.askopenfilename()
        i = Image.open(image_path)
        i = i.resize((300, 300), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(i)
        panel = tk.Label(self.window, image=img)
        panel.photo = img
        panel.grid(column=240, row=90)
        panel.place(x=300, y=90)

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
            class_prediction = self.fp.single_image_classify(self.image_path_to_classify)
            print("this is a {}.".format(class_prediction))
            mask_state = tk.Label(self.window, text="Mask State: " + class_prediction, font=('calibri', 14, 'bold'))
            mask_state.place(x=300, y=400)
            self.current_label = mask_state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-H', '--human_model_path', help='Human Model Path', required=True)
    parser.add_argument('-M', '--mask_model_path', help='Face Mask Model path', required=True)
    args = parser.parse_args()
    print(args)
    gc = Gui(args.human_model_path, args.mask_model_path)
    gc.window.mainloop()
