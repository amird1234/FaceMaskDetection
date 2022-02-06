import argparse
import tkinter as tk
from tkinter import filedialog
from tkinter import *
import torch
from PIL import Image, ImageTk
from FaceMaskClassificationUtils import DEVICE, test_transform


class Gui:
    def __init__(self, filepath):
        checkpoint = torch.load(filepath, map_location=DEVICE)
        self.model = checkpoint['model']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

        self.current_label = None
        self.image_path_to_classify = None
        # Creates the window from the imported Tkinter module
        self.window = tk.Tk()
        # Creates the size of the window
        self.window.geometry("600x400")
        # Adds a title to the Windows GUI for the window
        self.window.title("Gender Classifier")
        title = tk.Label(self.window, foreground="black", text="Gender Classifier", font="30")
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
        i = i.resize((150, 150), Image.ANTIALIAS)
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
        classify_button.place(x=50, y=300)

    def classify(self):
        with torch.no_grad():
            img = test_transform(Image.open(self.image_path_to_classify))
            class_prediction = torch.argmax(self.model(img.unsqueeze(0))).item()
            print("this is a {}.".format(class_names_list[class_prediction]))
            gender = tk.Label(self.window, text="Gender: " + class_names_list[class_prediction], font=('calibri', 14, 'bold'))
            gender.place(x=300, y=300)
            self.current_label = gender


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-C', '--model_path', help='Dataset path', required=True)
    args = parser.parse_args()
    print(args)
    gc = Gui(args.model_path)
    gc.window.mainloop()
