import numpy as np
import cv2
import tkinter as tk
import datetime
from PIL import Image
import os
from PIL import ImageTk

#Set up GUI
window = tk.Tk()  #Makes main window
window.wm_title("Digital Microscope")
window.config(background="#FFFFFF")

#Graphics window
imageFrame = tk.Frame(window, width=600, height=500)
imageFrame.grid(row=0, column=0, padx=10, pady=2)

#Capture video frames

cap = cv2.VideoCapture(0)
imageFrame.output_path = "images/*"
def show_frame():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    display1.imgtk = imgtk #Shows frame for display 1
    display1.configure(image=imgtk)
    window.after(10, show_frame)


#Slider window (slider controls stage position)
def snapshot():

    print("click111111!")


def auth():
    print("Окно авторизации пользователя")
    window1 = tk.Toplevel()
    name = "Введите имя"
    label_name = tk.Label(window1,name)
    label_name.grid(row=1, column=0, padx=10, pady=2)
    e1 = tk.Entry(window1, width=50)
    e1.grid()

    label = tk.Label(window1, text=id)
    label.grid(fill="both", padx=10, pady=10)


def take_snapshot():
    img = cv2.imread('1.jpg', 1)
    path = '/old_images/'
    cv2.imwrite(os.path.join(path, 'waka.jpg'), img)
    cv2.waitKey(0)



display1 = tk.Label(imageFrame)
display1.grid(row=1, column=0, padx=10, pady=2)  #Display 1

b1 = tk.Button(window, text="screenshot", command=snapshot)
b1.grid()
b2 = tk.Button(window, text="window", command=auth)
b2.grid()

btn = tk.Button(window, text="Snapshot!", command=take_snapshot)
btn.grid()

show_frame() #Display
window.mainloop()  #Starts GUI