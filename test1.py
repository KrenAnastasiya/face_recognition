import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
from keras import backend as K
import datetime
import time
from multiprocessing.dummy import Pool
K.set_image_data_format('channels_first')
import cv2
import os
import glob
import numpy as np
from numpy import genfromtxt
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
# import win32com.client as wincl
from tkinter import *
from tkinter import messagebox


PADDING = 50
ready_to_detect_identity = True
detect_frame_id = 0
# windows10_voice_interface = wincl.Dispatch("SAPI.SpVoice")

FRmodel = faceRecoModel(input_shape=(3, 96, 96))


def prepare_database():
    for i in os.walk('images/'):
        for j in i[2]:
            print(i[0] + '/' + j)
    # load all the images of individuals to recognize into the database
    database = {}
    for i in os.walk('images/'):
        for j in i[2]:
            file = i[0] + '/' + j
            folder = i[0]
            folder = folder.split("/")
            # identity = i[1]
            identity = folder[1]
            database[identity] = img_path_to_encoding(file, FRmodel)
    print("database prepare")
    return database


class App:


    def __init__(self, window, window_title, video_source=0):

        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()
        # Button that lets the user take a snapshot
        self.btn_snapshot=tkinter.Button(window, text="Snapshot", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)
        self.btn_auth = tkinter.Button(window, text="auth", width=50, command=self.auth)
        self.btn_auth.pack(anchor=tkinter.CENTER, expand=True)
        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()
        self.window.mainloop()

    def show_message():
        messagebox.showinfo("GUI Python", self.message.get())

    def create_window():
        window = tk.Toplevel(root)



    def auth(self):
        def check():
            s = E.get()
            #print(s)
            newpath = '/home/data-scientist/projects/test/facenet-face-recognition/images/'+s
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            #window_save = tkinter.Tk()
            #L1 = tkinter.Label(window_save, text = newpath)
            #L1.pack()
            ret, frame = self.vid.get_frame()
            # cv2.namedWindow('image')

            if ret:
                now = datetime.datetime.now()
                part_image = frame[100:376, 200:476]
                path = newpath+"/"+str(now)+"test.jpg"
                cv2.imwrite(path, part_image)
                img = cv2.imread(path, 0)
                cv2.imshow('image', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            return newpath
        root = tkinter.Tk()
        L =tkinter.Label(root, text="Введите имя пользователя")
        L.pack()
        E = tkinter.Entry(root)
        E.pack()
        b = tkinter.Button(root, text="Create new img", command=check)
        b.pack()

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
    	#cv2.namedWindow('image')
        if ret:
            cv2.imwrite("test.jpg", frame)
            img = cv2.imread('test.jpg', 0)
            cv2.imshow('image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
        self.window.after(self.delay, self.update)
class MyVideoCapture:
    def process_frame(img, frame, face_cascade):
        global detect_frame_id
        """
        Determine whether the current frame contains the faces of people from our database
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        fontColor = (255, 255, 255)
        lineType = 2

        global ready_to_detect_identity
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Loop through all the faces detected and determine whether or not they are in the database
        identities = []
        for (x, y, w, h) in faces:
            x1 = x - PADDING
            y1 = y - PADDING
            x2 = x + w + PADDING
            y2 = y + h + PADDING

            # img = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            identity = MyVideoCapture.find_identity(frame, x1, y1, x2, y2)
            if identity is not None:
                cv2.putText(img, identity, (200, 100), font, fontScale, fontColor, lineType)
                identities.append(identity)

        if identities != []:
            frame = cv2.rectangle(frame, (200, 100), (476, 376), (0, 255, 0), 2)
            file_name = 'example' + str(detect_frame_id) + '.png'
            cv2.imwrite(file_name, img)
            cv2.imshow('detect', img)
            detect_frame_id = detect_frame_id + 1

            ready_to_detect_identity = False
            pool = Pool(processes=1)
            # We run this as a separate process so that the camera feedback does not freeze
            pool.apply_async(MyVideoCapture.welcome_users, [identities])
            # welcome_users(identities)
        return img

    def find_identity(frame, x1, y1, x2, y2):
        """
        Determine whether the face contained within the bounding box exists in our database

        x1,y1_____________
        |                 |
        |                 |
        |_________________x2,y2

        """
        height, width, channels = frame.shape
        # The padding is necessary since the OpenCV face detector creates the bounding box around the face and not the head
        part_image = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
        cv2.imwrite('face.png', part_image)

        return MyVideoCapture.who_is_it(part_image, database, FRmodel)

    def who_is_it(image, database, model):
        """
        Implements face recognition for the happy house by finding who is the person on the image_path image.

        Arguments:
        image_path -- path to an image
        database -- database containing image encodings along with the name of the person on the image
        model -- your Inception model instance in Keras

        Returns:
        min_dist -- the minimum distance between image_path encoding and the encodings from the database
        identity -- string, the name prediction for the person on image_path
        """
        encoding = img_to_encoding(image, model)

        min_dist = 100
        identity = None

        # Loop over the database dictionary's names and encodings.
        for (name, db_enc) in database.items():

            # Compute L2 distance between the target "encoding" and the current "emb" from the database.
            dist = np.linalg.norm(db_enc - encoding)

            # print('distance for %s is %s' %(name, dist))

            # If this distance is less than the min_dist, then set min_dist to dist, and identity to name
            if dist < min_dist:
                min_dist = dist
                identity = name

        if min_dist > 0.45:
            return None
        else:
            return str(identity)

    def welcome_users(identities):
        """ Outputs a welcome audio message to the users """
        global ready_to_detect_identity
        welcome_message = 'Welcome '

        if len(identities) == 1:
            welcome_message += '%s, have a nice day.' % identities[0]
        else:
            for identity_id in range(len(identities) - 1):
                welcome_message += '%s, ' % identities[identity_id]
            welcome_message += 'and %s, ' % identities[-1]
            welcome_message += 'have a nice day!'

        print(welcome_message)
        # windows10_voice_interface.Speak(welcome_message)

        # Allow the program to start detecting identities again
        ready_to_detect_identity = True
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)

        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)


    def get_frame(self):
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        global ready_to_detect_identity
        if self.vid.isOpened():
            ret, frame = self.vid.read()

            frame = cv2.rectangle(frame, (200, 100), (476, 376), (255, 0, 0), 2)
            img = frame
            if ready_to_detect_identity:
                cv2.imwrite("test1.jpg", img)
                img = MyVideoCapture.process_frame(img, frame, face_cascade)

            if ret:
                # Return a boolean success flag and the current frame converted to BGR
               return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
               return (ret, None)
        else:
           return (frame, None)



    # Release the video source when the object is destroyed

    def webcam_face_recognizer(database):
        """
        Runs a loop that extracts images from the computer's webcam and determines whether or not
        it contains the face of a person in our database.

        If it contains a face, an audio message will be played welcoming the user.
        If not, the program will process the next frame from the webcam
        """
        global ready_to_detect_identity

        cv2.namedWindow("preview")

        vc = cv2.VideoCapture(0)

        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        while vc.isOpened():
            _, frame = vc.read()
            frame = cv2.rectangle(frame, (200, 100), (476, 376), (255, 0, 0), 2)

            img = frame
            # We do not want to detect a new identity while the program is in the process of identifying another person
            if ready_to_detect_identity:
                img = database.process_frame(img, frame, face_cascade)

            key = cv2.waitKey(100)
            cv2.imshow("preview", img)

            if key == 27:  # exit on ESC
                break
        cv2.destroyWindow("preview")



    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
# Create a window and pass it to the Application object
database = prepare_database()
App(tkinter.Tk(), "Face detector")