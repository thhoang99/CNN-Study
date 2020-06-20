import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from tensorflow.keras import utils
import numpy as np
import tensorflow as tf

# loading the dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# # building the input vector from the 32x32 pixels
X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalizing the data to help with the training
X_train /= 255
X_test /= 255


# building a linear stack of layers with the sequential model
model = Sequential()

# convolutional layer
model.add(Conv2D(50, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(32, 32, 3)))

# convolutional layer
model.add(Conv2D(75, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(125, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# flatten output of conv
model.add(Flatten())

# hidden layer
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.3))
# output layer
model.add(Dense(10, activation='softmax'))

model.load_weights('D:/tf/cifar10_model_weights.h5')



# Visualize predictions.
import matplotlib.pyplot as plt
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
# Predict 5 images from validation set.
n_images = 5
np.random.shuffle(X_test)
test_images = X_test[1]
#predictions = model.predict(test_images)
from PIL import Image
from numpy import asarray
import tensorflow as tf
import requests
from io import BytesIO




import tkinter
import cv2
import PIL.Image, PIL.ImageTk
 
class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        fr=tkinter.Frame(self.window)
        fr.pack(fill="x", side="top")
        
        """
        # Load an image using OpenCV
        self.cv_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        # Get the image dimensions (OpenCV stores image data as NumPy ndarray)
        self.height, self.width, no_channels = self.cv_img.shape

         # Create a canvas that can fit the above image
        self.canvas = tkinter.Canvas(window, width = self.width, height = self.height)
        self.canvas.pack()

         # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_img))

         # Add a PhotoImage to the Canvas
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        """
        self.tbox = tkinter.Text(self.window, height=10, width=30)
        self.tbox.pack(side=tkinter.LEFT,pady =8)
        
        # Button that lets the user blur the image
        self.btn_li=tkinter.Button(fr, text="Open Local Image", width=50, command=self.local_image)
        self.btn_li.pack(anchor=tkinter.N, pady= 10)
        
                
        self.url = tkinter.Entry(fr, width=35)
        self.url.pack(anchor=tkinter.N,side = tkinter.LEFT, pady=4, padx =4)
        self.btn_url=tkinter.Button(fr, text="Open url", width=20, command=self.url_image)
        self.btn_url.pack(anchor=tkinter.N,side = tkinter.LEFT)
        
        fr=tkinter.Frame(self.window)
        fr.pack(fill="x", side="top")
        
        self.labelre = tkinter.Entry(fr, width=10)
        self.labelre.pack(anchor=tkinter.N,side = tkinter.LEFT, pady=12, padx =4)
        self.btn_ret=tkinter.Button(fr, text="Retrain", width=10, command=self.retrain)
        self.btn_ret.pack(anchor=tkinter.N,side = tkinter.LEFT,pady =8)
        
        self.window.mainloop()

    def local_image(self):  
        self.tbox.delete('1.0', tkinter.END)
        filename = tkinter.filedialog.askopenfilename()
        with Image.open(filename) as image:
            image = image.resize((32, 32), Image.ANTIALIAS)
            image = asarray(image)
            image = tf.reshape(image, [-1, 32, 32, 3])
            test_image = image / 255
            predictions = model.predict(test_image)
            self.tbox.insert(tkinter.END, "Model prediction: %s" % class_names[np.argmax(predictions)])
            
    def url_image(self):
        self.tbox.delete('1.0', tkinter.END)
        
        
        # step 1
        filenames = tf.constant(["F:/a.jpg"])
        labels = tf.constant([3])

        # step 2: create a dataset returning slices of `filenames`
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        print(dataset)
        dataset = dataset.map(self.im_file_to_tensor)
        print(dataset)
        """try:
            response = requests.get(self.url.get())
            image = Image.open(BytesIO(response.content))
            image = image.resize((32, 32), Image.ANTIALIAS)
            image = asarray(image)
            image = tf.reshape(image, [-1, 32, 32, 3])
            test_image = image / 255
            predictions = model.predict(test_image)
            self.tbox.insert(tkinter.END, "Model prediction: %s" % class_names[np.argmax(predictions)])
        except:
            self.tbox.insert(tkinter.END, "Not image or broken image") """ 


    def im_file_to_tensor(file, label):
        def _im_file_to_tensor(file, label):
            image = Image.open(file)
            image = image.resize((32, 32), Image.ANTIALIAS)
            image = asarray(image)
            image = tf.reshape(image, [-1, 32, 32, 3])
            test_image = image / 255
            return test_image, label
        return tf.py_function(_im_file_to_tensor, 
                            inp=(file, label), 
                            Tout=(tf.float32, tf.uint8))

       
    def retrain(self):
        print('test')
        

 
# Create a window and pass it to the Application object
App(tkinter.Tk(), "TEST")


