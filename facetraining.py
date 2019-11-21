''''
Training Multiple Faces stored on a DataBase:
	==> Each face should have a unique numeric integer ID as 1, 2, 3, etc                       
	==> LBPH computed model will be saved on trainer/ directory. (if it does not exist, pls create one)
	==> for using PIL, install pillow library with "pip install pillow"

Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition    

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18   

'''

##import cv2
##import numpy as np
##from PIL import Image
##import os
##
### Path for face image database
##path = r"C:\Users\Admin\Downloads\User\1"
##
##recognizer = cv2.face.LBPHFaceRecognizer_create()
##detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
##
### function to get the images and label data
##def getImagesAndLabels(path):
##
##    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
##    faceSamples=[]
##    ids = []
##
##    for imagePath in imagePaths:
##
##        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
##        img_numpy = np.array(PIL_img,'uint8')
##
##        id = int(os.path.split(imagePath)[-1].split(".")[1])
##        faces = detector.detectMultiScale(img_numpy)
##
##        for (x,y,w,h) in faces:
##            faceSamples.append(img_numpy[y:y+h,x:x+w])
##            ids.append(id)
##
##    return faceSamples,ids
##
##print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
##faces,ids = getImagesAndLabels(path)
##recognizer.train(faces, np.array(ids))
##
### Save the model into trainer/trainer.yml
##recognizer.write(trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi
##
### Print the numer of faces trained and end program
##print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

import cv2,os
import numpy as np
from PIL import Image

#Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.face.LBPHFaceRecognizer_create()
#Using prebuilt frontal face training model, for face detection
detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
path = r"C:\Users\Admin\Downloads\User\1."

#Create method to get the images and label data
def getImagesAndLabels(path):
    # Get all file path
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)] 

    # Initialize empty face sample
    faceSamples=[]

    # Initialize empty id
    ids = []

    # Loop all the file path
    for imagePath in imagePaths:

        # Get the image and convert it to grayscale
        PIL_img = Image.open(imagePath).convert('L')

        # PIL image to numpy array
        img_numpy = np.array(PIL_img,'uint8')

        # Get the image id
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        print(id)

        # Get the face from the training images
        faces = detector.detectMultiScale(img_numpy)

        # Loop for each face, append to their respective ID
        for (x,y,w,h) in faces:

            # Add the image to face samples
            faceSamples.append(img_numpy[y:y+h,x:x+w])

            # Add the ID to IDs
            ids.append(id)

    # Pass the face array and IDs array
    return faceSamples,ids


print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))
recognizer.save(r'C:\Users\Admin\Downloads\User\1\trainer.yml')
#recognizer.save('trainner/trainner.yml')
#recognizer.write('trainner/trainner.yml')
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.array(ids))))
