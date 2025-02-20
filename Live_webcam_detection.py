import cv2
import  os
import sys
import numpy as np
from PIL import Image
import pickle
import glob
from time import sleep
from unittest import mock


# This code was created by following the guide, but with some minor adjustements: 
# https://github.com/RitvikDayal/Face-Recognition-LBPH/blob/master/Face%20Recognition.ipynb



# GLOBAL VARIABLES
sys.stdout.reconfigure(encoding='utf-8')
# Haar file for detection of faces
haar_file = "haarcascade_frontalface_default.xml"
# Face detector
face_detector = cv2.CascadeClassifier(haar_file)
# LBPHF
recognizer = cv2.face.LBPHFaceRecognizer_create()
# Camera used
camera = cv2.VideoCapture(0)

def creating_pickle():
        names = []
        filename = "names.pkl"
        f = open(filename, 'wb')
        pickle.dump(names,f)
        f.close()

def preview(camera):
    """
    Preview of the camera function with cv2

    Parameters: camera used

    returns: a live feed of the camera
    """
    cv2.namedWindow("Preview")
    if camera.isOpened():
        rval, frame = camera.read()
    else:
        rval = False

    while rval:
        cv2.imshow("Preview", frame)
        rval, frame = camera.read()
        key = cv2.waitKey(20)
        if key == 27:
            break
    camera.release()
    cv2.destroyAllWindows()

#@mock.patch("builtins.input", return_value = "Aleksander")
def face_sampling(camera, max_samples):
    """
    Getting samples from live feed to train the model.
    Uses pickle to save data from camera feed

    parameters: camera used

    return live feed samples
    """
    creating_pickle()

    with open('names.pkl', 'rb') as reader:
        names = pickle.load(reader)

    print("Name for the Face: ")
    name = "Aleksander"
    names.append(name)
    id = names.index(name)
    print('''\n
    Look in the camera Face Sampling has started!.
    Try to move your face and change expression for better face memory registration.\n
    ''')
    # Counts the individual samples
    count = 0

    while(True):

        ret, img = camera.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray)

        for (x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            count += 1

            # Save the captured image into the datasets folder
            cv2.imwrite("dataset/"+name+"_ID_" + str(id) + '_' + str(count) + ".jpg", gray[y:y+h,x:x+w])

            cv2.imshow('image', img)

        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= max_samples: # Take max_samples face sample and stop video
             break

    with open('names.pkl', 'wb') as writer:
        pickle.dump(names, writer)

    # Do a bit of cleanup
    print("Your Face has been registered as {}\n\nExiting Sampling Program".format(name.upper()))
    camera.release()
    cv2.destroyAllWindows()

def face_training():
    dataset_path = "dataset"
    def getTrainImages(dataset_path):
        # A list for images
        faceSamples = []
        ids = []
        for filename in glob.glob(f'{dataset_path}/*.jpg'):
            im = Image.open(filename).convert('L')
            img_numpy = np.array(im, 'uint8')
            id = int(filename.split("_")[2])
            faces = face_detector.detectMultiScale(img_numpy)
            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)
        return faceSamples, ids
    print ("\nTraining for the faces has been started. It might take a while.\n")
    faces,ids = getTrainImages(dataset_path)
    recognizer.train(faces, np.array(ids))
    # Save the model into trainer/trainer.yml
    recognizer.write('trainer/trainer.yml') 
    # Print the numer of faces trained and end program
    print("{0} faces trained. Exiting Training Program".format(len(np.unique(ids)))) 

def faceRecognition(camera):
    print('\nStarting Recognizer....')
    
    recognizer.read('trainer/trainer.yml')

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Starting realtime video capture
    camera.set(3, 640) # set video widht
    camera.set(4, 480) # set video height

    # Define min window size to be recognized as a face
    minW = 0.1*camera.get(3)
    minH = 0.1*camera.get(4)

    with open('names.pkl', 'rb') as f:
        names = pickle.load(f)

    while True:

        ret, img =camera.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
           )

        for(x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

            # Check if confidence is less them 100 ==> "0" is perfect match 
            if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  

        cv2.imshow('camera',img) 

        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break

    # Do a bit of cleanup
    print("\nExiting Recognizer.")
    camera.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    #preview(camera)
    #----------------
    #face_sampling(camera, 10)
    #sleep(2)
    #face_training()
    #----------------
    #faceRecognition(camera)