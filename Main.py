from lib2to3.pgen2.token import NAME
import speech_recognition as sr
import pyttsx3
from time import sleep
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import shutil

###################################################################
# Add New User
###################################################################
def Add_usr(l):
    l = str(l)
    name = "name.jpg"
    vid = cv2.VideoCapture(0)
    ret, frame = vid.read()
    cv2.imwrite(name,frame)
    os.rename("name.jpg",str(l)+".jpg")
    try:
        shutil.move(str(l)+".jpg","images")
        sound = "Thank you added new user",str(l)
        Speak(sound)
    except:
        sound = "username already exits in database"
        os.remove(str(l)+".jpg")
        Speak(sound)

    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()




def Add():
    Speak("what is Your Name?")
    r = sr.Recognizer()
    try:
            with sr.Microphone() as source2:
                r.adjust_for_ambient_noise(source2, duration=0.2)
                audio2 = r.listen(source2)
                MyText = r.recognize_google(audio2)
                MyText = MyText.lower()
                print("Did you say "+MyText)
                Add_usr(MyText)
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
    except sr.UnknownValueError:
        print("unknown error occured")




###################################################################################
# Face Encoding
###################################################################################
def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

########################################################################################
# Images Matching 
########################################################################################
def matching():
    path = 'images'
    images = []
    personNames = []
    myList = os.listdir(path)
    for cu_img in myList:
        current_Img = cv2.imread(f'{path}/{cu_img}')
        images.append(current_Img)
        personNames.append(os.path.splitext(cu_img)[0])

    encodeListKnown = faceEncodings(images)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        faces = cv2.resize(frame,(0,0), None, fx=0.25, fy=0.25)

        faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

        facesCurrentFrame = face_recognition.face_locations(faces)
        encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)

        for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            # print(faceDis)
            matchIndex = np.argmin(faceDis)
            if matches[matchIndex]:
                name = personNames[matchIndex].lower()
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                sound = "i know you are"+name
                Speak(sound)
            else:
                sound = "sorry i dont know you "
                Speak(sound)
                Add()


        break

    cap.release()
    cv2.destroyAllWindows()

###################################################################################
# Manage 
###################################################################################
def Manage(v):
    v = str(v)
    if v == "hello":
        sound = "hello sir i am activated "
        Speak(sound)
    elif v == "who am i" or v == "do you know me":

        matching()




###################################################################################
# Speak Function
###################################################################################
def Speak(name):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice' , voices[1].id)
    rate = engine.getProperty('rate')
    engine.setProperty('rate', 150)
    engine.say(name)
    engine.runAndWait()


######################################################################################
# Reconizer
#######################################################################################
def Reconizer():
    r = sr.Recognizer()
    while(1):
        try:
            with sr.Microphone() as source2:
                r.adjust_for_ambient_noise(source2, duration=0.2)
                audio2 = r.listen(source2)
                MyText = r.recognize_google(audio2)
                MyText = MyText.lower()
                print("Did you say "+MyText)
                Manage(MyText)

        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
        except sr.UnknownValueError:
            print("unknown error occured")



Reconizer()






