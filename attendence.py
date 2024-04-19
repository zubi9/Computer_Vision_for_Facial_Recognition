import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# retrieving images from directories
path = 'imagesAttendence'
images = []
facultyNames = []
myList = os.listdir(path)
# print(myList)
for flt in myList:
    curImg = cv2.imread(f'{path}/{flt}')
    images.append(curImg)
    facultyNames.append(os.path.splitext(flt)[0])
# print(facultyNames)

# find encodings of the sample images


def findEndcodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeImg = face_recognition.face_encodings(img)[0]
        encodeList.append(encodeImg)
    return encodeList


def markAttendence(name):
    with open('Attendence.csv','r+') as f:
        mydateList = f.readlines()
        nameList = []
        # print(mydateList)
        for line in mydateList:
            entry = line.split(',')
            nameList.append(entry[0])
            # print(entry)

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name}, {dtString}')


encodListOfKnownImgs = findEndcodings(images)
print('encoding complete')
# capturing images from camera

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgScale = cv2.resize(img,(0, 0),None, 0.25, 0.25)
    imgScale = cv2.cvtColor(imgScale, cv2.COLOR_BGR2RGB)

    facesInCurFram = face_recognition.face_locations(imgScale)
    encodeOfCurFram = face_recognition.face_encodings(imgScale, facesInCurFram)

    for encodFace, faceLoc in zip(encodeOfCurFram, facesInCurFram):
        matches = face_recognition.compare_faces(encodListOfKnownImgs, encodFace)
        faceDist = face_recognition.face_distance(encodListOfKnownImgs, encodFace)
        # print(faceDist)
        # print(matches)
        matchIndex = np.argmin(faceDist)
        print(matchIndex)

        if matches[matchIndex]:
            name = facultyNames[matchIndex].upper()
            print(name)

            y1, x1, y2, x2 = faceLoc
            y1, x1, y2, x2 = y1*4, x1*4, y2*4, x2*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.rectangle(img, (x1, y2-24), (x2, y2), (0,255,0),cv2.FILLED)
            cv2.putText(img, name, (x2+6, y2-6),cv2.FONT_HERSHEY_PLAIN,1,(255, 255, 255), 2)
            markAttendence(name)

    cv2.imshow('webcam',img)
    cv2.waitKey(1)
