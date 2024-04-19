import cv2
import numpy as np
import face_recognition

# 1 loading images & changes ratio to rgb
imgelon = face_recognition.load_image_file('imagesBasic/elonsample.jpg')
imgelon = cv2.cvtColor(imgelon, cv2.COLOR_BGR2RGB)
imgtest = face_recognition.load_image_file('imagesBasic/zbr.jpg')
imgtest = cv2.cvtColor(imgtest, cv2.COLOR_BGR2RGB)

# 3 points out faces with frame
faceLoc = face_recognition.face_locations(imgelon)[0]
encode_elon = face_recognition.face_encodings(imgelon)[0]
cv2.rectangle(imgelon,(faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255),2)

faceLoctest = face_recognition.face_locations(imgtest)[0]
encode_elon_test = face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(faceLoctest[3], faceLoctest[0]), (faceLoctest[1], faceLoctest[2]), (255, 0, 255),2)

# 4 compatering measurmets of sample and test images
                                # list of known images , image to be tested
results = face_recognition.compare_faces([encode_elon],encode_elon_test)
face_dist = face_recognition.face_distance([encode_elon],encode_elon_test)
# print(results, face_dist)
cv2.putText(imgtest,f'{results} {round(face_dist[0], 2)}',(50 ,50), cv2.QT_FONT_NORMAL,1, (0, 0, 255), 2)

# 2 shows imgages
cv2.imshow('Elon Musk', imgelon)
cv2.imshow('Elon Musk test', imgtest)
cv2.waitKey(0)
