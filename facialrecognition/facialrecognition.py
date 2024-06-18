import cv2
import os
import pickle
import face_recognition
import numpy as np
import cvzone

cap = cv2.VideoCapture(0)  # Accessing the webcam

cap.set(3, 640)  # Setting width
cap.set(4, 480)  # Setting height

# Background image
imgbackground = cv2.imread(r"C:\Users\siddh\facial_recognition\facialrecognition\resources\background.png")

# Importing mode images into a list
foldermodepath = r'C:\Users\siddh\facial_recognition\facialrecognition\resources\Modes'

modepathlist = os.listdir(foldermodepath)  # A list containing images of modes

imgmodelist = []

for path in modepathlist:
    imgmodelist.append(cv2.imread(os.path.join(foldermodepath, path)))  # Joining path and appending images to the list

# Load the encoding file
print("Loading encoded file...")
with open("encodefile.p", 'rb') as file:
    encodelistknownwithids = pickle.load(file)  # Loading contents from the encoding file

encodelistknown, studentid = encodelistknownwithids  # Separating encodings and IDs
print("Encoded file loaded")

while True:
    success, img = cap.read()

    if not success:
        print("Failed to capture image from webcam")
        break

    # Resizing image for faster processing
    imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)  # Converting into RGB

    facecurframe = face_recognition.face_locations(imgs)  # Detecting the locations of faces in the provided image
    encodecurframe = face_recognition.face_encodings(imgs, facecurframe)  # Computing the face encodings for each detected face

    imgbackground[162:162 + 480, 55:55 + 640] = img  # Slicing the background
    imgbackground[44:44 + 633, 808:808 + 414] = imgmodelist[1]  # Adding the mode image

    for encodeface, faceloc in zip(encodecurframe, facecurframe):  # Releasing contents
        matches = face_recognition.compare_faces(encodelistknown, encodeface)  # Comparing faces
        facedis = face_recognition.face_distance(encodelistknown, encodeface)  # Comparing face distances
        # print("matches", matches)
        # print("facedistance", facedis)

        matchIndex = np.argmin(facedis)  # Finding the least distance
        # print(matchIndex)

        if matches[matchIndex]:  # If current face matches faces in matchIndex
            print("Known face detected")
            print(studentid[matchIndex])
            y1, x2, y2, x1 = faceloc  # According to format
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # Multiply since we have reduced scale by 0.25
            bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1  # Defining bounding box
            imgbackground = cvzone.cornerRect(imgbackground, bbox, rt=0)  # Drawing the bounding box

    # cv2.imshow("webcam", img)
    cv2.imshow("Face attendance", imgbackground)

    if cv2.waitKey(1) & 0xff == ord("q"):  # Exit if 'q' is pressed
        break

cap.release()
cv2.destroyAllWindows()
