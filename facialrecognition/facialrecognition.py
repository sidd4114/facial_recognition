import cv2
import os
import pickle
import face_recognition


cap=cv2.VideoCapture(0)  #accessing webcam

cap.set(3,640)                              #setting width
cap.set(4,480)                               #setting height

imgbackground=cv2.imread(r"C:\Users\siddh\facial_recognition\facialrecognition\resources\background.png")  #background img

#importing mode images into list
foldermodepath=r'facialrecognition\resources\Modes'

modepathlist=os.listdir(foldermodepath)                                                    #a list containing imgs

imgmodelist=[] 

for path in modepathlist:
  imgmodelist.append(cv2.imread(os.path.join(foldermodepath,path)))   #joining path
  
  
#load the encoding file
print("loading encoded file...")
file=open("encodefile.p",'rb')             
encodelistknownwithids=pickle.load(file)   #loading contents
file.close()
encodelistknown,studentid=encodelistknownwithids #seperating ids and encoding
print("encoded file loaded")

while True:
    success,img=cap.read()

    imgs=cv2.resize(img,(0,0,None,0.25,0.25))   #resizing image
    imgs=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)    #converting into rgb
    
    facecurframe=face_recognition.face_locations(imgs)                              # detects the locations of faces in the provided image (imgs) and returns a list of tuples. Each tuple represents the coordinates of a detected face in the format
    
    encodecurframe=face_recognition.face_encodings(imgs,facecurframe)

    imgbackground[162:162+480,55:55+640]=img   #slicing the background
    imgbackground[44:44+633,808:808+414]=imgmodelist[1]
    


    #cv2.imshow("webcam",img)
    cv2.imshow("Face attendance",imgbackground)

    if cv2.waitKey(1) & 0xff==ord("q"):
        break
