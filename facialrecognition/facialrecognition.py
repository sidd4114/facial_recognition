import cv2

cap=cv2.VideoCapture(0)

cap.set(3,640)                              #setting width
cap.set(4,480)                               #setting height

imgbackground=cv2.imread(r"C:\Users\siddh\facial_recognition\facialrecognition\resources\background.png")

while True:
    success,img=cap.read()

    imgbackground[162:162+480,55:55+640]=img   #slicing the background


    #cv2.imshow("webcam",img)
    cv2.imshow("Face attendance",imgbackground)

    if cv2.waitKey(1) & 0xff==ord("q"):
        break
