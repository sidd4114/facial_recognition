import cv2
import pickle
import os
import face_recognition

#importing images of students into list
folderpath=r'facialrecognition\images'

pathlist=os.listdir(folderpath)    #list of stud imgs 
#print(pathlist)

imglist=[] #empty list
studentid=[] #empty list for id

for path in pathlist:
  imglist.append(cv2.imread(os.path.join(folderpath,path)))   #joining path
  #os.path.splitext(path,[0])  #spilting id and .png and getting id
  studentid.append(os.path.splitext(path)[0])


#print(studentid)  #printing student ids
#print(len(imglist))



def findencodings(imageslist):
  encodelist=[]  #creating empty list for encodeings
  for img in imageslist:
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)      #converting bgr to rgb since opencv uses bgr and facerecog uses 
    encode=face_recognition.face_encodings(img)[0]  #finding encodings
    encodelist.append(encode)

  return encodelist

print("encoding started")
encodelistknown=findencodings(imglist) #generating imgs
encodelistknownwithids=[encodelistknown,studentid]         #storing encodings with resp student id
print("encoding comlete")

file=open("encodefile.p",'wb')  #opening file(writing in binary)
pickle.dump(encodelistknownwithids,file)
file.close()
print("file saved")