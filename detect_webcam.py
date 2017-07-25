#to run open terminal and enter this command
# "python detect_webcam.py" without inverted commas
import cv2
import sys


cascPath = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
                        #proper path of cascade file

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)  #loading cascade in memory
print(faceCascade)    #printing cascade info
# Capturing the image
cam=cv2.VideoCapture(0)
raw_input("Press Enter to Capture")
rv,image=cam.read()
cv2.imshow("Press 0 two times, if Ok, else only one time",image)
cv2.waitKey(0)
del(cam)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #converting it to greyscale

# Detect faces in the image
#faces(list) are coordinates and length of found faces
faces = faceCascade.detectMultiScale(
    gray,     #greyscale image
    scaleFactor=1.1,    #for closer (the faces which are bigger) faces
    minNeighbors=4,
          #how many objects are detected near the current one before it declares the face found
    minSize=(30, 30),   #size of each window
    flags = cv2.CASCADE_SCALE_IMAGE
)
nof=len(faces)   #total no of faces found
print("Found "+str(nof)+" Faces")

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)   #showing image with faces with rectangle
cv2.waitKey(0)
