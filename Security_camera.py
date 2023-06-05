import cv2
import winsound

cap=cv2.VideoCapture(r'C:\COMPUTER VISION\CAMERA\pedestrians (1).avi')
while True:
    fb_cascade=cv2.CascadeClassifier(r"C:\COMPUTER VISION\CAMERA\haarcascade_fullbody.xml")  
    suc,frames1=cap.read()
    gray1=cv2.cvtColor(frames1,cv2.COLOR_BGR2GRAY)
    thresh1,binary_img1=cv2.threshold(gray1,20,255,cv2.THRESH_BINARY)
    fb1=fb_cascade.detectMultiScale(binary_img1,scaleFactor=1.1,minNeighbors=5)

    suc,frames2=cap.read()
    gray2=cv2.cvtColor(frames2,cv2.COLOR_BGR2GRAY)
    thresh2,binary_img2=cv2.threshold(gray2,20,255,cv2.THRESH_BINARY)
    fb2=fb_cascade.detectMultiScale(binary_img2,scaleFactor=1.1,minNeighbors=5)

    diff=cv2.absdiff(frames1,frames2)
    gray3=cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
    thresh3,binary_img3=cv2.threshold(gray3,20,255,cv2.THRESH_BINARY)

    if(len(fb2)>0):
       dilation=cv2.dilate(binary_img3,None,iterations=4)
       contours, _=cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
       for c in contours:
            if cv2.contourArea(c)<3000:
                continue
            x,y,w,h=cv2.boundingRect(c)
            cv2.rectangle(frames1, (x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(frames1,str('ALERT!!! Motion Detected'),(25,430),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            winsound.Beep(1000,200)
    if cv2.waitKey(1)==ord('q'):
        break
    cv2.imshow('webcam',frames1)
 
   
