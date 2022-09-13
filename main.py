from cv2 import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
from PIL import Image
from tkinter import *
from PIL import Image,ImageTk
from PIL import GifImagePlugin
import time
model= joblib.load('mymodel5.pkl')
def roi(img):
    images={}
    imgs=[]
    im=medianBlur(img,5)
    im=GaussianBlur(im,(3,3),0)
    gray=cvtColor(im,COLOR_BGR2GRAY)
    ret,new=threshold(gray,90,255,0)
    cnts,x=findContours(new,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE)
    if len(cnts)>=1:
        for cnt in cnts:
            x,y,w,h = boundingRect(cnt)
            crop=im[y-25:y+h+20,x-20:x+w+20]
            crop=resize(crop,(24,24))
            kernel = np.ones((3,3),np.uint8)
            crop=cv2.dilate(crop,kernel,iterations = 1)
            images[(x+w)//2]=crop
        for i in sorted(images):
            imgs.append(images[i])
    return(imgs)
def predict(images):
    li=[]
    global lst
    global model
    global pca
    for i in images:
        img=flip(resize(i,(28,28)),1)
        gray=cvtColor(img,COLOR_BGR2GRAY)
        M = cv2.getRotationMatrix2D((14,14),90,1)
        img = cv2.warpAffine(gray,M,(28,28))
        img=img.flatten()
        li.append('%s'%(model.predict([img])[0]))
    return(li)
def go():
    im=imread('new.jpg')
    im=medianBlur(im,5)
    im=GaussianBlur(im,(5,5),0)
    img=roi(im)
    li=predict(img)
    s=''
    for i in li:
        s=s+i
    return(s)
    imshow('as',resize(np.hstack(img),(400,100)))
    waitKey(0)
draw=[]
def fun(x):
    pass
def main():
    global draw
    dct=[]
    global lst
    lst=[]
    word=''
    global cam
    cam=VideoCapture(0)
    bg=np.zeros(480*640).reshape(480,640)
    flag=True
    txt=''
    px=0
    py=0
    root=Tk()
    root.overrideredirect(1)
    root.attributes("-topmost",1) 
    root.geometry("%dx%d%+d%+d" % (540,380,400,180))
    can=Canvas(root,height=480,width=640)
    im = Image.open("source.gif")
    for i in range(1):
        for frame in range(0,im.n_frames):
            im.seek(frame)
            im.save("frame.png")
            img=imread('frame.png')
            #rectangle(img,(300,500),(500,400),(255,255,255),-1)
            #putText(img,'L O A D I N G  ',(340,550),FONT_HERSHEY_SIMPLEX,0.8,(200,200,200),2,LINE_AA)
            imwrite('frame.png',resize(img,(540,380)))
            img=Image.open('frame.png')
            img=ImageTk.PhotoImage(img)
            can.create_image(0,0,image=img,anchor=NW)
            can.pack()
            root.update()
    root.destroy()
    while True:
        li=[]
        ct=[]
        r,frame=cam.read()
        fx=frame.shape[0]
        fy=frame.shape[1]
        frame=flip(frame,1)
        x1,y1=0,0
        kernel = np.ones((5,5),np.uint8)
        cr=medianBlur(frame,5)
        hsv=cvtColor(cr,COLOR_BGR2HSV)
        green=inRange(hsv,(45,100,40),(91,255,255))
        net=GaussianBlur(green,(5,5),0)
        kernel = np.ones((9,9),np.uint8)
        grn=morphologyEx(net,MORPH_OPEN, kernel)
        opening=Canny(grn,0,255)
        cnt,h=findContours(opening,RETR_EXTERNAL,CHAIN_APPROX_NONE)
        ar=[]     
        if cnt!=None:
            for c in cnt:
                x,y,w,h = boundingRect(c)
                dx=x+(w//2)
                dy=y+(h//2)
                ar.append([dx,dy,w*h])
            if len(ar)>0:
                p=sorted([x[2] for x in ar])[-1]
                pt=[x for x in ar if x[2]==p]
                x=15+int(pt[0][0])
                y=int(pt[0][1])-70
                if flag==True: 
                        draw.append([x,y])
                        if px!=0 and py!=0:
                            if len(draw)>0: 
                                for i in range(len(draw)-1):
                                    if draw[i][0]!='*' and draw[i+1][0]!='*':
                                        sx=draw[i][0]
                                        sy=draw[i][1]
                                        nx=draw[i+1][0]
                                        ny=draw[i+1][1]
                                        line(frame,(x1+sx-30,y1+sy-25),(x1-30+nx,y1+ny-25),(225,226,6),3)
                            line(bg,(px,py),(x,y),(255,255,255),3)
                        px,py=x,y
                else:
                    for i in range(len(draw)-1):
                                    if draw[i][0]!='*' and draw[i+1][0]!='*':
                                        sx=draw[i][0]
                                        sy=draw[i][1]
                                        nx=draw[i+1][0]
                                        ny=draw[i+1][1]
                                        line(frame,(x1+sx-30,y1+sy-25),(x1-30+nx,y1+ny-25),(225,226,6),3)
                    px,py=0,0
                    circle(frame,(x1+x-30,y+y1-25),3,(225,226,6),-1)
                    draw.append(['*'])
            else:
                ct.append(False)
        ovl=frame.copy()
        rectangle(ovl,(11,11),(fy-10,80),(95,44,32),-1)
        rectangle(ovl,(400,380),(628,440),(80,65,0),-1)
        frame=addWeighted(ovl,0.7,frame,1,0.0)
        putText(frame,'%s'%txt,(430,420),FONT_HERSHEY_SIMPLEX  ,0.8,(255,255,255),2,LINE_AA)
        putText(frame,'Gesture Recognition',(180,50),FONT_HERSHEY_DUPLEX ,1,(255,255,255),1,LINE_AA)
        putText(frame,' %s'%(word),(300,590),FONT_HERSHEY_DUPLEX,.8,(220,220,220),1,LINE_AA)
        rectangle(frame,(10,10),(fy-10,fx-10),(153,158,0),2)
        rectangle(frame,(400,380),(626,440),(255,255,255),2)
        crop=frame[15:75,90:150]
        g=resize(imread('ml.png'),(60,60))
        crop=addWeighted(g,1,crop,1,0)
        frame[15:75,90:150]=crop
        namedWindow('Gesture Recognition',WINDOW_NORMAL)
        moveWindow('Gesture Recognition', 360, 120)
        imshow('Gesture Recognition',frame)
        if len(ct)>1:
            word=''
            bg=np.zeros(fx*fy).reshape(fx,fy)
            draw=[]
        w=waitKey(1)           
        if w==27:
            break
        elif w==ord('a'):
            word=''
            bg=np.zeros(fx*fy).reshape(fx,fy)
            draw=[]
        elif w==ord('s'):
            flag=False
        elif w==ord('d'):
            flag=True
        elif w==ord('f'):
            s=''
            flag=True
            imwrite('new.jpg',bg)
            txt=go()
            for i in txt:
                s=s+i
            word=s
            bg=np.zeros(fx*fy).reshape(fx,fy)
            draw=[]    
    cam.release()
    destroyAllWindows()
if __name__==main():
    lst=[]
    main()
