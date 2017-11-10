
from Tkinter import *
import os
import Image
import ctypes
from PIL import Image
from PIL import ImageTk
from PIL import ImageOps
from tkFileDialog import*
import tkMessageBox
import imghdr
import ImageDraw
from collections import*
import cv2
import numpy

"""
CITATIONS
I found the command for changing the desktop background from this webpage:
http://stackoverflow.com/questions/14426475/change-wallpaper-in-python-for-user-while-being-system
"""
################ DRAW ################
def drawOnImage(canvas):
    canvas.data.colourPopToHappen=False
    canvas.data.cropPopToHappen=False
    canvas.data.drawOn=True
    drawWindow=Toplevel(canvas.data.mainWindow)
    drawWindow.title="Draw"
    drawFrame=Frame(drawWindow)
    blackButton=Button(drawFrame, bg="black",width=2,\
                       command=lambda: colourChosen(drawWindow,canvas, "black"))
    blackButton.grid(row=3,column=0)
    whiteButton=Button(drawFrame, bg="white",width=2, \
                       command=lambda: colourChosen(drawWindow,canvas, "white"))
    whiteButton.grid(row=3,column=1)
    drawFrame.pack(side=BOTTOM)

def colourChosen(drawWindow, canvas, colour):
    if canvas.data.image!=None:
        canvas.data.drawColour=colour
        canvas.data.mainWindow.bind("<B1-Motion>",\
                                    lambda event: drawDraw(event, canvas))
    drawWindow.destroy()
    
def drawDraw(event, canvas):
    if canvas.data.drawOn==True:
        x=int(round((event.x-canvas.data.imageTopX)*canvas.data.imageScale))
        y=int(round((event.y-canvas.data.imageTopY)*canvas.data.imageScale))
        draw = ImageDraw.Draw(canvas.data.image)
        draw.ellipse((x-3, y-3, x+ 3, y+3), fill=canvas.data.drawColour,\
                     outline=None)
        save(canvas)
        canvas.data.undoQueue.append(canvas.data.image.copy())
        canvas.data.imageForTk=makeImageForTk(canvas)
        drawImage(canvas)
    

######################## FEATURES ###########################       
def reset(canvas):
    canvas.data.colourPopToHappen=False
    canvas.data.cropPopToHappen=False
    canvas.data.drawOn=False
    ### change back to original image
    if canvas.data.image!=None:
        canvas.data.image=canvas.data.originalImage.copy()
        save(canvas)
        canvas.data.undoQueue.append(canvas.data.image.copy())
        canvas.data.imageForTk=makeImageForTk(canvas)
        drawImage(canvas)

def eyecorrection(canvas):
    if canvas.data.image!=None:
        img = cv2.imread(imageName)
        #Load HAAR cascade for faces
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        #Load HAAR cascade for eyes
        eye_cascade = cv2.CascadeClassifier("parojosG.xml")

        # Output image
        imgOut = img.copy()

        #Convert image to grayscale 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=4)
        #print("Number of faces",len(faces))

        if len(faces)>0:    
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                #cv2.imshow('img1', img)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]

                eyes = eye_cascade.detectMultiScale(roi_color,scaleFactor=1.3, minNeighbors=1)
                #print("Number of eyes",len(eyes))

                for (ex, ey, ew, eh) in eyes:
                    #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                    #cv2.imshow('img2', roi_color)
                    #cv2.waitKey(0)
                    eye = roi_color[ey:ey+eh, ex:ex+ew]
                    #cv2.imshow('roi_color', eye)

                    hsv = cv2.cvtColor(eye, cv2.COLOR_BGR2HSV)

                    h = hsv[:,:,0]
                    s = hsv[:,:,1]
                    v = hsv[:,:,2]

                    b = eye[:, :, 0]
                    g = eye[:, :, 1]
                    r = eye[:, :, 2]
                    bg = cv2.add(b,g)
                    #bg2 = cv2.add(b*b,g*g)
                    #rb = cv2.add(r,b)
                    #rgb = cv2.add(rb,g)
                    mask = (r > 100) & (r > bg)
                    #mask = (r > 120) & (r > 0.4*rgb) & (g > 0.31*rgb) & (b < 0.36*rgb)
                    mask = mask.astype(numpy.uint8)*255
                    ker1 = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
                    mask = cv2.erode(mask,ker1,iterations = 1)


                def fillHoles( mask ):
                    maskFloodfill = mask.copy()
                    h, w = maskFloodfill.shape[:2]
                    maskTemp = numpy.zeros((h+2, w+2), numpy.uint8)
                    cv2.floodFill(maskFloodfill, maskTemp, (0, 0), 255)
                    mask2 = cv2.bitwise_not(maskFloodfill)
                    return mask2 | mask


                # Clean up mask by filling holes and dilating
                mask = fillHoles(mask)
                #print("maskfill",mask)
                ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6))
                #mask = cv2.dilate(mask,ker,iterations = 1)
                mask = cv2.dilate(mask, None, anchor=(-1, -1), iterations=3, borderType=1, borderValue=1)


                # Calculate the mean channel by averaging
                # the green and blue channels. Recall, bg = cv2.add(b, g)
                mean = bg / 2
                mask = mask.astype(numpy.bool)[:, :, numpy.newaxis]
                mean = mean[:, :, numpy.newaxis]
                 
                # Copy the eye from the original image. 
                eyeOut = eye.copy()
                 
                # Copy the mean image to the output image. 
                numpy.copyto(eyeOut, mean, where=mask)

                # Copy the fixed eye to the output image. 
                #imgOut[ey+y:ey+y+eh, ex+x:ex+x+ew, :] = eyeOut
                imgOut[ey+y:ey+eh+y, ex+x:ex+ew+x, :] = eyeOut

        else:
            eyes = eye_cascade.detectMultiScale(img,scaleFactor=1.3, minNeighbors=4)
            #print("Number of eyes",len(eyes))
            for (ex, ey, ew, eh) in eyes:

                eye = img[ey:ey+eh, ex:ex+ew]
                #cv2.imshow('roi_color', eye)

                b = eye[:, :, 0]
                g = eye[:, :, 1]
                r = eye[:, :, 2]

                bg = cv2.add(b,g)
                #rb = cv2.add(r,b)
                #rgb = cv2.add(rb,g)
                mask = (r > 70) & (r > bg) 
                #mask = (r > 150) & (r > 0.4*rgb) & (g > 0.31*rgb) & (b < 0.36*rgb)
                mask = mask.astype(numpy.uint8)*255
                #ker1 = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
                #mask = cv2.erode(mask,ker1,iterations = 1)


            def fillHoles( mask ):
                maskFloodfill = mask.copy()
                h, w = maskFloodfill.shape[:2]
                maskTemp = numpy.zeros((h+2, w+2), numpy.uint8)
                cv2.floodFill(maskFloodfill, maskTemp, (0, 0), 255)
                mask2 = cv2.bitwise_not(maskFloodfill)
                return mask2 | mask


            # Clean up mask by filling holes and dilating
            mask = fillHoles(mask)
            mask = cv2.dilate(mask, None, anchor=(-1, -1), iterations=3, borderType=1, borderValue=1)


            # Calculate the mean channel by averaging
            # the green and blue channels. Recall, bg = cv2.add(b, g)
            mean = bg / 2
            mask = mask.astype(numpy.bool)[:, :, numpy.newaxis]
            mean = mean[:, :, numpy.newaxis]
             
            # Copy the eye from the original image. 
            eyeOut = eye.copy()
             
            # Copy the mean image to the output image. 
            numpy.copyto(eyeOut, mean, where=mask)

            # Copy the fixed eye to the output image. 
            imgOut[ey:ey+eh, ex:ex+ew, :] = eyeOut
    cv2.imwrite('Out.jpg', imgOut)
    im= Image.open('Out.jpg')
    canvas.data.image = im
    canvas.data.imageForTk=makeImageForTk(canvas)
    drawImage(canvas)

def autoenhance(canvas):
    if canvas.data.image!=None:
        img = cv2.imread(imageName,-1)
        c = len(img.shape)
        print(c)
        if (c==3):
            #Denoising
            img = cv2.medianBlur(img,5)
            dst = cv2.fastNlMeansDenoisingColored(img,None,3,3,7,21)
            #cv2.imshow('dst',dst)

            b = dst[:,:,0]
            #cv2.imshow('b',b)
            g = dst[:,:,1]
            #cv2.imshow('g',g)
            r = dst[:,:,2]
            #cv2.imshow('r',r)

            #using CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

            Ab = clahe.apply(b)
            #cv2.imshow('b1',Ab1)
            Ag = clahe.apply(g)
            #cv2.imshow('g1',Ag1)
            Ar = clahe.apply(r)
            #cv2.imshow('r1',Ar1)


            #Image Sharpening
            Lb = cv2.Laplacian(Ab,cv2.CV_64F)
            abs_Lb = numpy.absolute(Lb)
            Lb_8u = numpy.uint8(abs_Lb)
            #cv2.imshow('Lb',Lb_8u)

            Lg = cv2.Laplacian(Ag,cv2.CV_64F)
            abs_Lg = numpy.absolute(Lg)
            Lg_8u = numpy.uint8(abs_Lg)
            #cv2.imshow('Lg',Lg_8u)

            Lr = cv2.Laplacian(Ar,cv2.CV_64F)
            abs_Lr = numpy.absolute(Lr)
            Lr_8u = numpy.uint8(abs_Lr)
            #cv2.imshow('Lr',Lr_8u)

            Ab1 = cv2.add(Ab,Lb_8u)
            #cv2.imshow('Ab',Ab1)
            Ag1 = cv2.add(Ag,Lg_8u) 
            #cv2.imshow('Ag',Ag1)
            Ar1 = cv2.add(Ar,Lr_8u)
            #cv2.imshow('Ar',Ar1)

            out = cv2.merge((Ab1,Ag1,Ar1))

        else:
            #Denoising
            dst = cv2.fastNlMeansDenoising(img,None,5,7,21)
            #cv2.imshow('dst',dst)

            #Histrogram equalization
            equ = cv2.equalizeHist(dst)

            #Contrast Enhancement
            #using CLAHE (Contrast Limited Adaptive Histogram Equalization)
            #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(21,21))
            #equ = clahe.apply(dst)

            #Image Sharpening
            dst_sharp = cv2.Laplacian(dst,cv2.CV_64F)
            abs_dst = numpy.absolute(dst_sharp)
            dst_8u = numpy.uint8(abs_dst)
            #cv2.imshow('Lb',dst_8u)

            out = cv2.add(equ,dst_8u)
            #cv2.imshow('out',out)
    cv2.imwrite('Out1.jpg', out)
    im= Image.open('Out1.jpg')
    canvas.data.image = im
    canvas.data.imageForTk=makeImageForTk(canvas)
    drawImage(canvas)
        

################ EDIT MENU FUNCTIONS ############################

def keyPressed(canvas, event):
    if event.keysym=="z":
        undo(canvas)
    elif event.keysym=="y":
        redo(canvas)
        

# we use deques so as to make Undo and Redo more efficient and avoid
# memory space isuues 
# after each change, we append the new version of the image to
# the Undo queue
def undo(canvas):
    if len(canvas.data.undoQueue)>0:
        # the last element of the Undo Deque is the
        # current version of the image
        lastImage=canvas.data.undoQueue.pop()
        # we would want the current version if wehit redo after undo
        canvas.data.redoQueue.appendleft(lastImage)
    if len(canvas.data.undoQueue)>0:
        # the previous version of the image
        canvas.data.image=canvas.data.undoQueue[-1]
    save(canvas)
    canvas.data.imageForTk=makeImageForTk(canvas)
    drawImage(canvas)

def redo(canvas):
    if len(canvas.data.redoQueue)>0:
        canvas.data.image=canvas.data.redoQueue[0]
    save(canvas)
    if len(canvas.data.redoQueue)>0:
        # we remove this version from the Redo Deque beacuase it
        # has become our current image
        lastImage=canvas.data.redoQueue.popleft()
        canvas.data.undoQueue.append(lastImage)
    canvas.data.imageForTk=makeImageForTk(canvas)
    drawImage(canvas)

############# MENU COMMANDS ################

def saveAs(canvas):
    # ask where the user wants to save the file
    if canvas.data.image!=None:
        filename=asksaveasfilename(defaultextension=".jpg")
        im=canvas.data.image
        im.save(filename)

def save(canvas):
    if canvas.data.image!=None:
        im=canvas.data.image
        im.save(canvas.data.imageLocation)

def newImage(canvas):
    global imageName
    imageName=askopenfilename()
    filetype=""
    #make sure it's an image file
    try: filetype=imghdr.what(imageName)
    except:
        tkMessageBox.showinfo(title="Image File",\
        message="Choose an Image File!" , parent=canvas.data.mainWindow)
    # restrict filetypes to .jpg, .bmp, etc.
    if filetype in ['jpeg', 'bmp', 'png', 'tiff','tif','gif']:
        canvas.data.imageLocation=imageName
        im= Image.open(imageName)
        canvas.data.image=im
        canvas.data.originalImage=im.copy()
        canvas.data.undoQueue.append(im.copy())
        canvas.data.imageSize=im.size #Original Image dimensions
        canvas.data.imageForTk=makeImageForTk(canvas)
        drawImage(canvas)
    else:
        tkMessageBox.showinfo(title="Image File",\
        message="Choose an Image File!" , parent=canvas.data.mainWindow)


######## CREATE A VERSION OF IMAGE TO BE DISPLAYED ON THE CANVAS #########

def makeImageForTk(canvas):
    im=canvas.data.image
    if canvas.data.image!=None:
        # Beacuse after cropping the now 'image' might have diffrent
        # dimensional ratios
        imageWidth=canvas.data.image.size[0] 
        imageHeight=canvas.data.image.size[1]
        #To make biggest version of the image fit inside the canvas
        if imageWidth>imageHeight:
            resizedImage=im.resize((canvas.data.width,\
                int(round(float(imageHeight)*canvas.data.width/imageWidth))))
            # store the scale so as to use it later
            canvas.data.imageScale=float(imageWidth)/canvas.data.width
        else:
            resizedImage=im.resize((int(round(float(imageWidth)*canvas.data.height/imageHeight)),\
                                    canvas.data.height))
            canvas.data.imageScale=float(imageHeight)/canvas.data.height
        # we may need to refer to ther resized image atttributes again
        canvas.data.resizedIm=resizedImage
        return ImageTk.PhotoImage(resizedImage)
 
def drawImage(canvas):
    if canvas.data.image!=None:
        # make the canvas center and the image center the same
        canvas.create_image(canvas.data.width/2.0-canvas.data.resizedIm.size[0]/2.0,
                        canvas.data.height/2.0-canvas.data.resizedIm.size[1]/2.0,
                            anchor=NW, image=canvas.data.imageForTk)
        canvas.data.imageTopX=int(round(canvas.data.width/2.0-canvas.data.resizedIm.size[0]/2.0))
        canvas.data.imageTopY=int(round(canvas.data.height/2.0-canvas.data.resizedIm.size[1]/2.0))

############ INITIALIZE ##############

def init(root, canvas):

    buttonsInit(root, canvas)
    menuInit(root, canvas)
    canvas.data.image=None
    canvas.data.drawOn=True
    canvas.data.undoQueue=deque([], 10)
    canvas.data.redoQueue=deque([], 10)
    canvas.pack()

def buttonsInit(root, canvas):
    backgroundColour="gray"
    buttonWidth=14
    buttonHeight=2
    toolKitFrame=Frame(root)

    eyecorrectionButton=Button(toolKitFrame, text="Red eye correction",\
                           background=backgroundColour, width=buttonWidth,\
                           height=buttonHeight,command=lambda: eyecorrection(canvas))
    eyecorrectionButton.grid(row=0,column=1)

    autoenhanceButton=Button(toolKitFrame, text="Autoenhance",\
                           background=backgroundColour, width=buttonWidth,\
                           height=buttonHeight,command=lambda: autoenhance(canvas))
    autoenhanceButton.grid(row=0,column=2)

    drawButton=Button(toolKitFrame, text="Draw",\
                      background=backgroundColour ,width=buttonWidth,\
                      height=buttonHeight,command=lambda: drawOnImage(canvas))
    drawButton.grid(row=0,column=3)

    resetButton=Button(toolKitFrame, text="Reset",\
                       background=backgroundColour ,width=buttonWidth,\
                       height=buttonHeight, command=lambda: reset(canvas))
    resetButton.grid(row=0,column=4)

    toolKitFrame.pack(side=BOTTOM)

def menuInit(root, canvas):
    menubar=Menu(root)
    menubar.add_command(label="New", command=lambda:newImage(canvas))
    menubar.add_command(label="Save", command=lambda:save(canvas))
    menubar.add_command(label="Save As", command=lambda:saveAs(canvas))
    ## Edit pull-down Menu
    editmenu = Menu(menubar, tearoff=0)
    editmenu.add_command(label="Undo   Z", command=lambda:undo(canvas))
    editmenu.add_command(label="Redo   Y", command=lambda:redo(canvas))
    menubar.add_cascade(label="Edit", menu=editmenu)
    root.config(menu=menubar)
    
def run():
    # create the root and the canvas
    root = Tk()
    root.title("Image Editor")
    canvasWidth=500
    canvasHeight=500
    #canvas = Canvas(root, width=canvasWidth, height=canvasHeight, background="black")
    canvas = Canvas(root, width=canvasWidth, height=canvasHeight, background="black",relief='groove',borderwidth=5)
    # Set up canvas data and call init
    
    class Struct: pass
    canvas.data = Struct()
    canvas.data.width=canvasWidth
    canvas.data.height=canvasHeight
    canvas.data.mainWindow=root
    init(root, canvas)
    
    root.bind("<Key>", lambda event:keyPressed(canvas, event))
    # and launch the app
    root.mainloop()  # This call BLOCKS (so your program waits)


run()
