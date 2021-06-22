#!/usr/bin/env python
# ‐*‐ coding: utf‐8 ‐*‐
# coded in Python3.x

"""

@author: Akshay Anand
<www.akshayjha.co.nr>
<www.GitHub.com/AxayJha>
<akshayjha@live.in>
Last modified: 28/04/2017

"""

from tkinter import *
from tkinter.filedialog import *

from image import *
import PIL
from PIL import ImageTk
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


global img
global zoomedimg
# global Image class file to handle images

zoomval=1
allundo=[]
# array of all previous edits

allredo=[]
# array of all reverted edits

filename=""
# global filename storing the name and location of current file

originalPhoto=None
# ImageTk.PhotoImage file storing the initial photo

def view(image):
    """function to pack the provided image file in the canvas of the root"""

    global label, filename, canvas, hbar, vbar, root, originalPhoto, statusbar, img
    ###img = PILimage.open(file)

    #photo = ImageTk.PhotoImage(file=file)
    photo=ImageTk.PhotoImage(image.data)
    # opening file PhotoImage file
    originalPhoto=photo

    root.photo=photo # keeping a reference

    # sets the size of root equal to the size of image
    canvas.create_image(0,0, image=photo, anchor="nw")
    canvas.config(scrollregion=canvas.bbox(ALL))

    hbar.config(command=canvas.xview)
    vbar.config(command=canvas.yview)



    if photo.height()<root.winfo_screenheight() or photo.width()<root.winfo_screenwidth():
        # if the image's both dimensions are smaller than screen resolution
        root.wm_state('normal')
        root.geometry("%dx%d+%d+%d" % (photo.width()+17, photo.height()+17,root.winfo_x(), root.winfo_y()))
        canvas.config(width=photo.width(), height=photo.height(), scrollregion = (0, 0, photo.width(), photo.height()))
        return None

    canvas.config(width=photo.width(), height=photo.height(), scrollregion=(0, 0, photo.width(), photo.height()))

    root.wm_state('zoomed')  # maximizing the window

    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry("%dx%d+0+0" % (w, h))
    # ****** don't bother **********
    #label = Label(canvas, image=photo)
    #canvas.create_image(image=photo)
    #canvas.config(width=photo.width(), height=photo.height())
    #label.image = photo  # keep a reference!
    #label.pack()
    # ******************************


def open(event=None):
    global img, filename, root, allundo, statusbar

    try:
        # show the open file dialog
        filein = askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.bmp *.pgm *.ppm *.gif *.tiff"),], title="Open image file")
        if(len(filein)>0):
            # check if there is something in filename
            filename = filein
            f = (filename[-1::-1]).partition("/")[0][-1::-1]
            root.title(f+" - PixelView")
            # create Image object from the file
            img = Image()
            img.getData(filein)

            # add to the 'previous edits' array
            allundo.append(img)

            # view the image
            view(img)


    except:
        # ignore if file open dialog is invoked but closed without selecting anything
        raise



def check():
    """"checks if the image is loaded"""
    global filename
    if filename=="":
        # if not invoke the open file dialog again
        open()

def save(event=None):
    """saves the Image class file as output image file on desk in
    specified format"""

    if filename=="":
        return
    # check if any image is loaded

    global img


    # invoke the save file as dialog
    fileout = asksaveasfilename(filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"),], title="Save image file")
    if fileout=="": return None

    if "." in fileout:
        # if file extension provided, save it as it is
        img.data.save(fileout)
    else:
        # if no file extension provided, save it as .png file by default
        img.data.save(fileout+".png")

def gray():
    """converts the image into black and white equivalent
        Handler for Grayscale option from Menu bar
    """
    check()
    # check if any image is loaded

    global img, allundo
    img=img.grayscale() # convert and store the Image class file

    allundo.append(img) # saves an instant to have the previous states later

    # save in a temporary file and view the image
    
    view(img)

def PrewittX():
    """Prewitt horizonantal edge detection and filtering """

    check()
    # check if any image is loaded
    global img, allundo

    img=img.prewitt(0) # 0 parameter means horizontal edge detection
    # see Image class file for more details on implmentation

    allundo.append(img) # saves an instant to have the previous states later

    #save in a temporary file and view the image
    
    view(img)

def PrewittY():
    """Prewitt verticle edge detection and filtering """

    check()
    # check if any image is loaded

    global img, allundo

    img=img.prewitt(1) # 0 parameter means horizontal edge detection
    # see Image class file for more details on implmentation
    allundo.append(img)
    
    view(img)


def Sobel():
    """Sobel edge detection and filtering"""

    check()
    # check if any image is loaded
    global img, allundo
    img=img.sobel_grad()
    allundo.append(img)
    
    view(img)


def SobelX():
    """Sobel edge detection and filtering"""

    check()
    # check if any image is loaded
    global img, allundo
    img = img.sobel(0)
    allundo.append(img)

    view(img)


def SobelY():
    """Sobel edge detection and filtering"""

    check()
    # check if any image is loaded
    global img, allundo
    img = img.sobel(1)
    allundo.append(img)

    view(img)

def RobertX():

    check()
    global img, allundo
    img = img.roberts(0)
    allundo.append(img)
    
    view(img)

def RobertY():
    check()
    global img, allundo
    img = img.roberts(1)
    allundo.append(img)
    
    view(img)

def Canny():
    check()
    global img, allundo
    img = img.canny()
    allundo.append(img)
    
    view(img)

def Gaussian():
    check()
    global img, allundo
    img = img.gaussian_blur()
    allundo.append(img)
    
    view(img)

def Blur():
    check()
    global img, allundo
    img = img.blur()
    allundo.append(img)
    
    view(img)


def Crop():
    check()
    global img
    img.crop()


def Solarize():
    check()
    global img, allundo
    img = img.solarize()
    allundo.append(img)
    
    view(img)

def Mirror():
    check()
    global img, allundo
    img = img.mirror()
    allundo.append(img)
    
    view(img)

def Flip():
    check()
    global img, allundo
    img = img.flip()
    allundo.append(img)
    view(img)

def Fit(event=None):
    check()
    global img, root, zoomval
    width=root.winfo_reqwidth()
    height=root.winfo_reqheight()
    img = img.resize1(width,height)
    zoomval=1
    allundo.append(img)
    
    view(img)
    root.geometry("%dx%d+%d+%d" % (img.data.size[0], img.data.size[1], root.winfo_x(), root.winfo_y()))

def Equalize():
    check()
    global img, allundo
    img = img.equalize()
    allundo.append(img)
    
    view(img)


def removeBorderW():
    global removeborderW
    try:
        removeborderW.deiconify()
    except TclError:
        removeborderW.mainloop()
        removeborderW.deiconify()

def removeBorder(event=None):
    check()
    global img, allundo, removeborderW, borderpixels

    try:
        pixels = abs(int(borderpixels.get()))
        img = img.removeBorder(pixels)
        removeborderW.withdraw()
        allundo.append(img)
        
        view(img)

    except:
        from tkinter import messagebox
        messagebox.showinfo("Invalid input", "Enter a valid value ")
        removeBorderW()

def setWallpaper():
    """Doesnt work"""
    global img
    import ctypes
    #cache= set dir when using
    try:
        
        SPI_SETDESKWALLPAPER = 20
        ctypes.windll.user32.SystemParametersInfoW(SPI_SETDESKWALLPAPER, 0, os.path.abspath(cache), 3)
    except:
        im=Image(PILimage.fromarray(np.zeros(shape=(200,200)).astype(int)).convert('RGB'))
        im.save(cache)
        SPI_SETDESKWALLPAPER = 20
        ctypes.windll.user32.SystemParametersInfoW(SPI_SETDESKWALLPAPER, 0, os.path.abspath(cache), 3)

def Colourize():
    check()
    global img, allundo
    img=img.colourize()
    allundo.append(img)
    
    view(img)

def autocontrast():
    check()
    global img, allundo
    img = img.autocontrast()
    allundo.append(img)
    
    view(img)


def Posterize():
    check()
    global img, allundo
    img = img.posterize()
    allundo.append(img)
    
    view(img)

def Laplacian():
    check()
    global img, allundo
    img = img.laplacian()
    allundo.append(img)
    
    view(img)

def ReduceNoise():
    check()
    global img, allundo
    img = img.bilateral_blur()
    allundo.append(img)
    
    view(img)

def MedianBlur():
    check()
    global img, allundo
    img = img.median_blur()
    allundo.append(img)
    
    view(img)

def Sharpen():
    check()
    global img, allundo
    img = img.sharpen()
    allundo.append(img)
    
    view(img)

def SmallBlur():
    check()
    global img, allundo
    img = img.smallBlur()
    allundo.append(img)
    
    view(img)

def LargeBlur():
    check()
    global img, allundo
    img = img.largeBlur()
    allundo.append(img)
    
    view(img)

def Negative():
    check()
    global img, allundo
    img = img.negative()
    allundo.append(img)
    
    view(img)

def RotateL(event=None):
    check()
    global img, allundo
    img = img.rotate_image(90)
    allundo.append(img)
    view(img)


def RotateR(event=None):
    check()
    global img, allundo
    img = img.rotate_image(-90)
    allundo.append(img)
    
    view(img)

def Scharr():
    check()
    global img, allundo
    img = img.scharr()
    allundo.append(img)
    view(img)


def logT():
    check()
    global img, allundo
    img= img.log()
    allundo.append(img)
    
    view(img)

def powerT():
    check()
    global img, allundo
    img= img.power()
    allundo.append(img)
    
    view(img)

def contrastS():
    check()
    global img, allundo
    img= img.contrastStretch()
    allundo.append(img)
    
    view(img)

def sliceG():
    check()
    global img, allundo
    img= img.slice()
    allundo.append(img)
    
    label.pack_forget()
    view(img)

def sliceB():
    check()
    global img, allundo
    img= img.bitSlice()
    allundo.append(img)
    
    view(img)

def hist():
    check()
    global img
    img.histogram()

def About():
    global about
    try:
        about.deiconify()
    except TclError:
        about.mainloop()
        about.deiconify()

def ResizeW():
    global resize, resize_height
    try:
        resize_height.configure(state='normal')
        resize.deiconify()
    except TclError:
        resize.mainloop()
        resize.deiconify()

def Resize1(event=None):
    global resize_width, resize_height, img, resize, allundo
    w, h = resize_width.get(), resize_height.get()
    if (len(w)==0 or int(w)<=0 or w.isdigit()==False) and (len(h)==0 or int(h)<=0 or h.isdigit()==False):
        from tkinter import messagebox
        messagebox.showinfo("Error", "Enter a proper value for width")
        ResizeW()
    elif (len(w) > 0 and int(w) > 0 or w.isdigit() == True) and (len(h)==0 or int(h)<=0 or h.isdigit()==False):

        w=int(w)
        h=(w/img.data.size[0])*(img.data.size[1])
        h = int(h)
        img=img.resize1(w,h)
        allundo.append(img)
        resize.withdraw()
        
        view(img)
    elif (len(h) > 0 and int(h) > 0 or h.isdigit() == True) and (len(w)==0 or int(w)<=0 or w.isdigit()==False):
        h = int(h)
        w = (h/img.data.size[1]) * (img.data.size[0])
        w = int(w)
        img = img.resize1(w, h)
        allundo.append(img)
        resize.withdraw()
        
        view(img)
    else:
        h = int(h)
        w = int(w)
        img = img.resize1(w, h)
        allundo.append(img)
        resize.withdraw()
        
        view(img)


def Resize(event=None):
    check()
    global resize_width, resize_height, img, resize, allundo
    w,h=resize_width.get(), resize_height.get()
    if (len(w)==0 and len(h)==0) or  (w.isdigit()==0 and h.isdigit()==0):
        from tkinter import messagebox
        messagebox.showinfo("Invalid input", "Enter at least one value")
        ResizeW()
    elif len(h)==0 or  h.isdigit()==0:
        basewidth = int(resize_width.get())
        img1 = img.data
        wpercent = (basewidth / float(img1.size[0]))
        hsize = int((float(img1.size[1]) * float(wpercent)))
        img1 = img1.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
        img.data=img1
        print(len(allundo))
        allundo.append(img1)
        print(len(allundo))
        resize.withdraw()
        
        view(img)
    elif len(w)==0 or w.isdigit()==0:
        basewidth = int(resize_height.get())
        img1 = img.data
        wpercent = (basewidth / float(img1.size[1]))
        hsize = int((float(img1.size[0]) * float(wpercent)))
        img1 = img1.resize((hsize, basewidth), PIL.Image.ANTIALIAS)
        img.data = img1
        allundo.append(img1)
        resize.withdraw()
        
        view(img)
    else:
        resize_height.configure(state='normal')
        img=img.fit(size=(int(resize_width.get()),int(resize_height.get())))
        allundo.append(img)
        resize.withdraw()
        
        view(img)



def kmeansW():
    global kmeansw
    try:
        kmeansw.deiconify()
    except TclError:
        kmeansw.mainloop()
        kmeansw.deiconify()

def kmeans(event=None):
    check()
    global kmeansw, img, kmeansk
    k=None
    try:
        k = int(kmeansk.get())

    except:
        from tkinter import messagebox
        messagebox.showinfo("Invalid input", "Enter a valid value ")
        kmeansW()
        return None

    if k==None or k<0:
        from tkinter import messagebox
        messagebox.showinfo("Invalid input", "Enter a valid number of clusters ")
        kmeansW()
        return None
    else:
        img = img.kmeans(K=k)
        allundo.append(img)
        kmeansw.withdraw()
        
        view(img)





def Zoom(event=None):
    check()
    global img, allundo, zoom_amount, zoom
    z=zoom_amount.get()
    img1 = img.zoom(z)
    zoom.withdraw()
    view(img1)

def Zoom_on_plus(event=None):
    global img, zoomval
    if zoomval>=1:
        zoomval+=0.2
    elif zoomval >= -1 and zoomval<1  :
        zoomval=1.2
    else:
        zoomval += 0.2
    img1 = img.zoom(zoomval)
    view(img1)
def Zoom_on_minus(event=None):
    global img, zoomval
    if zoomval<-1:
        zoomval-=0.2
    elif zoomval >-1 and zoomval<=1  :
        zoomval=-1.2
    else:
        zoomval-=0.2


    img1 = img.zoom(zoomval)
    view(img1)

def ZoomW(event=None):
    global zoom, zoom_amount
    zoom_amount.focus_set()
    try:
        zoom.deiconify()
    except TclError:
        zoom.mainloop()
        zoom.deiconify()

def Info(event=None):
    global info, filename, img, file_name, dir_name, pic_dim, pic_type
    pictypes={'jpg':'Joint Photographic Experts Group (JPEG)','JPG':'Joint Photographic Experts Group (JPEG)',
              'png':'Portable Network Graphics (PNG)', 'PNG':'Portable Network Graphics (PNG)',
              'BMP': 'Bitmap Image', 'bmp':'Bitmap Image',
              'pgm':'Portable Graymap Format (PGM)', 'PGM':'Portable Graymap Format (PGM)',
              'ppm':'Portable Pixmap Format (PPM)', 'PPM':'Portable Pixmap Format (PPM)',
              'gif':'Graphics Interchange Format (GIF)', 'GIF':'Graphics Interchange Format (GIF)',
              'tiff':'Tagged Image Format File (TIFF)', 'TIFF':'Tagged Image Format File (TIFF)'}
    info.focus_set()
    f=(filename[-1::-1]).partition("/")[0][-1::-1]
    d=(filename[-1::-1]).partition("/")[2][-1::-1]
    pic=Image()
    try:
        pic.getData(filename)
        h,w=pic.data.size
        pic_type.config(text=pictypes[(filename[-1::-1]).partition(".")[0][-1::-1]])
        pic_dim.config(text=str(w)+" X "+str(h))
        file_name.config(text=f)
        dir_name.config(text=d)
    except:
        pass
    try:
        info.deiconify()
    except TclError:
        info.mainloop()
        info.deiconify()

def houghL():
    check()
    global img, allundo
    img = img.houghLine()
    allundo.append(img)
    
    view(img)

def houghC():
    check()
    global img, allundo
    img = img.houghCircle()
    allundo.append(img)
    
    view(img)


def threshold():
    check()
    global img, allundo
    img = img.threshold()
    allundo.append(img)
    
    view(img)

def AdaptiveMean():
    check()
    global img, allundo
    img = img.adaptiveMeanThreshold()
    allundo.append(img)
    
    view(img)

def GaussianMean():
    check()
    global img, allundo
    img = img.adaptiveGaussianThreshold()
    allundo.append(img)
    
    view(img)

def brighten():
    check()
    global img, allundo
    img = img.brighten()
    allundo.append(img)
    
    view(img)

def darken():
    check()
    global img, allundo
    img = img.darken()
    allundo.append(img)
    view(img)

def Erode():
    check()
    global img, allundo
    img = img.erode()
    allundo.append(img)
    view(img)

def Dilate():
    check()
    global img, allundo
    img = img.dilate()
    allundo.append(img)
    view(img)

def MorphOpen():
    check()
    global img, allundo
    img = img.morph_open()
    allundo.append(img)
    view(img)

def MorphClose():
    check()
    global img, allundo
    img = img.morph_close()
    allundo.append(img)
    view(img)

def MorphGrad():
    check()
    global img, allundo
    img = img.morph_gradient()
    allundo.append(img)
    view(img)

def TopHat():
    check()
    global img, allundo
    img = img.morph_top_hat()
    allundo.append(img)
    view(img)

def BlackHat():
    check()
    global img, allundo
    img = img.morph_black_hat()
    allundo.append(img)
    view(img)

def original(event=None):
    global img, filename, zoomval
    zoomval=1
    img=Image()
    img.getData(filename)
    allundo.append(img)
    view(img)

def exit(event=None):
    global root
    root.quit()

def undo(event=None):
    global allundo, allredo, img
    if(len(allundo)>1):
        allredo.append(allundo.pop())
        img=allundo[-1]
        
        view(img)

def redo(event=None):
    global allundo, allredo, img
    if(len(allredo)>0):
        img = allredo[-1]
        allundo.append(allredo.pop())

        
        view(img)

class ResizingCanvas(Canvas):
    def __init__(self,parent,**kwargs):
        Canvas.__init__(self,parent,**kwargs)
        self.bind("<Configure>", self.on_resize)
        self.height = self.winfo_reqheight()
        self.width = self.winfo_reqwidth()

    def on_resize(self,event):
        # determine the ratio of old width/height to new width/height
        wscale = float(event.width)/self.width
        hscale = float(event.height)/self.height
        self.width = event.width
        self.height = event.height
        # resize the canvas
        self.config(width=self.width, height=self.height)
        # rescale allundo the objects tagged with the "allundo" tag
        self.scale("all",0,0,wscale,hscale)




root = Tk()

root.title("PixelView")

#w, h = root.winfo_screenwidth(), root.winfo_screenheight()
w,h=512,512
root.geometry("%dx%d+300+100" % (w, h))

## Key bindings
root.bind('<Control-z>', undo)
root.bind('<Control-r>', redo)
root.bind('<Control-s>', save)
root.bind('<Control-o>', open)
root.bind('<Control-q>', exit)
root.bind('<Control-O>', original)
root.bind('<Control-plus>', Zoom_on_plus)
root.bind('<Control-equal>', Zoom_on_plus)
root.bind('<Control-minus>', Zoom_on_minus)
root.bind('<Control-Left>', RotateL)
root.bind('<Control-Right>', RotateR)
root.bind('<Control-f>', Fit)
root.bind('<Control-i>', Info)


frame=Frame(root,width=root.winfo_reqwidth(),height=root.winfo_reqheight())
frame.pack(side=BOTTOM, expand=True,fill=BOTH)
canvas=ResizingCanvas(frame,bg='#000000',width=w,height=h,scrollregion=(0,0,w,h), highlightthickness=0)
hbar=Scrollbar(frame,orient=HORIZONTAL)
hbar.pack(side=BOTTOM,fill=X)
hbar.config(command=canvas.xview)
vbar=Scrollbar(frame,orient=VERTICAL)
vbar.pack(side=RIGHT,fill=Y)
vbar.config(command=canvas.yview)
canvas.config(width=w,height=h)
canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
canvas.addtag_all("all")
canvas.pack(side=LEFT, expand=True,fill=BOTH)



## Resize
resize=Tk()
resize.title("Resize")
resize.geometry('300x100+300+300')
Label(resize, text="Set size:   ", fg="black", font=("Verdana",12)).grid(row=0, pady=10, sticky=W)


resize_width= Entry(resize, width=12)
resize_width.grid(row=0, column=1, sticky=W)

resize_height=Entry(resize, width=12)
resize_height.grid(row=0, column=2, sticky=W, padx=5)
resize_ok_button=Button(resize, text="OK", width=8, command=Resize1)
resize_ok_button.grid(row=2, column=2, pady=5,  sticky=E)
#resize.bind('<Return>', Resize1)
resize_height.bind('<Return>', Resize1)
resize_width.bind('<Return>', Resize1)
resize.protocol("WM_DELETE_WINDOW", resize.withdraw)
resize.withdraw()
resize.resizable(width=False, height=False)


#info

info=Toplevel()
info.title("Image Properties")
info.geometry('500x300+300+300')
Label(info, text="File: ", fg="black", font=("Verdana 12 bold")).grid(row=0, pady=10, sticky=W)
file_name=Label(info, text=filename, fg="black", font=("Verdana 8"))
file_name.grid(row=1, pady=1, sticky=W)
Label(info, text="Location: ", fg="black", font=("Verdana 12 bold")).grid(row=2, pady=10, sticky=W)
dir_name=Label(info, text="", fg="black", font=("Verdana 8"))
dir_name.grid(row=3, pady=1, sticky=W)
Label(info, text="Size: ", fg="black", font=("Verdana 12 bold")).grid(row=4, pady=10, sticky=W)
pic_dim=Label(info, text="", fg="black", font=("Verdana 8"))
pic_dim.grid(row=5, pady=1, sticky=W)
Label(info, text="Type: ", fg="black", font=("Verdana 12 bold")).grid(row=6, pady=10, sticky=W)
pic_type=Label(info, text="", fg="black", font=("Verdana 8"))
pic_type.grid(row=7, pady=1, sticky=W)
info.protocol("WM_DELETE_WINDOW", info.withdraw)
info.withdraw()
info.resizable(width=False, height=False)


## Zoom
"""
zoom=Tk()
zoom.title("Zoom")
zoom.geometry('200x120+300+300')
zoom_amount = Scale(zoom, from_=10, to=-10, resolution=0.2)
zoom_amount.focus_set()
zoom_amount.grid(row=0, column=1, padx=20)
Button(zoom,text="OK", width=9, command=Zoom).grid(row=0, column=2, padx=30)
zoom.bind('<Return>', Zoom)
zoom.bind('<Escape>', lambda event: zoom.withdraw())
zoom.protocol("WM_DELETE_WINDOW", zoom.withdraw)
zoom.withdraw()
zoom.resizable(width=False, height=False)
"""

## kmeans
kmeansw=Tk()
kmeansw.title("k-means")
kmeansw.geometry('350x100+300+300')
Label(kmeansw, text="Number of clusters:   ", fg="black", font=("Verdana",12)).pack(side=LEFT)
kmeansk= Entry(kmeansw, width=12)
kmeansk.bind('<Return>', kmeans)
kmeansk.pack(side=LEFT)
Button(kmeansw, text="OK", width=8, command=kmeans).pack(side=RIGHT)
kmeansw.protocol("WM_DELETE_WINDOW", kmeansw.withdraw)
kmeansw.withdraw()
kmeansw.resizable(width=False, height=False)

## Remove border
removeborderW=Tk()
removeborderW.title("Remove border")
removeborderW.geometry('350x100+300+300')
Label(removeborderW, text="Number of pixels:   ", fg="black", font=("Verdana",12)).pack(side=LEFT)
borderpixels= Entry(removeborderW, width=12)
borderpixels.pack(side=LEFT)
borderpixels.bind('<Return>', removeBorder)
Button(removeborderW, text="OK", width=8, command=removeBorder).pack(side=RIGHT)
removeborderW.protocol("WM_DELETE_WINDOW", removeborderW.withdraw)
removeborderW.withdraw()
removeborderW.resizable(width=False, height=False)
## Menu bar
menubar = Menu(root)

filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="Open (Ctrl+O)", command=open)
filemenu.add_command(label="Save (Ctrl+S)", command=save)
filemenu.add_command(label="Properties (Ctrl+I)", command=Info)
filemenu.add_separator()
filemenu.add_command(label="Exit (Ctrl+Q)", command=exit)
menubar.add_cascade(label="File", menu=filemenu)


editmenu=Menu(menubar, tearoff=0)
editmenu.add_command(label="Undo (Ctrl+Z)", command=undo)
editmenu.add_command(label="Redo (Ctrl+R)", command=redo)
editmenu.add_command(label="Crop", command=Crop)
editmenu.add_command(label="Fit (Ctrl+F)", command=Fit)
editmenu.add_command(label="Resize", command=ResizeW)
editmenu.add_command(label="Remove Border", command=removeBorderW)


menubar.add_cascade(label="Edit", menu=editmenu)

viewmenu=Menu(menubar, tearoff=0)
viewmenu.add_command(label="Original (Ctrl+Shift+O)", command=original)
viewmenu.add_command(label="Zoom In (Ctrl +)", command=Zoom_on_plus)
viewmenu.add_command(label="Zoom Out (Ctrl -)", command=Zoom_on_minus)
viewmenu.add_command(label="Rotate Left (Ctrl+Left)", command=RotateL)
viewmenu.add_command(label="Rotate Right (Ctrl+Right)", command=RotateR)
viewmenu.add_command(label="Mirror", command=Mirror)
viewmenu.add_command(label="Flip", command=Flip)
viewmenu.add_command(label="Grayscale", command=gray)
menubar.add_cascade(label="View", menu=viewmenu)





imagemenu=Menu(menubar, tearoff=0)

blurmenu=Menu(imagemenu, tearoff=0)
blurmenu.add_command(label="Gaussian Blur", command=Gaussian)
blurmenu.add_command(label="Average Blur", command=Blur)
blurmenu.add_command(label="Median Blur", command=MedianBlur)
imagemenu.add_cascade(label="Blur", menu=blurmenu)
imagemenu.add_command(label="Reduce Noise", command=ReduceNoise)
effectsmenu=Menu(imagemenu, tearoff=0)
effectsmenu.add_command(label="Posterize", command=Posterize)
effectsmenu.add_command(label="Solarize", command=Solarize)
effectsmenu.add_command(label="Colourize", command=Colourize)
effectsmenu.add_command(label="Negative", command=Negative)
imagemenu.add_cascade(label="Effects", menu=effectsmenu)

clustermenu=Menu(imagemenu, tearoff=0)
clustermenu.add_command(label="k-means", command=kmeansW)
imagemenu.add_cascade(label="Clustering", menu=clustermenu)

imagemenu.add_command(label="Equalize", command=Equalize)
imagemenu.add_command(label="Auto-contrast", command=autocontrast)



imagemenu.add_command(label="Histogram", command=hist)

menubar.add_cascade(label="Image", menu=imagemenu)

filtermenu = Menu(menubar, tearoff=0)
edge_detectmenu=Menu(filemenu, tearoff=0)
edge_detectmenu.add_command(label="Prewitt X", command=PrewittX)
edge_detectmenu.add_command(label="Prewitt Y", command=PrewittY)
edge_detectmenu.add_command(label="Sobel X", command=SobelX)
edge_detectmenu.add_command(label="Sobel Y", command=SobelY)
edge_detectmenu.add_command(label="Robert's Cross X", command=RobertX)
edge_detectmenu.add_command(label="Robert's Cross Y", command=RobertY)
edge_detectmenu.add_command(label="Canny", command=Canny)
edge_detectmenu.add_command(label="Scharr derivative", command=Scharr)
edge_detectmenu.add_command(label="Sobel derivative", command=Sobel)
edge_detectmenu.add_command(label="Laplacian derivative", command=Laplacian)

thresholdmenu=Menu(menubar, tearoff=0)
thresholdmenu.add_command(label="Global threshold", command=threshold)
thresholdmenu.add_command(label="Adaptive Mean", command=AdaptiveMean)
thresholdmenu.add_command(label="Gaussian Mean", command=GaussianMean)


filtermenu.add_command(label="Sharpen", command=Sharpen)
filtermenu.add_command(label="Small Blur", command=SmallBlur)
filtermenu.add_command(label="Large Blur", command=LargeBlur)
filtermenu.add_cascade(label="Edge Detection", menu=edge_detectmenu)
filtermenu.add_cascade(label="Threshold", menu=thresholdmenu)
menubar.add_cascade(label="Filter", menu=filtermenu)

transformmenu=Menu(menubar, tearoff=0)

grayscalemenu=Menu(transformmenu, tearoff=0)
grayscalemenu.add_command(label="Brighten", command=brighten)
grayscalemenu.add_command(label="Darken", command=darken)
grayscalemenu.add_command(label="Log transformation", command=logT)
grayscalemenu.add_command(label="Power transformation", command=powerT)
grayscalemenu.add_command(label="Contrast Stretch", command=contrastS)
transformmenu.add_cascade(label="Grayscale Transformations", menu=grayscalemenu)

morphmenu=Menu(transformmenu, tearoff=0)
morphmenu.add_command(label="Erosion", command=Erode)
morphmenu.add_command(label="Dilation", command=Dilate)
morphmenu.add_command(label="Opening", command=MorphOpen)
morphmenu.add_command(label="Closing", command=MorphClose)
morphmenu.add_command(label="Gradient", command=MorphGrad)
morphmenu.add_command(label="Top Hat", command=TopHat)
morphmenu.add_command(label="Black Hat", command=BlackHat)
transformmenu.add_cascade(label="Morphological Transformations", menu=morphmenu)
houghmenu=Menu(transformmenu, tearoff=0)

houghmenu.add_command(label="Hough Lines", command=houghL)
houghmenu.add_command(label="Hough Circles", command=houghC)
transformmenu.add_cascade(label="Hough Transformations", menu=houghmenu)
menubar.add_cascade(label="Transform", menu=transformmenu)

helpmenu=Menu(menubar, tearoff=0)
helpmenu.add_command(label="About", command=About)
menubar.add_cascade(label="Help", menu=helpmenu)
about = Toplevel()
about.geometry('340x200+400+200')
about.title("About PixelView")
about.config(background='white')
i=None
try:
    path_name=os.path.dirname('__file__')
    j=PILimage.open(str(path_name)+"img/abt.jpg")
    i=ImageTk.PhotoImage(j)
except:pass
frame1=Frame(about, width=100)
frame1.grid(row=0, sticky=W)
frame2=Frame(about)
frame2.grid(row=0, sticky=W, padx=120)
Label(frame1,bg='white', text="", fg="black",   width=100, image=i,font=("Verdana",18)).grid(row=0, column=0, sticky=W)
Label(frame2,bg='white', text="PixelView            ",  fg="black", font=("Verdana",18)).grid(row=0, sticky=W)
Label(frame2, bg='white',text="Version 0.8.8                          ", fg="black",  font=("Verdana",10)).grid(row=1, sticky=W)
Label(frame2, bg='white',text="                          ", fg="black",  font=("Verdana",18)).grid(row=2, sticky=W)
Label(frame2, bg='white', text="Coded by: Akshay Anand                  ", fg="black", font=("Verdana",10)).grid(row=3)
Label(frame2, bg='white', text="Mentor:   Amiya Halder                     ", fg="black", font=("Verdana",10)).grid(row=4)
Label(frame2, bg='white', text="                                                   ", fg="black", font=("Verdana",10)).grid(row=5)
Label(frame2, bg='white', text="www.akshayjha.co.nr                              ", fg="black", font=("Verdana",9)).grid(row=6)
Label(frame2, bg='white', text="© Copyright 2017 Akshay Anand                 ", fg="black", font=("Verdana",8)).grid(row=7)

about.protocol("WM_DELETE_WINDOW", about.withdraw)
about.withdraw()
about.resizable(width=False, height=False)

try:
    root.iconbitmap (default='img/favicon.ico')
except:
    pass

root.config(menu=menubar, background='black')
root.protocol("WM_DELETE_WINDOW", exit)
root.mainloop()