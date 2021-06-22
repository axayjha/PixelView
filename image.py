#!/usr/bin/env python
# ‐*‐ coding: utf‐8 ‐*‐
# coded in Python3.x

"""

@author: Akshay Anand
<https://akshayjha.tech>
<www.GitHub.com/AxayJha>
<akshayjha@live.in>
Last modified: 28/04/2017

"""

import math
import numpy as np
import cv2
from PIL import Image as PILimage
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import PIL.ImageOps


class Image(object):

    def __init__(self, data=None):
        self.data = data
        self.height = 0
        self.width = 0
        if type(data)==str:
            self.getData(data)


    def getData(self, filein):
        """
        Gets and sets image data
        Using Pil.Image

        :param filein: string containing the filename
        :return: PIL image file
        """
        try:
            im=None
            if ".pgm" in filein or ".PGM" in filein:
                pgmf=cv2.imread(filein,-1)
                im=PILimage.fromarray(pgmf)
            else:
                im = PILimage.open(filein)
            try:
                self.data=im.convert('RGB')
            except:
                self.data=im
            self.height, self.width = self.data.size
        except FileNotFoundError:
            print("Error accessing the file.")

    def save(self, fileout):
        """
        Saves the image file in user-specified file format

        :param fileout: Output Image file name
        :return: Final Image File
        """
        self.data.save(fileout)

    def grayscale(self):
        return Image(self.data.convert('L').convert('RGB'))

    def negative(self):
        inverted_image = PIL.ImageOps.invert(self.data)
        return Image(inverted_image)

    def autocontrast(self):
        """
            From Pillow:
            Maximize (normalize) image contrast. This function calculates a
            histogram of the input image, removes **cutoff** percent of the
            lightest and darkest pixels from the histogram, and remaps the image
            so that the darkest pixel becomes black (0), and the lightest
            becomes white (255).

            :param image: The image to process.
            :param cutoff: How many percent to cut off from the histogram.
            :param ignore: The background pixel value (use None for no background).
            :return: An image.
            """
        result=PIL.ImageOps.autocontrast(self.data)
        return Image(result)

    def colourize(self, black=(10,30,50), white=(200,70,80)):
        """
            From Pillow:
            Colorize grayscale image.  The **black** and **white**
            arguments should be RGB tuples; this function calculates a color
            wedge mapping all black pixels in the source image to the first
            color, and all white pixels to the second color.

            :param image: The image to colorize.
            :param black: The color to use for black input pixels.
            :param white: The color to use for white input pixels.
            :return: An image.
        """
        self.data = self.data.convert('L')
        result=PIL.ImageOps.colorize(self.data, black, white)
        return Image(result)

    def removeBorder(self, border):
        """
            Remove border from image.  The same amount of pixels are removed
            from all four sides.  This function works on all image modes.


            :param image: The image to crop.
            :param border: The number of pixels to remove.
            :return: An image.
        """
        result=Image()
        result.data=(PIL.ImageOps.crop(self.data, border))
        return result

    def equalize(self, mask=None):
        """
        Equalize the image histogram. This function applies a non-linear
        mapping to the input image, in order to create a uniform
        distribution of grayscale values in the output image.

        :param image: The image to equalize.
        :param mask: An optional mask.  If given, only the pixels selected by
                     the mask are included in the analysis.
        :return: An image.
        """
        result=Image()
        result.data=PIL.ImageOps.equalize(self.data)
        return result

    def fit(self, size=(512,512), method=None, bleed=None, param=None):
        """
        Returns a sized and cropped version of the image, cropped to the
        requested aspect ratio and size.

        This function was contributed by Kevin Cazabon.

        :param size: The requested output size in pixels, given as a
                     (width, height) tuple.
        :param method: What resampling method to use. Default is
                       :py:attr:`PIL.Image.NEAREST`.
        :param bleed: Remove a border around the outside of the image (from all
                      four edges. The value is a decimal percentage (use 0.01 for
                      one percent). The default value is 0 (no border).
        :param centering: Control the cropping position.  Use (0.5, 0.5) for
                          center cropping (e.g. if cropping the width, take 50% off
                          of the left side, and therefore 50% off the right side).
                          (0.0, 0.0) will crop from the top left corner (i.e. if
                          cropping the width, take all of the crop off of the right
                          side, and if cropping the height, take all of it off the
                          bottom).  (1.0, 0.0) will crop from the bottom left
                          corner, etc. (i.e. if cropping the width, take all of the
                          crop off the left side, and if cropping the height take
                          none from the top, and therefore all off the bottom).
        :return: An image.
        """

        return Image(PIL.ImageOps.fit(self.data, size))

    def zoom(self, z):
        """
        Returns the same image in different size depending on the
        value of z, maintaining the aspect ratio.


        :param z: zoom magnitude -
                  either >1 (for zooming in) or < -1 (for zooming out)
        :return: Zoomed image
        """
        if z>=1:
            w=round(self.data.size[0]*z)
            h=round(self.data.size[1]*z)
            if w<1: w=1
            if h<1: h=1
            return self.fit(size=(w,h))
        elif z<-1:
            w = round(self.data.size[0] / (-1 * z))
            h = round(self.data.size[1] / (-1 * z))
            if w < 1: w = 1
            if h < 1: h = 1
            return self.fit(size=(w, h))
        else:
            return Image(self.data)




    def flip(self):
        """
        Flips the image vertically (top to bottom).

        :param image: The image to flip.
        :return: An image.
        """

        return Image(self.data.transpose(PILimage.FLIP_TOP_BOTTOM))

    def mirror(self):
        """
        Flip image horizontally (left to right).

        :param image: The image to mirror.
        :return: An image.
        """

        return Image(self.data.transpose(PILimage.FLIP_LEFT_RIGHT))

    def posterize(self, bits=2):
        """
        Reduce the number of bits for each color channel.

        :param image: The image to posterize.
        :param bits: The number of bits to keep for each channel (1-8).
        :return: An image.
        """
        return Image(PIL.ImageOps.posterize(self.data, bits))

    def solarize(self, threshold=128):
        """
        Invert all pixel values above a threshold.

        :param image: The image to solarize.
        :param threshold: All pixels above this greyscale level are inverted.
        :return: An image.
        """
        return Image(PIL.ImageOps.solarize(self.data, threshold))

    #Image blurring/smoothing
    def blur(self):
        """
            Averages the image using box filer
            :return: An image
        """
        result = cv2.blur(np.array(self.data), (5, 5), 0)
        return Image(PILimage.fromarray(result))

    def gaussian_blur(self):
        """
            Averages the image using Gaussian kernel
            :return: An image
        """
        blur = cv2.GaussianBlur(np.array(self.data), (5, 5), 0)
        return Image(PILimage.fromarray(blur))

    def median_blur(self):
        """
         takes median of all the pixels under kernel area and
         central element is replaced with this median value
         :return: An image
         """
        median = cv2.medianBlur(np.array(self.data), 5)
        return Image(PILimage.fromarray(median))

    def bilateral_blur(self):
        """
        Reduces Noise
        cv2.bilateralFilter() is highly effective in noise removal
        while keeping edges sharp. But the operation is slower compared to other filters.
        :return: An image
        """

        blur = cv2.bilateralFilter(np.array(self.data), 9, 75, 75)
        return Image(PILimage.fromarray(blur))

    def sobel(self, m=0, kernel=3):
        """
        Sobel edge detection filtering
        Detects the edges in either horizontal or vertical direction

        :param m: Mode. 0 means sobelX and 1 and means sobelY
        :param kernel: size of kernel to use to mask
        :return: An image

        """
        x,y=None, None
        if m==0:
            x,y = 1,0
        elif m==1:
            x,y = 0,1

        img = self.data.convert('L')

        #sobelx64f = cv2.Sobel(np.array(self.data), cv2.CV_64F, x, y, ksize=kernel)
        #abs_sobel64f = np.absolute(sobelx64f)
        #sobel_8u = np.uint8(abs_sobel64f)
        result=PILimage.fromarray(cv2.Sobel(np.array(img), cv2.CV_64F, x, y, ksize=kernel)).convert('RGB')

        return Image(result)

    def sobel_grad(self, kernel=3):
        """
        Sobel filtering derivative
        Returns the derivative of sobel x and y direction gradients

        :param kernel: size of kernel to use to mask the image
        :return: An image
        """
        img = self.gaussian_blur().data.convert('L')

        grad_x = cv2.Sobel(np.array(img), cv2.CV_64F, 1,0, ksize=kernel)
        grad_y = cv2.Sobel(np.array(img), cv2.CV_64F, 0,1, ksize=kernel)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        result = cv2.add(abs_grad_x, abs_grad_y)
        result = PILimage.fromarray(result).convert('RGB')

        return Image(result)



    def scharr(self, kernel=3):
        """
         Returns the Scharr derivative of the image
         Scharr operator is used to find image gradients(or edge detection).
         This operator uses two kernels to convolve the image and calculate derivatives
         in two directions. The derivatives track changes in horizontal as well as
         vertical directions. Scharr operator tries to overcome Sobel operator’s
         drawback of not having perfect rotational symmetry.

        :param kernel: size of kernel to use to mask the image
        :return: An image
        """
        img = self.gaussian_blur().data.convert('L')

        grad_x = cv2.Scharr(np.array(img), cv2.CV_64F, 1, 0)
        grad_y = cv2.Scharr(np.array(img), cv2.CV_64F, 0, 1)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        result= cv2.add(abs_grad_x, abs_grad_y)
        result = PILimage.fromarray(result).convert('RGB')

        return Image(result)


    def laplacian(self):
        """
        Returns Laplacian derivative of the given image
        Runs the laplacian filter over the image and returns the masked image

        :return: An Image
        """
        img = self.data.convert('L')
        result = cv2.Laplacian(np.array(img), cv2.CV_64F)
        return Image(PILimage.fromarray(result).convert('RGB'))

    def laplacian_sharpening(self):
        """
        Returns the sharpened image

        :return: An Image
        """
        img=self.laplacian()
        result=np.array(img.data)+np.array(self.data)
        return Image(PILimage.fromarray(result).convert('RGB'))

    def canny(self):
        temp=self.data
        self.data = self.data.convert('L')
        result=cv2.Canny(np.array(self.data), 50, 240)
        self.data=temp
        return Image(PILimage.fromarray(result))

    def prewitt(self, x=0):
        img = self.data.convert('L')

        kernelY=np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        kernelX=np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
        kernels={0:kernelX, 1:kernelY}
        try:
            result = cv2.filter2D(np.array(img), -1, kernels[x])
        except:
            result = cv2.filter2D(np.array(img), -1, kernels[1])
        #result=(np.sqrt(np.square(x_grad) + np.square(y_grad)))
        #result =result.astype(int)
        #result=rescale_intensity(result, in_range=(0, 255))
        result.clip(0,255)
        return Image(PILimage.fromarray(result))




    def roberts(self, x=0):
        temp=self.data
        self.data = self.data.convert('L')
        kernelY=np.array([[1,0],[0,-1]])
        kernelX=np.array([[0,1],[-1,0]])
        kernels={0:kernelX, 1:kernelY}
        try:
            result = cv2.filter2D(np.array(self.data), -1, kernels[x])
        except:
            result = cv2.filter2D(np.array(self.data), -1, kernels[1])
        self.data=temp
        return Image(PILimage.fromarray(result))

    def largeBlur(self):
        largeblur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))
        #img = self.data.convert('L')
        result = cv2.filter2D(np.array(self.data), -1, largeblur)
        return Image(PILimage.fromarray(result))




    def smallBlur(self):
        smallblur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
        #img = self.data.convert('L')
        result = cv2.filter2D(np.array(self.data), -1, smallblur)
        return Image(PILimage.fromarray(result))


    def sharpen(self):
        Sharpen = np.array((
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]), dtype="int")

        #self.data = self.data.convert('L')
        result = cv2.filter2D(np.array(self.data), -1, Sharpen)

        return Image(PILimage.fromarray(result))

    def histogram(self):
        img=np.array(self.data)
        color = ('b', 'g', 'r')
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        fig.canvas.set_window_title('Histogram')
        plt.ylabel('Frequency')
        plt.xlabel('Intensity')
        for i, col in enumerate(color):
            histr = cv2.calcHist([img], [i], None, [256], [0, 256])
            ax.plot(histr, color=col, lw=0.8)
            plt.xlim([0, 256])
        plt.show()

    def log(self, c=20):
        img = self.data.convert('L')
        mat = np.array(img)
        mat= c*np.log(mat)
        mat=mat.astype(int)
        mat=np.clip(mat, 0, 255)
        return Image(PILimage.fromarray(mat).convert('RGB'))

    def power(self, c=0.065, l=1.475):
        img = self.data.convert('L')
        mat = np.array(img)
        mat = c * (mat**l)
        mat = mat.astype(int)
        mat = np.clip(mat, 0, 255)
        return Image(PILimage.fromarray(mat).convert('RGB'))

    def contrastStretch(self):
        img=np.array(self.data.convert('L'))
        equ = cv2.equalizeHist(img)
        #res = np.hstack((img, equ))
        return Image(PILimage.fromarray(equ).convert('RGB'))

    def contrastStretch1(self, c=13, d=242):
        img = self.data.convert('L')
        a,b=0,255
        mat = np.array(img)
        mat = np.array(img)
        for i in range(self.data.size[0]-1):
            for j in range(self.data.size[1]-1):
                mat[i][j] = ((mat[i][j]-c)*(b-a)//(d-c))+a
                if mat[i][j] < 0:
                    mat[i][j] = 0
                elif mat[i][j] > 255:
                    mat[i][j] = 255
        return Image(PILimage.fromarray(mat).convert('RGB'))

    def kmeans(self, K=4):
        img = np.array(self.data)
        try:
            Z = img.reshape((-1, 3))
        except:
            Z = img.reshape((-1, 2))

        # convert to np.float32
        Z = np.float32(Z)

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))
        return Image(PILimage.fromarray(res2).convert('RGB'))

    def rotateLeft(self):
        return Image(self.data.rotate(90))
    def rotateRight(self):
        return Image(self.data.rotate(-90))

    def rotate_image(self, angle):
        mat=np.array(self.data)
        height, width = mat.shape[:2]
        image_center = (width / 2, height / 2)

        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

        radians = math.radians(angle)
        sin = math.sin(radians)
        cos = math.cos(radians)
        bound_w = int((height * abs(sin)) + (width * abs(cos)))
        bound_h = int((height * abs(cos)) + (width * abs(sin)))

        rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
        rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])

        rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
        return Image(PILimage.fromarray(rotated_mat).convert('RGB'))

    def houghCircle(self):
        try:
            img=np.array(self.data.convert('L'))
            img = cv2.medianBlur(img, 5)
            cimg = np.array(self.data.convert('RGB'))


            circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                                       param1=50, param2=38, minRadius=0, maxRadius=300)

            circles = np.uint16(np.around(circles))
            count=0
            for i in circles[0, :]:
                count+=1
                if(count>8): break
                # draw the outer circle
                cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # draw the center of the circle
                cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

            return Image(PILimage.fromarray(cimg).convert('RGB'))
        except:
            return Image(self.data)


    def houghLine(self):
        img=np.array(self.data.convert('RGB'))
        gray = np.array(self.data.convert('L'))
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        minLineLength = 1
        maxLineGap = 1
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)

        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return Image(PILimage.fromarray(img).convert('RGB'))

    def brighten(self):

        from pylab import array, plot, show, axis, arange, figure, uint8
        # Image data
        image = np.array(self.data.convert('L')) # load as 1-channel 8bit grayscale

        maxIntensity = 255.0  # depends on dtype of image data
        # Parameters for manipulating image data
        phi = 1
        theta = 1
        # Increase intensity such that
        # dark pixels become much brighter,
        # bright pixels become slightly bright
        newImage0 = (maxIntensity / phi) * (image / (maxIntensity / theta)) ** 0.5
        newImage0 = array(newImage0, dtype=uint8)
        return Image(PILimage.fromarray(newImage0).convert('RGB'))




    def darken(self):
        from pylab import array,  arange, uint8

        # Image data
        image = np.array(self.data.convert('L'))  # load as 1-channel 8bit grayscale

        maxIntensity = 255.0  # depends on dtype of image data


        # Parameters for manipulating image data
        phi = 1
        theta = 1

        # Decrease intensity such that
        # dark pixels become much darker,
        # bright pixels become slightly dark
        newImage1 = (maxIntensity / phi) * (image / (maxIntensity / theta)) ** 2
        newImage1 = array(newImage1, dtype=uint8)
        return Image(PILimage.fromarray(newImage1).convert('RGB'))


    def crop(self):
        import matplotlib.widgets as widgets

        def onselect(eclick, erelease):
            if eclick.ydata > erelease.ydata:
                eclick.ydata, erelease.ydata = erelease.ydata, eclick.ydata
            if eclick.xdata > erelease.xdata:
                eclick.xdata, erelease.xdata = erelease.xdata, eclick.xdata
            ax.set_ylim(erelease.ydata, eclick.ydata)
            ax.set_xlim(eclick.xdata, erelease.xdata)
            fig.canvas.draw()

        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(111)
        ax.set_axis_off()

        im = self.data
        arr = np.asarray(im)
        fig.gca().set_axis_off()
        fig.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
        plt.margins(0, 0)
        fig.gca().xaxis.set_major_locator(plt.NullLocator())
        fig.gca().yaxis.set_major_locator(plt.NullLocator())
        plt_image = plt.imshow(arr)
        rs = widgets.RectangleSelector(
            ax, onselect, drawtype='box',
            rectprops=dict(facecolor='red', edgecolor='black', alpha=0.5, fill=True))

        plt.show()


    def threshold(self):
        img=np.array(self.data.convert('L'))
        l,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        return Image(PILimage.fromarray(img).convert('RGB'))

    def adaptiveMeanThreshold(self):
        img = np.array(self.data.convert('L'))
        img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
        return Image(PILimage.fromarray(img).convert('RGB'))

    def adaptiveGaussianThreshold(self):
        img = np.array(self.data.convert('L'))
        img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        return Image(PILimage.fromarray(img).convert('RGB'))


    def resize1(self, sizex, sizey):
        im1=None
        im = cv2.resize(np.array(self.data), (sizex, sizey), im1, 0, 0)
        return Image(PILimage.fromarray(im).convert('RGB'))


    def convolve1(self, kernelx, kernely):
        # grab the spatial dimensions of the image, along with
        # the spatial dimensions of the kernel
        kernelx=np.array(kernelx)
        kernely=np.array(kernely)
        image=np.array(self.data)
        (iH, iW) = image.shape[:2]
        (kH, kW) = kernelx.shape[:2]

        # allocate memory for the output image, taking care to
        # "pad" the borders of the input image so the spatial
        # size (i.e., width and height) are not reduced
        pad = int((kW - 1) / 2)
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        output = np.zeros((iH, iW), dtype="float32")
        # loop over the input image, "sliding" the kernel across
        # each (x, y)-coordinate from left-to-right and top to
        # bottom
        for y in np.arange(pad, iH + pad):
            for x in np.arange(pad, iW + pad):
                # extract the ROI of the image by extracting the
                # *center* region of the current (x, y)-coordinates
                # dimensions
                roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

                # perform the actual convolution by taking the
                # element-wise multiplicate between the ROI and
                # the kernel, then summing the matrix
                k = (roi * kernelx).sum()
                l = (roi * kernely).sum()
                g=np.sqrt(k**2 + l**2)

                # store the convolved value in the output (x,y)-
                # coordinate of the output image
                output[y - pad, x - pad] = g
        # rescale the output image to be in the range [0, 255]
        output=output.clip(0,255)
        #utput = (output * 255).astype("uint8")

        # return the output image
        return Image(PILimage.fromarray(output).convert('L'))



    def convolve(self, kernelX, kernelY):
        """
        Returns the resultant derivative of the gradients obtained
        by spatial filtering the image with the kernels of corresponding
        direction.
        :param kernelX: Kernel to get gradient image in x direction
        :param kernelY: Kernel to get gradient image in y direction
        :return: Filtered image
        """
        # convert the image to grayscale
        img=np.array(self.data.convert('L'))
        # the kernels to mask the image with
        # using them as numpy array for faster
        # and easier operations
        kernelX=np.array(kernelX)
        kernelY=np.array(kernelY)
        # getting the dimensions of the input image
        w,h=self.data.size[0], self.data.size[1]
        # padding the image on all boundaries
        padded_img=np.zeros(shape=(w+2, h+2))
        padded_img[1:-1, 1:-1] = img
        # allocating memory for output image
        result=np.zeros(shape=(w,h))

        # moving the kernel over each pixel
        for i in range(1,w+1):
            for j in range(1,h+1):
                Gx = 0 #gradient in x dir
                Gy = 0 #gradient in y dir
                # obtaining and storing the response of filter at each point
                for k in range(-1,2):
                    for l in range(-1,2):
                        # calculating the gradient of the image in x and y direction
                        Gx += (padded_img[i + k][j + l] * kernelX[k + 1][l + 1])
                        Gy += (padded_img[i + k][j + l] * kernelY[k + 1][l + 1])

                # calculating the resulatant derivative
                grad=int(np.sqrt(Gx**2)+np.sqrt(Gy**2))
                # making sure the pixel intensity value remains between 0 and 255
                if grad>255: grad=255
                elif grad < 0 : grad = 0
                result[i-1][j-1]=grad
        #returning the final image
        return Image(PILimage.fromarray(result).convert('RGB'))

    def prewitt_grad(self):
        preX = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]
        preY = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
        return self.convolve(preX, preY)


    def prewitt1(self, x=0):
        """doesn't work as desired"""
        img = self.data.convert('L')

        kernelY=np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        kernelX=np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
        gx = cv2.filter2D(np.array(img), -1, kernelX)

        gy = cv2.filter2D(np.array(img), -1, kernelY)
        result=np.sqrt(gx**2 + gy**2)
        result=result.astype(int)

        #result=(np.sqrt(np.square(x_grad) + np.square(y_grad)))
        #result =result.astype(int)
        #result=rescale_intensity(result, in_range=(0, 255))
        result.clip(0,255)
        return Image(PILimage.fromarray(result).convert('L'))

    def prewitt2(self):
        """doesn't work as desired"""
        img = self.data.convert('L')
        img_gaussian= Image(img).gaussian_blur()
        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
        img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
        abs_grad_x = cv2.convertScaleAbs(img_prewittx)
        abs_grad_y = cv2.convertScaleAbs(img_prewitty)
        result = cv2.add(abs_grad_x, abs_grad_y)
        return Image(PILimage.fromarray(result).convert('RGB'))


    def erode(self):
        kernel = np.ones((5, 5), np.uint8)
        result = cv2.erode(np.array(self.data), kernel, iterations=1)
        return Image(PILimage.fromarray(result).convert('RGB'))

    def dilate(self):
        kernel = np.ones((5, 5), np.uint8)
        result = cv2.dilate(np.array(self.data), kernel, iterations=1)
        return Image(PILimage.fromarray(result).convert('RGB'))

    def morph_open(self):
        """Morphological opening: erosion followed by dilation"""
        kernel = np.ones((5, 5), np.uint8)
        result = cv2.morphologyEx(np.array(self.data), cv2.MORPH_OPEN, kernel)
        return Image(PILimage.fromarray(result).convert('RGB'))

    def morph_close(self):
        """Morphological closing: dilation followed by erosion"""
        kernel = np.ones((5, 5), np.uint8)
        result = cv2.morphologyEx(np.array(self.data), cv2.MORPH_CLOSE, kernel)
        return Image(PILimage.fromarray(result).convert('RGB'))

    def morph_gradient(self):
        """Morphological gradient: difference between dilation and erosion"""
        kernel = np.ones((5, 5), np.uint8)
        result = cv2.morphologyEx(np.array(self.data), cv2.MORPH_GRADIENT, kernel)
        return Image(PILimage.fromarray(result).convert('RGB'))

    def morph_top_hat(self):
        """ Morphological top hat:
        difference between input image and Opening of the image"""
        kernel = np.ones((9, 9), np.uint8)
        result = cv2.morphologyEx(np.array(self.data), cv2.MORPH_TOPHAT, kernel)
        return Image(PILimage.fromarray(result).convert('RGB'))

    def morph_black_hat(self):
        """Morphological black hat:
        difference between the closing of the input image and input image"""
        kernel = np.ones((9, 9), np.uint8)
        result = cv2.morphologyEx(np.array(self.data), cv2.MORPH_BLACKHAT, kernel)
        return Image(PILimage.fromarray(result).convert('RGB'))

    def gen_convolve(self, kernelX, kernelY=np.zeros(shape=(3,3))):
        """
        Generalized convolution function:
        Returns the resultant derivative of the gradients obtained
        by spatial filtering the image with the kernels of corresponding
        direction.
        :param kernelX: Kernel to get gradient image in x direction
        :param kernelY: Kernel to get gradient image in y direction
        :return: Filtered image
        """
        # convert the image to grayscale
        img=np.array(self.data.convert('L'))
        # the kernels to mask the image with
        # using them as numpy array for faster
        # and easier operations
        kernelX=np.array(kernelX)
        kernelY=np.array(kernelY)
        # getting the dimensions of the input image
        w,h=self.data.size[0], self.data.size[1]
        # padding the image on all boundaries
        pad=(kernelX.shape[0]-1)//2
        padded_img=np.zeros(shape=(h+2*pad, w+2*pad))
        padded_img[1*pad:-1*pad, 1*pad:-1*pad] = img
        # allocating memory for output image
        result=np.zeros(shape=(h,w))
        laplacian= (kernelY.sum() == 0)
        if laplacian:
            kernelY=np.zeros(shape=(kernelX.shape[0], kernelX.shape[1]))
        # moving the kernel over each pixel
        for i in range(pad, h+pad):
            for j in range(pad, w+pad):
                Gx = 0 #gradient in x dir
                Gy = 0 #gradient in y dir
                # obtaining and storing the response of filter at each point
                for k in range(-pad, pad+1):
                    for l in range(-pad, pad+1):
                        # calculating the gradient of the image in x and y direction
                        Gx += (padded_img[i + k][j + l] * kernelX[k + 1][l + 1])
                        Gy += (padded_img[i + k][j + l] * kernelY[k + 1][l + 1])

                # calculating the resulatant derivative
                if laplacian : grad=Gx
                else:
                    grad=int(np.sqrt(Gx**2)+np.sqrt(Gy**2))
                # making sure the pixel intensity value remains between 0 and 255
                if grad>255: grad=255
                elif grad < 0 : grad = 0
                result[i-pad][j-pad]=grad
        #returning the final image
        return Image(PILimage.fromarray(result).convert('L'))

    def clip_range(self, cutoff=20):
        """
            Maximize (normalize) image contrast.
            (Grayscale operation)
            This function calculates a
            histogram of the input image, removes **cutoff** percent of the
            lightest and darkest pixels from the histogram, and remaps the image
            so that the darkest pixel becomes black (0), and the lightest
            becomes white (255).

            :param image: The image to process.
            :param cutoff: What intensity range to cut off from both higher and lower ends
            :return: An image (grayscale in RGB form)
        """
        # convert the image to grayscale
        img=np.array(self.data.convert('L'))
        # getting the dimensions of the input image
        w,h=self.data.size[0], self.data.size[1]
        # padding the image on all boundaries

        for i in range(h):
            for j in range(w):
                if img[i][j] < cutoff:
                    img[i][j]= 0
                elif img[i][j] > 255-cutoff:
                    img[i][j] = 255
        #returning the final image
        return Image(PILimage.fromarray(img).convert('RGB'))


    def normalize_contrast(self, newrange=(20,220)):
        """
            Maximize (normalize) image contrast.
            (Grayscale operation)
            This function calculates a
            histogram of the input image, removes **cutoff** percent of the
            lightest and darkest pixels from the histogram, and remaps the image
            so that the darkest pixel becomes black (0), and the lightest
            becomes white (255).

            :param image: The image to process.
            :param cutoff: What intensity range to cut off from both higher and lower ends
            :return: An image (grayscale in RGB form)
        """
        # convert the image to grayscale
        img=np.array(self.data.convert('L'))
        # getting the dimensions of the input image
        w,h=self.data.size[0], self.data.size[1]

        low=newrange[0]
        high=newrange[1]

        for i in range(h):
            for j in range(w):
                img[i][j] = (img[i][j]-low)*(high-low)/255 + low
        #returning the final image
        return Image(PILimage.fromarray(img).convert('RGB'))
