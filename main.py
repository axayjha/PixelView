from image import *
from globals import *
Image("inputs/lena.jpg").normalize_contrast().save("outputs/norm.jpg")

#im.gen_convolve(np.ones((5,5), np.uint8)).save("outputs/erode.jpg")
#im.laplacian_sharpening().save("outputs/sharp2.jpg")
