# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# img = cv2.imread('/home/sumedh/Machine Learning/Mask_RCNN/images/cool3.jpg')
# cv2.imshow("cool2",img)
# blur = cv2.GaussianBlur(img,(39,39),0)
# cv2.imshow("Cool",blur)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
from PIL import Image,ImageFilter
import cv2
# bck = Image.open("background.png")
# fgk = Image.open("foreground.png")
# Image.alpha_composite(bck,fgk).save("test.png")
# #blended = Image.blend(bck,fgk,alpha=0.8)

def callling():
    foreground = cv2.imread("foreground.png")

    background_original = cv2.imread("/home/sumedh/Machine Learning/Mask_RCNN/images/cool.jpg")
    background = cv2.GaussianBlur(background_original,(15,15),0)
    # blur_original = cv2.GaussianBlur(image1,(49,49),0)
    alpha = cv2.imread("mask.png")

    # Convert uint8 to float
    foreground = foreground.astype(float)
    background = background.astype(float)

    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha.astype(float) / 255

    # Multiply the foreground with the alpha matte
    foreground = cv2.multiply(alpha, foreground)

    # Multiply the background with ( 1 - alpha )
    background = cv2.multiply(1.0 - alpha, background)

    # Add the masked foreground and background.
    outImage = cv2.add(foreground, background)
    cv2.imwrite("out.png",outImage)

    # Display image
    cv2.imshow("outImg", outImage / 255)

    cv2.waitKey(0)