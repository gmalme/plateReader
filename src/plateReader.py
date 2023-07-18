import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr
from PIL import Image
import PIL

class plateReader:
    def init(self):
        pass

    def print(self, old_image, new_image, arg_a ='Imagem Original', arg_b ='Imagem Resultante'):
        # Exibir a imagem original e a imagem equalizada ( equalização )
        plt.subplot(1, 2, 1)
        plt.imshow(old_image, cmap='gray')
        plt.title(arg_a)
        plt.subplot(1, 2, 2)
        plt.imshow(new_image, cmap='gray')
        plt.title(arg_b)
        plt.show()

    def region_colorizer(self, labels):
        label_hue = np.uint8(255*labels/np.max(labels))
        white_ch = 255*np.ones_like(label_hue)
        color_image = cv2.merge([label_hue, white_ch, white_ch])
        color_image = cv2.cvtColor(color_image, cv2.COLOR_HSV2BGR)
        color_image[label_hue==0] = 0
        return color_image

    def resize_image(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]
        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))
        return cv2.resize(image, dim, interpolation=inter)

    def read_image(self, image):
        reader = easyocr.Reader(['en'])
        result = reader.readtext(image)
        text = result[0][-2]
        return text

    def segment_image(self, image):
        # otimização 
        image = cv2.GaussianBlur(image,(3,3),3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        image = cv2.morphologyEx(image,cv2.MORPH_CLOSE,kernel,iterations=3)

        # Binarizacao
        _, image = cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        # Erosao
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
        image = cv2.erode(image,kernel)

        # dilatacao
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(6,6)) 
        image = cv2.dilate(image,kernel)

        # extracao elementos conectados
        connectivity = 4
        (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(image , connectivity , cv2.CV_32S)
        regioesColoridas = self.region_colorizer(labels)

        return regioesColoridas

    def process_image(self):
        image = cv2.imread("input/17.png")

        image = self.resize_image(image, width=300)
        gray_image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        segment_image = self.segment_image(gray_image)

        result = self.read_image(segment_image)
        for palavra in result:
            print(palavra)

        self.print(gray_image, segment_image)
