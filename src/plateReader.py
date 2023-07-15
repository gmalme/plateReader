import cv2
import numpy as np

class plateReader:
    def init(self):
        pass
    
    def maintain_image_size(self, image, width=None, height=None, inter=cv2.INTER_AREA):
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

    def segment_image(self, image):

        #fechamento
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        closing = cv2.morphologyEx(image,cv2.MORPH_CLOSE,kernel,iterations=2)

        return closing

    def processa_img(self):
        image = cv2.imread("input/17.png")

        gray_image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        gray_image = self.maintain_image_size(gray_image, width=300)
        image = self.segment_image(gray_image)

        cv2.imshow("Resultado",image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()