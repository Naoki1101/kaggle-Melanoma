import os
import cv2
import random
import numpy as np
import albumentations as album


class Microscope(album.ImageOnlyTransform):
    def __init__(self, p: float = 0.5, always_apply=False):
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        if random.random() < self.p:
            circle = cv2.circle((np.ones(img.shape) * 255).astype(np.uint8),
                        (img.shape[0]//2, img.shape[1]//2),
                        random.randint(img.shape[0]//2 - 3, img.shape[0]//2 + 15),
                        (0, 0, 0),
                        -1)

            mask = circle - 255
            img = np.multiply(img, mask)

        return img


# https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/159176
class HairAugmentation(album.ImageOnlyTransform):
    def __init__(self, max_hairs:int = 4, hairs_folder: str = "../data/input/hair/", p=0.5, always_apply=False):
        super().__init__(always_apply, p)
        self.max_hairs = max_hairs
        self.hairs_folder = hairs_folder

    def apply(self, img, **params):
        n_hairs = random.randint(0, self.max_hairs)

        if not n_hairs:
            return img

        height, width, _ = img.shape  # target image width and height
        hair_images = [im for im in os.listdir(self.hairs_folder) if 'png' in im]

        for _ in range(n_hairs):
            hair = cv2.imread(os.path.join(self.hairs_folder, random.choice(hair_images)))
            hair = cv2.flip(hair, random.choice([-1, 0, 1]))
            hair = cv2.rotate(hair, random.choice([0, 1, 2]))

            h_height, h_width, _ = hair.shape  # hair image width and height
            hair = cv2.resize(hair, (int(h_width*0.8), int(h_height*0.8)))

            h_height, h_width, _ = hair.shape  # hair image width and height
            roi_ho = random.randint(0, img.shape[0] - hair.shape[0])
            roi_wo = random.randint(0, img.shape[1] - hair.shape[1])
            roi = img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width]

            img2gray = cv2.cvtColor(hair, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            hair_fg = cv2.bitwise_and(hair, hair, mask=mask)

            dst = cv2.add(img_bg, hair_fg)
            img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width] = dst

        return img


# https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/165582
class HairRemove(album.ImageOnlyTransform):
    def __init__(self, p=0.5, always_apply=False):
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        # convert image to grayScale
        grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # kernel for morphologyEx
        kernel = cv2.getStructuringElement(1,(17,17))

        # apply MORPH_BLACKHAT to grayScale image
        blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

        # apply thresholding to blackhat
        _,threshold = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)

        # inpaint with original image and threshold image
        img = cv2.inpaint(img,threshold,1,cv2.INPAINT_TELEA)

        return img