import os
from glob import glob
from glob import glob

import cv2
import numpy as np


class DataLoader():
    def __init__(self):
        bald_folder = '/mnt/ssd2/Datasets/bald_for_GAN/'
        self.hair_folder = '/mnt/ssd2/Datasets/hair_for_GAN/'
        self.hair_mask_folder = '/mnt/ssd2/Datasets/hair_masks_for_GAN/'
        
        self.bald_images = glob(os.path.join(bald_folder, '*.jpg'))
        self.hair_images = glob(os.path.join(self.hair_folder, '*.png'))        

    def load_data(self, batch_size=1, is_testing=False):

        batch_images = np.random.choice(self.bald_images, size=batch_size)

        imgs_A = []
        imgs_B = []
        imgs_alpha = []
        
        for img_path in batch_images:
            img = self.imread(img_path)
            if np.random.randint(2) == 0:
                img = img[:,::-1,:]
            hair_imgpath= np.random.choice(self.hair_images)
            hair_img = self.imread(hair_imgpath)
            mask = self.imread(hair_imgpath.replace(self.hair_folder, self.hair_mask_folder))
            mask = mask / 255
            
            if np.random.randint(2) == 0:
                hair_img = hair_img[:,::-1,:]
                mask = mask[:,::-1,:]

            img_A = img
            img_B = img * (1-mask) + hair_img * mask
            
            imgs_A.append(img_A)
            imgs_B.append(img_B)
            imgs_alpha.append(mask)

        imgs_A = np.array(imgs_A)/127.5 - 1.
        imgs_B = np.array(imgs_B)/127.5 - 1.
        imgs_alpha = np.array(imgs_alpha)

        return imgs_A, imgs_B, imgs_alpha

    def load_batch(self, batch_size=1, is_testing=False):
        self.n_batches = int(len(self.bald_images) / batch_size)

        for i in range(self.n_batches-1):
            batch = self.bald_images[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B, imgs_alpha = [], [], []
            for img_path in batch:
                img = self.imread(img_path)
                if np.random.randint(2) == 0:
                    img = img[:,::-1,:]
                hair_imgpath= np.random.choice(self.hair_images)
                hair_img = self.imread(hair_imgpath)
                mask = self.imread(hair_imgpath.replace(self.hair_folder, self.hair_mask_folder))
                mask = mask / 255
                
                if np.random.randint(2) == 0:
                    hair_img = hair_img[:,::-1,:]
                    mask = mask[:,::-1,:]

                img_A = img
                img_B = img * (1-mask) + hair_img * mask

                imgs_A.append(img_A)
                imgs_B.append(img_B)
                imgs_alpha.append(mask)


            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.
            imgs_alpha = np.array(imgs_alpha)

            yield imgs_A, imgs_B, imgs_alpha


    def imread(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
