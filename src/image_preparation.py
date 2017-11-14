from moleimages import MoleImages
import glob
import os
import numpy as np


def resize_images():
    print('Resizing Benign')
    moles = MoleImages('data/benign/*.jpg')
    benigns = moles.resize_bulk()
    moles.save_png(benigns, 'data_scaled/benign', tag='bimg-')

    print('Resizing Malign')
    moles = MoleImages('data/malignant/*.jpg')
    malignants = moles.resize_bulk()
    moles.save_png(malignants,'data_scaled/malign', tag='mimg-')

def cv_images(dir_b='data_scaled_validation/benign', dir_m='data_scaled_validation/malign', pct=0.1):
    image_b = glob.glob('data_scaled/benign/*.png')
    image_m = glob.glob('data_scaled/malign/*.png')
    n_images_b = int(pct*len(image_b))
    n_images_m = int(pct*len(image_m))
    image_b = np.random.choice(image_b,n_images_b, replace=False)
    image_m = np.random.choice(image_m,n_images_m, replace=False)
    for img in image_b:
        filename = img.split('/')[-1]
        print('Moving {} to {}'.format(img,dir_b + '/' + filename))
        os.rename(img,dir_b + '/' + filename)
    for img in image_m:
        filename = img.split('/')[-1]
        print('Moving {} to {}'.format(img,dir_m + '/' + filename))
        os.rename(img,dir_m + '/' + filename)


if __name__ == '__main__':
    #resize_images()
    cv_images()
