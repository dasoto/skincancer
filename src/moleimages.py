import numpy as np
from skimage import io
from skimage.transform import resize

import matplotlib.pyplot as plt
import glob
import h5py


class MoleImages():
    def __init__(self, dir=None):
        self.dir = dir
        self.size = None

    def resize_bulk(self, size=(128,128)):
        '''
        Resize Images and create matrix
        Input: size of the images (128,128)
        Output: Numpy array of (size,num_images)
        '''
        self.size = size
        X = []
        image_list = glob.glob(self.dir)
        n_images = len(image_list)
        print('Resizing {} images:'.format(n_images))
        for i, imgfile in enumerate(image_list):
            print('Resizing image {} of {}'.format(i+1, n_images))
            img = io.imread(imgfile)
            img = resize(img, self.size)
            X.append(img)
        return np.array(X)

    def load_test_images(self, dir_b, dir_m):
        X = []
        image_list_b = glob.glob(dir_b + '/*.png')
        n_images_b = len(image_list_b)
        print('Loading {} images of class benign:'.format(n_images_b))
        for i, imgfile in enumerate(image_list_b):
            print('Loading image {} of {}'.format(i+1, n_images_b))
            img = io.imread(imgfile)
            X.append(img)
        image_list_m = glob.glob(dir_m + '/*.png')
        n_images_m = len(image_list_m)
        print('Loading {} images of class benign:'.format(n_images_m))
        for i, imgfile in enumerate(image_list_m):
            print('Loading image {} of {}'.format(i+1, n_images_m))
            img = io.imread(imgfile)
            X.append(img)
        y = np.hstack((np.zeros(n_images_b), np.ones(n_images_m)))

        return np.array(X), y.reshape(len(y),1)

    def load_image(self, filename, size=(128,128)):
        self.size = size
        img = io.imread(filename)
        img = resize(img, self.size, mode='constant') * 255
        if img.shape[2] == 4:
            img = img[:,:,0:3]
        return img.reshape(1, self.size[0], self.size[1], 3)

    def save_h5(self, X, filename, dataset):
        '''
        Save a numpy array to a data.h5 file specified.
        Input:
        X: Numpy array to save
        filename: name of h5 file
        dataset: label for the dataset
        '''
        with h5py.File(filename, 'w') as hf:
            hf.create_dataset(dataset, data=X)
        print('File {} saved'.format(filename))

    def load_h5(self, filename, dataset):
        '''
        Load a data.h5 file specified.
        Input: filename, dataset
        Output: Data
        '''
        with h5py.File(filename, 'r') as hf:
            return hf[dataset][:]

    def save_png(self, matrix, dir, tag='img', format='png'):
        for i, img in enumerate(matrix):
            if dir[-1] != '/':
                filename = dir + '/' + tag + str(i) + '.' + format
            else:
                filename = dir + tag + str(i) + '.' + format
            print('Saving file {}'.format(filename))
            io.imsave(filename, img)



if __name__ == '__main__':
    # benign = MoleImages('data/malignant/*.jpg')
    # ben_images = benign.resize_bulk()
    # print('Shape of benign images: ', ben_images.shape)
    # benign.save_h5(ben_images, 'benigns.h5', 'benign')
    benign = MoleImages()
    X = benign.load_h5('benigns.h5','benign')
