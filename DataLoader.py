import os
import numpy as np
import scipy.misc
import scipy.ndimage
import skimage.color as ski
import h5py
np.random.seed(123)

# loading data from .h5
class DataLoaderH5(object):
    def __init__(self, **kwargs):
        self.load_size = int(kwargs['load_size'])
        self.fine_size = int(kwargs['fine_size'])
        self.data_mean = np.array(kwargs['data_mean'])
        self.randomize = kwargs['randomize']

        # read data info from lists
        f = h5py.File(kwargs['data_h5'], "r")
        self.im_set = np.array(f['images'])
        self.lab_set = np.array(f['labels'])

        self.num = self.im_set.shape[0]
        assert self.im_set.shape[0]==self.lab_set.shape[0], '#images and #labels do not match!'
        assert self.im_set.shape[1]==self.load_size, 'Image size error!'
        assert self.im_set.shape[2]==self.load_size, 'Image size error!'
        print('# Images found:', self.num)

        self.shuffle()
        self._idx = 0
        
    def next_batch(self, batch_size):
        labels_batch = np.zeros(batch_size)
        images_batch = np.zeros((batch_size, self.fine_size, self.fine_size, 3)) 
        
        for i in range(batch_size):
            image = self.im_set[self._idx]
            image = image.astype(np.float32)/255. - self.data_mean
            if self.randomize:
                flip = np.random.random_integers(0, 1)
                if flip>0:
                    image = image[:,::-1,:]
                offset_h = np.random.random_integers(0, self.load_size-self.fine_size)
                offset_w = np.random.random_integers(0, self.load_size-self.fine_size)
            else:
                offset_h = (self.load_size-self.fine_size)//2
                offset_w = (self.load_size-self.fine_size)//2

            images_batch[i, ...] = image[offset_h:offset_h+self.fine_size, offset_w:offset_w+self.fine_size, :]
            labels_batch[i, ...] = self.lab_set[self._idx]
            
            self._idx += 1
            if self._idx == self.num:
                self._idx = 0
                if self.randomize:
                    self.shuffle()
        
        return images_batch, labels_batch
    
    def size(self):
        return self.num

    def reset(self):
        self._idx = 0

    def shuffle(self):
        perm = np.random.permutation(self.num)
        self.im_set = self.im_set[perm] 
        self.lab_set = self.lab_set[perm]

# Loading data from disk
class DataLoaderDisk(object):
    def __init__(self, **kwargs):

        self.load_size = int(kwargs['load_size'])
        self.fine_size = int(kwargs['fine_size'])
        self.data_mean = np.array(kwargs['data_mean'])
        self.randomize = kwargs['randomize']
        self.num = 58741
        self.test = kwargs['test']
        self.data_root = os.path.join(kwargs['data_root'])
        #self.ra = RadialAugmenter(128, 128, 3)

        print('# Images found:', self.num)

        self._idx = 0
        
    def next_batch(self, batch_size):
        images_batch = np.zeros((batch_size, self.fine_size, self.fine_size, 1)) 
        labels_batch = np.zeros((batch_size, self.fine_size, self.fine_size, 3))

        for i in range(batch_size):
            file_name = 'lamem/images/{0:0>8}.jpg'.format(self._idx + 1)
            image = scipy.misc.imread(file_name)

            size_h = self.load_size
            size_w = self.load_size

            if self.randomize:
                stretch_h = np.random.randint(200)
                stretch_v = np.random.randint(200)

                size_h = self.load_size+stretch_h
                size_w = self.load_size+stretch_v

                image = scipy.misc.imresize(image, (size_h, size_w))
            else:
                image = scipy.misc.imresize(image, (self.load_size, self.load_size))

            image = image.astype(np.float32)/255.
            image = image - self.data_mean
            if self.randomize:
                offset_h = np.random.random_integers(0, size_h-self.fine_size)
                offset_w = np.random.random_integers(0, size_w-self.fine_size)
            else:
                offset_h = (self.load_size-self.fine_size)//2
                offset_w = (self.load_size-self.fine_size)//2

            #image = self.ra.radial_augment(image)
            #image = scipy.misc.imresize(image, (self.fine_size, self.fine_size))

            color_image = image[offset_h:offset_h+self.fine_size, offset_w:offset_w+self.fine_size, :]

            bw_im = ski.rgb2gray(color_image)

            images_batch[i, :, :, 0] = bw_im

            if not self.test:
                labels_batch[i, ...] = color_image
            
            self._idx += 1
            if self._idx == self.num:
                self._idx = 0
        
        if self.test:
            return images_batch, None
        else:
            return images_batch, labels_batch
    
    def size(self):
        return self.num

    def reset(self):
        self._idx = 0


