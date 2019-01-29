import threading
import os
import glob
import numpy as np
import tensorflow as tf
import cv2
import sklearn
import keras.backend as K
import random
import pickle
import json
import imgaug
from imgaug import augmenters as iaa
import imageio


def download(dataset_name):
    datasets_dir = './datasets/'
    mkdir(datasets_dir)
    URL='https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/%s.tar.gz' % (dataset_name)
    TAR_FILE='./datasets/%s.tar.gz' % (dataset_name)
    TARGET_DIR='./datasets/%s/' % (dataset_name)
    os.system('wget -N %s -O %s' % (URL, TAR_FILE))
    os.mkdir(TARGET_DIR)
    os.system('tar -zxf %s -C ./datasets/' % (TAR_FILE))
    os.remove(TAR_FILE)


class DLoader(object):
    def __init__(self, config, resize=False, augment=False, resample=False, phmin=False,
                 catin=False, cat_outchannels=False, apply_mask=False, debug=False):
        self.resample = resample
        self.resize = resize
        self.augment = augment
        self.debug = debug
        self.color_augment_precent = 0.0
        self.img_lab_aug_precent = 0.5
        self.rotate_precent = 0.3
        self.batch_size = config['batch_size']
        self.thread_num = config['thread_num']
        self.pad_divisable = config['pad_divisable']
        self.min_size = config['min_size']
        self.max_size = config['max_size']
        self.img_pad_val = config['img_pad_val']
        self.label_pad_val = config['label_pad_val']
        self.aug_pix_enlarge = config['aug_pix_enlarge']
        self.data_root = config['data_root']
        self.pickled_data = config['train_val_lists']
        self.train_data, self.val_data, self.test_data = self.load_data()
        self.data_size = len(self.train_data)
        self.data_indice = range(self.data_size - 1)
        self.img_shape = config['img_inp_shape']
        self.out_shape = config['out_shape']
        self.out_channels = self.out_shape[-1]
        self.resize_res = 512
        with tf.device('/gpu:0'):
            self.img_data = tf.placeholder(tf.float32, shape=[None] + self.img_shape)
            self.label_data = tf.placeholder(tf.float32, shape=[None] + self.out_shape)

            queue_types = [tf.float32, tf.float32]
            queue_objects = [self.label_data, self.img_data]

            self.queue = tf.FIFOQueue(shapes=None, #shapes=[self.out_shape, self.img_shape],
                                      dtypes=queue_types,
                                      capacity=10)
            self.enqueue_ops = self.queue.enqueue_many(queue_objects)
        self.print_load_done()
        
    def print_load_done(self):
        if not os.path.exists(self.pickled_data):
            print('bad data root. update config')
        else:
            print("Batch size: %d, Thread num: %d" % (self.batch_size, self.thread_num))
            print('in shape:', self.img_shape, 'label shape:', self.out_shape)
            print("load dataset done")
            print('data size: %d' % self.data_size)
    
    def load_data(self):
        data = pickle.load(open(self.pickled_data, 'rb'))
        return data['x_train'], data['x_val'], data['x_test']
    
    def load_img(self, img_path):
        img = imageio.imread(img_path)
        return np.expand_dims(img, axis=-1)
        
    def load_img_label(self, img_name):
        img_path = os.path.join(self.data_root, 'jpg', img_name+ '.jpg')
        label_path = os.path.join(self.data_root, 'bmp', img_name+ '.bmp')
        img = self.load_img(img_path)
        label = self.load_img(label_path)
        return img, label
    
    def pad_img(self, img, pad_val=0):
        org_size = img.shape
        if self.resize: 
            scale = np.min(self.resize_res/np.array(org_size)[:2])
            img = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            pad = self.resize_res - np.array(img.shape)[:2]
        else:
            scale = 1
            if np.max(org_size[:2]) > self.max_size:
                scale = self.max_size/np.max(org_size[:2])
            elif np.min(org_size[:2]) < self.min_size:
                scale = self.min_size/np.min(org_size[:2])
            if scale != 1:
                img = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            pad = -np.array(img.shape)[:2] % self.pad_divisable
        img = np.pad(img, ((0,pad[0]), (0,pad[1]), (0,0)), 'constant', constant_values=pad_val)
        return img, pad
    
    def resize_img_label(self, img, label):
        org_size = img.shape
        if self.resize: 
            scale = np.min(self.resize_res/np.array(org_size)[:2])
            img = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            pad = self.resize_res - np.array(img.shape)[:2]
        else:
            scale = 1
            if np.max(org_size[:2]) > self.max_size:
                scale = self.max_size/np.max(org_size[:2])
            elif np.min(org_size[:2]) < self.min_size:
                scale = self.min_size/np.min(org_size[:2])
            if scale != 1:
                img = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                label = cv2.resize(label, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            pad = -np.array(img.shape)[:2] % self.pad_divisable
        return img, label, pad
    
    def pad_img_label(self, img, label, pad):
        if self.img_pad_val != 0 and self.phmin:
            print('error! either change img_pad val to zero or pad phm seperatly')
        img = np.pad(img, ((0,pad[0]), (0,pad[1]), (0,0)), 'constant', constant_values=255)
        label = np.pad(label, ((0,pad[0]), (0,pad[1]), (0,0)), 'constant', constant_values=self.label_pad_val)
        return img, label
    
    def img_shift(self, img, mean=127.5):
        img = img / mean - 1.
        return img
    
    def get_img_label(self, folder):
        img, label = self.load_img_label(folder)
            
        if self.augment and random.uniform(0,1) < self.color_augment_precent:
                img = self.color_augment(img)
        if self.augment and random.uniform(0,1) < self.img_lab_aug_precent:
            img, label = self.augment_img_mask(img, label)
        img, label, pad = self.resize_img_label(img, label)
        img, label = self.pad_img_label(img, label, pad)
        
        img = self.img_shift(img)
        return img, label
    
    
    def img_augment(self, img, label, folder=''):
        org_shape = img.shape
        scale = (org_shape[0]+self.aug_pix_enlarge)/org_shape[0]
        img = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        if self.debug:
            print(folder, '\norg_shape:', org_shape, 'aug resize:', img.shape, label.shape)
        
        h1 = int(np.ceil(np.random.uniform(0, self.aug_pix_enlarge)))
        w1 = int(np.ceil(np.random.uniform(0, self.aug_pix_enlarge)))
        img = img[h1:h1+org_shape[0], w1:w1+org_shape[1]]
        label = label[h1:h1+org_shape[0], w1:w1+org_shape[1]]
        if self.debug:
            print('after aug shapes:', img.shape, label.shape, h1, w1)
        if np.random.random() > 0.5:
            img = np.fliplr(img)
            label = np.fliplr(label)
        return img, label
    
    
    def augment_img_mask(self, image, mask):

        aug_list = self.get_augmenters_img_mask()

        seq = iaa.Sequential(aug_list)

        # Make augmenters deterministic to apply similarly to images and masks
        seq_det = seq.to_deterministic()
        
        if self.debug:
            print('rotate aug list:', aug_list)

        aug_img = seq_det.augment_images([image])[0]
        aug_mask = seq_det.augment_images([mask])[0]
        # aug_mask[:,:,0] = 1 - aug_mask[:,:,1]
        return aug_img, aug_mask
    
    
    def get_augmenters_img_mask(self):
        aug_list = []
        if random.uniform(0,1) < self.rotate_precent:
            aug_list.append(iaa.Affine(rotate=(-10,10)))
        if random.uniform(0,1) > 0.4:
            aug_list.append(iaa.Affine(scale=(0.7, 1.0)))
        if random.uniform(0,1) > 0.4:
            aug_list.append(iaa.Affine(translate_px={"x": (-20, 20), "y": (-20, 20)}))
        return sklearn.utils.shuffle(aug_list)
    
    
    def get_color_augmenters(self):
        AUGS = [
            iaa.Multiply((0.5, 1.5), per_channel=0.5), # Multiply all pixels with a specific value
#             iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5), # changes the contrast of images
#             iaa.Add((-25, 25), per_channel=0.25), # Add a value to all pixels
#             iaa.AddToHueAndSaturation((-20, 20)),
            iaa.Grayscale(alpha=(0.5, 1.0)),
#             iaa.AdditiveGaussianNoise(scale=0.05*255, per_channel=0.25), #Add gaussian noise (white noise)
            iaa.CoarseDropout((0.02,0.1), size_percent=(0.001,0.05), per_channel=1) #sets rectangular areas to zero
        ]
        return AUGS
    
    
    def color_augment(self, img):
        IMG_AUGS = self.get_color_augmenters()
        img_seq = iaa.SomeOf((1, 3),IMG_AUGS, random_order=True)
        img = img_seq.augment_images([img])[0]
        return img
              
    def load_pickled_obj(self, names, path):
        fopen = pickle.load(open(path, "rb"))
        objs = {}
        for i in range(len(names)):
            objs[names[i]] = fopen[names[i]]
        print('loaded pickled data')
        return objs

    def batch_iterator(self, augment=True, shuff=True):
        samp_list = self.train_data
        while True:
            if shuff:
                samp_list = sklearn.utils.shuffle(samp_list)
            for i in range(len(self.data_indice)//self.batch_size):
                if self.debug:
                    print(i, 'of', len(self.data_indice)//self.batch_size)
                img_batch, label_batch, cat_batch = [], [], []
                for j in range(i*self.batch_size, (i+1)*self.batch_size):
                    img, label = self.get_img_label(samp_list[j])
                    label_batch.append(label)
                    img_batch.append(img)    
                yield np.array(img_batch), np.array(label_batch)

               
    def get_inputs(self):
        with tf.device('/gpu:0'):
            if self.batch_size > 1:
                labels, imgs = self.queue.dequeue_many(self.batch_size)
            else:
                labels, imgs = self.queue.dequeue()
                labels = tf.expand_dims(labels, 0)
                imgs = tf.expand_dims(imgs, 0)
            return imgs, labels
    
    def thread_main(self, sess):
        for imgs, labels in self.batch_iterator():
            _ = sess.run(self.enqueue_ops, feed_dict={self.img_data: imgs, self.label_data: labels})
    
    
    def start_threads(self, sess):
        threads = []
        for n in range(self.thread_num):
            t = threading.Thread(target=self.thread_main, args=(sess,))
            t.daemon = True
            t.start()
            threads.append(t)
        return threads
    
    def get_size(self):
        return self.data_size

    def get_shape(self):
        return self.img_shape, self.out_shape
    
  
    def get_batch_imgs(self, folders=[], train=False, ret_names=False):
        test_imgs, test_labels = [], []
        
        if len(folders) < 1:
            if train:
                folders.append(self.train_data[random.randint(0, len(self.train_data)-1)])
            else:
                folders.append(self.val_data[random.randint(0, len(self.val_data)-1)])
                
        for i in range(len(folders)):
            test_img, test_label = self.get_img_label(folders[i])
            
            test_imgs.append(test_img)
            test_labels.append(test_label)
            
        if ret_names:
            return np.array(test_imgs), np.array(test_labels), folders
        else:
            return np.array(test_imgs), np.array(test_labels)     