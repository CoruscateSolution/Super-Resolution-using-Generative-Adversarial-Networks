from keras.layers import Lambda
import tensorflow as tf
from skimage import data, io, filters
import numpy as np
from numpy import array
from numpy.random import randint
from scipy.misc import imresize, imsave
import os
import sys
from keras.models import load_model

from model import *

def load_data_from_dirs(dirs, ext):
    files = []
    file_names = []
    count = 0
    for d in dirs:
        for f in os.listdir(d): 
            if f.endswith(ext):
                image = data.imread(os.path.join(d,f))
                image = imresize(image, (384, 384))
                if len(image.shape) > 2:
                    files.append(image)
                    file_names.append(os.path.join(d,f))
                count = count + 1
    return files  

# Takes list of images and provide HR images in form of numpy array
def hr_images(images):
    images_hr = array(images)
    return images_hr

# Takes list of images and provide LR images in form of numpy array
def lr_images(images_real , downscale):
    
    images = []
    for img in  range(len(images_real)):
        images.append(imresize(images_real[img], [images_real[img].shape[0]//downscale,images_real[img].shape[1]//downscale], interp='bicubic', mode=None))
    images_lr = array(images)
    return images_lr
    

def normalize(input_data):
    return (input_data.astype(np.float32) - 127.5)/127.5 
    
def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)
   
 
def load_path(path):
    directories = []
    if os.path.isdir(path):
        directories.append(path)
    for elem in os.listdir(path):
        if os.path.isdir(os.path.join(path,elem)):
            directories = directories + load_path(os.path.join(path,elem))
            directories.append(os.path.join(path,elem))
    return directories
    
def load_data_from_dirs(dirs, ext):
    files = []
    file_names = []
    count = 0
    for d in dirs:
        for f in os.listdir(d): 
            if f.endswith(ext):
                image = data.imread(os.path.join(d,f))
                image = imresize(image, (384, 384))
                if len(image.shape) > 2:
                    files.append(image)
                    file_names.append(os.path.join(d,f))
                count = count + 1
    return files 


def load_test_data_for_model(directory, ext, number_of_images = 100):

    files = load_data_from_dirs(load_path(directory), ext)
    
    if len(files) < number_of_images:
        print("Number of image files are less then you specified")
        print("Please reduce number of images to %d" % len(files))
        sys.exit()
        
    x_test_hr = hr_images(files)
    x_test_hr = normalize(x_test_hr)
    
    x_test_lr = lr_images(files, 4)
    x_test_lr = normalize(x_test_lr)
    
    return x_test_lr, x_test_hr

def load_test_data(directory, ext, number_of_images = 1):

    files = load_data_from_dirs(load_path(directory), ext)
    
    if len(files) < number_of_images:
        print("Number of image files are less then you specified")
        print("Please reduce number of images to %d" % len(files))
        sys.exit()
        
    x_test_lr = lr_images(files, 4)
    x_test_lr = normalize(x_test_lr)


x_test_lr, x_test_hr = load_test_data_for_model("./images", 'jpg', 1)

image_shape = (96,96,3)
loss = VGG_LOSS(image_shape)  
model = load_model('./weights/gen_model3000.h5' , custom_objects={'vgg_loss': loss.vgg_loss})

examples = x_test_hr.shape[0]
image_batch_hr = denormalize(x_test_hr)
image_batch_lr = x_test_lr
gen_img = model.predict(image_batch_lr)
generated_image = denormalize(gen_img)
image_batch_lr = denormalize(image_batch_lr)
output_dir = "./output"
for i,image in enumerate(generated_image):
    imsave(output_dir + "/" + (str(i)+"_hr.jpg"), image_batch_hr[i])
    imsave(output_dir + "/" +(str(i)+"lr.jpg"), image_batch_lr[i])
    imsave(output_dir + "/" + (str(i)+"_predict.jpg"), image)
    