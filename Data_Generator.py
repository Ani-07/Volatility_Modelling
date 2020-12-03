# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 14:04:06 2020

@author: Anirudh Raghavan
"""

import os

import keras
from keras.models import Sequential
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from keras.optimizers import Adam
from keras.layers import Dense, Input, Flatten, Reshape, Concatenate, Dropout, Embedding, LeakyReLU, Embedding
from keras import Model
import numpy as np
import pandas as pd
from keras import backend as K

new_loc = r"C:\Users\Anirudh Raghavan\Desktop\Stevens - Courses\Fall 2020\FE 690 - Machine Learning\Assignment_3_Volatility_Modelling\Data"

os.chdir(new_loc)

def relu_1(x):
    return K.relu(x, max_value=1)


# Load our model

custom_objects = {'relu_1': relu_1}

model = keras.models.load_model('cgan_generator.h5', custom_objects)

def generate_latent_points(latent_dim, n_samples, n_classes=2):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	labels = randint(0, n_classes, n_samples)
	return [z_input, labels]
 
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	z_input, labels_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	images = generator.predict([z_input, labels_input])
	# create class labels
	return [images, labels_input]


latent_dim = 100
n_samples = 2000
[X_fake, labels] = generate_fake_samples(model, latent_dim, n_samples)

np.savetxt("GAN_X.csv", X_fake, delimiter=",")

np.savetxt("GAN_y.txt", labels, delimiter=",")


