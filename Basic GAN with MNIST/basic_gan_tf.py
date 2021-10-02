import os 
import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)

import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, ReLU, LeakyReLU

from utils import *
from variables import *

def get_noise():
    return tf.random.normal([batch_size, z_dim])

def generator_block(input_dim, output_dim):
    gen = Sequential()
    gen.add(Dense(input_dim=input_dim, units=output_dim))
    gen.add(BatchNormalization())
    gen.add(ReLU())
    return gen

def discriminator_block(input_dim, output_dim):
    dis = Sequential()
    dis.add(Dense(input_dim=input_dim, units=output_dim))
    dis.add(LeakyReLU(0.2))
    return dis
    
class Generator(tf.keras.Model):
    def __init__(self, z_dim=z_dim, im_dim=image_dim**2, hidden_dim=hidden_dim):
        super(Generator, self).__init__()

        self.im_dim = im_dim
        self.gen1 = generator_block(z_dim, hidden_dim)
        self.gen2 = generator_block(hidden_dim, hidden_dim * 2)
        self.gen3 = generator_block(hidden_dim * 2, hidden_dim * 4)
        self.gen4 = generator_block(hidden_dim * 4, hidden_dim * 8)
        
    def call(self, noise):
        x = self.gen1(noise)
        x = self.gen2(x)
        x = self.gen3(x)
        x = self.gen4(x)
        x = Dense(units=self.im_dim)(x)
        x = tf.nn.tanh(x)
        return x

class Discriminator(tf.keras.Model):
    def __init__(self, im_dim=image_dim**2, hidden_dim=hidden_dim):
        super(Discriminator, self).__init__()

        self.dis1 = discriminator_block(im_dim, hidden_dim * 4)
        self.dis2 = discriminator_block(hidden_dim * 4, hidden_dim * 2)
        self.dis3 = discriminator_block(hidden_dim * 2, hidden_dim)
        
    def call(self, image):
        x = self.dis1(image)
        x = self.dis2(x)
        x = self.dis3(x)
        x = Dense(units=1)(x)
        x = tf.nn.sigmoid(x)
        return x

def load_data():
    (X_train, _), _ = mnist.load_data()
    X_train = X_train.reshape(-1, image_dim**2)
    X_train = X_train.astype('float32') / 255.0

    X = tf.data.Dataset.from_tensor_slices(X_train).shuffle(1000)
    X = X.batch(batch_size, drop_remainder=True).prefetch(1)
    return X

def create_gan(generator, discriminator):
    gan = Sequential()
    gan.add(generator)
    gan.add(discriminator)
    return gan

def compile_models():
    generator = Generator()
    discriminator = Discriminator()

    discriminator.build(input_shape=(None, image_dim**2))
    discriminator.summary()

    discriminator.compile(
            loss='binary_crossentropy', 
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
            metrics=['accuracy']
            )

    discriminator.trainable = False
    gan = create_gan(generator, discriminator)
    gan.build(input_shape=(None, z_dim))
    gan.summary()

    gan.compile(
            loss='binary_crossentropy', 
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
            metrics=['accuracy']
            )

    return gan, discriminator, generator

class GanTrainer:
    def __init__(self):
        self.gan, self.discriminator, self.generator = compile_models()
        self.X = load_data()

    def train_loop(self):
        for epoch in range(n_epochs):
            print('Epoch: {}/{}'.format(epoch, n_epochs))
            for i, image_batch in enumerate(self.X):
                ################################# Phase 1 (Train Discriminator) #################################
                noise = get_noise()
                generated_images = self.generator(noise)
                mixed_images = tf.concat([image_batch, generated_images], axis=0)

                real_labels = tf.ones((batch_size, 1))
                fake_labels = tf.zeros((batch_size, 1))
                mixed_labels = tf.concat([real_labels, fake_labels], axis=0)

                self.discriminator.trainable = True
                self.discriminator.train_on_batch(mixed_images, mixed_labels)

                ################################## Phase 2 (Train Generator) ####################################
                noise = get_noise()
                real_labels = tf.ones((batch_size, 1))  # Fool the discriminator
                self.discriminator.trainable = False
                self.gan.train_on_batch(noise, real_labels)                

                if i % 100 == 0:
                    plot_multiple_images(generated_images, (epoch, i), 8)    

model = GanTrainer()
model.train_loop()