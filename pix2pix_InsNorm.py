from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Lambda
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Add, GlobalAveragePooling2D, multiply
from keras.layers.convolutional import UpSampling2D, Conv2D, AtrousConvolution2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from data_loader_alpha_sintes import DataLoader
import numpy as np
import os
import cv2

from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input

def L2(A,B):
    return K.sqrt(K.sum(K.pow(A-B, 2), axis=[1,2,3]))

class Pix2Pix():
    def __init__(self):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels_in = 3
        self.channels_out = 3
        self.img_shape_in = (self.img_rows, self.img_cols, self.channels_in)
        self.img_shape_out = (self.img_rows, self.img_cols, self.channels_out)
        
        vgg19 = VGG19()
        selectedLayers = [4,5,7]
        selectedOutputs = [vgg19.layers[i].output for i in selectedLayers]
        for i in np.arange(len(vgg19.layers)):
            vgg19.layers[i].trainable=False
        self.lossModel = Model(vgg19.inputs, selectedOutputs)
        self.lossModel.summary()

        # Configure data loader
        self.dataset_name = 'faces_bald_InsNorm_4x4_D2'
        self.data_loader = DataLoader()


        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        
        self.discriminator_mask = self.build_discriminator()
        self.discriminator_mask.compile(loss='mse', loss_weights=[2],
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        self.generator = self.build_generator()
        self.generator.summary()

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape_out)
        img_B = Input(shape=self.img_shape_in)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)
        
        #find perceptual loss between fake_A and img_A
        fake_A_preproc = Lambda(lambda x: ((x + 1.)*127.5))(fake_A)
        img_A_preproc = Lambda(lambda x: ((x + 1.)*127.5))(img_A)
        fake_A_preproc = Lambda(lambda x: preprocess_input(x))(fake_A_preproc)
        img_A_preproc = Lambda(lambda x: preprocess_input(x))(img_A_preproc)
        embeddings_fake_A = self.lossModel(fake_A_preproc)
        embeddings_img_A = self.lossModel(img_A_preproc)
        diffs = []
        for emb_fake_A, emb_img_A in zip(embeddings_fake_A, embeddings_img_A):
            l2_dist = Lambda(lambda x: L2(x[0], x[1]))([emb_fake_A, emb_img_A])
            diffs.append( l2_dist )
        diffs = Add()(diffs)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        self.discriminator_mask.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])
        valid_mask = self.discriminator_mask([fake_A, img_B])
        
        def empty_loss(y_true, y_pred):
            return y_pred

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, valid_mask, fake_A, diffs])
        self.combined.compile(loss=['mse', 
                                    'mse', 
                                    'mae', empty_loss],
                              loss_weights=[1, 2, 1, 0.00001],
                              optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""
        
        def squeeze_excite_block(input, ratio=4):
            ''' Create a channel-wise squeeze-excite block
            Args:
                input: input tensor
                filters: number of output filters
            Returns: a keras tensor
            References
            -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
            '''
            init = input
            channel_axis = 1 if K.image_data_format() == "channels_first" else -1
            filters = init._keras_shape[channel_axis]
            se_shape = (1, 1, filters)

            se = GlobalAveragePooling2D()(init)
            se = Reshape(se_shape)(se)
            se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
            se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

            if K.image_data_format() == 'channels_first':
                se = Permute((3, 1, 2))(se)

            x = multiply([init, se])
            return x

        def conv2d(layer_input, filters, f_size=4, bn=True, se=False):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                #d = BatchNormalization(momentum=0.8)(d)
                d = InstanceNormalization()(d)
            if se:
                d = squeeze_excite_block(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            #u = BatchNormalization(momentum=0.8)(u)
            u = InstanceNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u
        
        def atrous(layer_input, filters, f_size=4, bn=True):
            a_list = []
            for rate in [2,4,8]:
                a = AtrousConvolution2D(filters, f_size, atrous_rate=rate, border_mode='same')(layer_input)
                a_list.append(a)
            a = Concatenate()(a_list)
            a = LeakyReLU(alpha=0.2)(a)
            if bn:
                #a = BatchNormalization(momentum=0.8)(a)
                a = InstanceNormalization()(a)
            return a

        # Image input
        d0 = Input(shape=self.img_shape_in)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False, se=True)
        d2 = conv2d(d1, self.gf*2, se=True)
        d3 = conv2d(d2, self.gf*4, se=True)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*8)
        
        a1 = atrous(d5, self.gf*8)

        # Upsampling
        u3 = deconv2d(a1, d4, self.gf*8)
        u4 = deconv2d(u3, d3, self.gf*4)
        u5 = deconv2d(u4, d2, self.gf*2)
        u6 = deconv2d(u5, d1, self.gf)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(self.channels_out, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True, dropout_rate=0):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if dropout_rate:
                d = Dropout(dropout_rate)(d)
            if bn:
                #d = BatchNormalization(momentum=0.8)(d)
                d = InstanceNormalization()(d)
            return d

        img_A = Input(shape=self.img_shape_out)
        img_B = Input(shape=self.img_shape_in)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8, dropout_rate=0.2)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity)

    def train(self, epochs, batch_size=1, sample_interval=50):
        
        #self.generator.load_weights("saved_model/faces_bald_InsNorm_4x4_D1/model_G_76_184.hdf5")
        #self.discriminator.load_weights("saved_model/faces_bald_InsNorm_4x4_D1/model_D_76_184.hdf5")
        #self.discriminator_mask.load_weights("saved_model/faces_bald_InsNorm_4x4_D1/model_D_mask_76_184.hdf5")

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)
        perceptual = np.zeros((batch_size,))

        total_count = 0
        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B, mask) in enumerate(self.data_loader.load_batch(batch_size)):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Condition on B and generate a translated version
                fake_A = self.generator.predict(imgs_B)

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
                
                d_loss_real_mask = self.discriminator_mask.train_on_batch([imgs_A, imgs_B], valid)
                fake_A_mask = imgs_A * (1 - mask) + fake_A * mask
                d_loss_fake_mask = self.discriminator_mask.train_on_batch([fake_A_mask, imgs_B], fake)
                
                d_loss = [0, 0]
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                d_loss += 0.5 * np.add(d_loss_real_mask, d_loss_fake_mask)

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, valid, imgs_A, perceptual])

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                        batch_i, self.data_loader.n_batches,
                                                                        d_loss[0], 100*d_loss[1],
                                                                        g_loss[0],
                                                                        elapsed_time))

                # If at save interval => save generated image samples
                if total_count % sample_interval == 0:
                    self.sample_images(epoch, batch_i)
                total_count += 1

    def sample_images(self, epoch, batch_i):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        os.makedirs('saved_model/%s' % self.dataset_name, exist_ok=True)
        r, c = 3, 3

        imgs_A, imgs_B, mask = self.data_loader.load_data(batch_size=3, is_testing=True)
        fake_A = self.generator.predict(imgs_B)

        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[i])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        self.generator.save_weights("saved_model/%s/model_G_%d_%d.hdf5" % (self.dataset_name, epoch, batch_i))
        self.discriminator.save_weights("saved_model/%s/model_D_%d_%d.hdf5" % (self.dataset_name, epoch, batch_i))
        self.discriminator_mask.save_weights("saved_model/%s/model_D_mask_%d_%d.hdf5" % (self.dataset_name, epoch, batch_i))
        plt.close()


if __name__ == '__main__':
    gan = Pix2Pix()
    gan.train(epochs=4000, batch_size=10, sample_interval=1000)
