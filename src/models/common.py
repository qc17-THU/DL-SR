from keras.layers.convolutional import Conv2D, Conv3D
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
import numpy as np
import tensorflow as tf


def gelu(x):
    cdf = 0.5 * (1.0 + tf.erf(x / tf.sqrt(2.0)))
    return x * cdf


def fft2d(input, gamma=0.1):
    temp = K.permute_dimensions(input, (0, 3, 1, 2))
    fft = tf.fft2d(tf.complex(temp, tf.zeros_like(temp)))
    absfft = tf.pow(tf.abs(fft)+1e-8, gamma)
    output = K.permute_dimensions(absfft, (0, 2, 3, 1))
    return output


def fft3d(input, gamma=0.1):
    input = apodize3d(input, napodize=5)
    temp = K.permute_dimensions(input, (0, 4, 1, 2, 3))
    fft = tf.fft3d(tf.complex(temp, tf.zeros_like(temp)))
    absfft = tf.pow(tf.abs(fft) + 1e-8, gamma)
    output = K.permute_dimensions(absfft, (0, 2, 3, 4, 1))
    return output


def fftshift2d(input, size_psc=128):
    bs, h, w, ch = input.get_shape().as_list()
    fs11 = input[:, -h // 2:h, -w // 2:w, :]
    fs12 = input[:, -h // 2:h, 0:w // 2, :]
    fs21 = input[:, 0:h // 2, -w // 2:w, :]
    fs22 = input[:, 0:h // 2, 0:w // 2, :]
    output = tf.concat([tf.concat([fs11, fs21], axis=1), tf.concat([fs12, fs22], axis=1)], axis=2)
    output = tf.image.resize_images(output, (size_psc, size_psc), 0)
    return output


def fftshift3d(input, size_psc=64):
    bs, h, w, z, ch = input.get_shape().as_list()
    fs111 = input[:, -h // 2:h, -w // 2:w, -z // 2 + 1:z, :]
    fs121 = input[:, -h // 2:h, 0:w // 2, -z // 2 + 1:z, :]
    fs211 = input[:, 0:h // 2, -w // 2:w, -z // 2 + 1:z, :]
    fs221 = input[:, 0:h // 2, 0:w // 2, -z // 2 + 1:z, :]
    fs112 = input[:, -h // 2:h, -w // 2:w, 0:z // 2 + 1, :]
    fs122 = input[:, -h // 2:h, 0:w // 2, 0:z // 2 + 1, :]
    fs212 = input[:, 0:h // 2, -w // 2:w, 0:z // 2 + 1, :]
    fs222 = input[:, 0:h // 2, 0:w // 2, 0:z // 2 + 1, :]
    output1 = tf.concat([tf.concat([fs111, fs211], axis=1), tf.concat([fs121, fs221], axis=1)], axis=2)
    output2 = tf.concat([tf.concat([fs112, fs212], axis=1), tf.concat([fs122, fs222], axis=1)], axis=2)
    output0 = tf.concat([output1, output2], axis=3)
    output = []
    for iz in range(z):
        output.append(tf.image.resize_images(output0[:, :, :, iz, :], (size_psc, size_psc), 0))
    output = tf.stack(output, axis=3)
    return output


def apodize2d(img, napodize=10):
    bs, ny, nx, ch = img.get_shape().as_list()
    img_apo = img[:, napodize:ny-napodize, :, :]

    imageUp = img[:, 0:napodize, :, :]
    imageDown = img[:, ny-napodize:, :, :]
    diff = (imageDown[:, -1::-1, :, :] - imageUp) / 2
    l = np.arange(napodize)
    fact_raw = 1 - np.sin((l + 0.5) / napodize * np.pi / 2)
    fact = fact_raw[np.newaxis, :, np.newaxis, np.newaxis]
    fact = tf.convert_to_tensor(fact, dtype=tf.float32)
    fact = tf.tile(fact, [tf.shape(img)[0], 1, nx, ch])
    factor = diff * fact
    imageUp = tf.add(imageUp, factor)
    imageDown = tf.subtract(imageDown, factor[:, -1::-1, :, :])
    img_apo = tf.concat([imageUp, img_apo, imageDown], axis=1)

    imageLeft = img_apo[:, :, 0:napodize, :]
    imageRight = img_apo[:, :, nx-napodize:, :]
    img_apo = img_apo[:, :, napodize:nx-napodize, :]
    diff = (imageRight[:, :, -1::-1, :] - imageLeft) / 2
    fact = fact_raw[np.newaxis, np.newaxis, :, np.newaxis]
    fact = tf.convert_to_tensor(fact, dtype=tf.float32)
    fact = tf.tile(fact, [tf.shape(img)[0], ny, 1, ch])
    factor = diff * fact
    imageLeft = tf.add(imageLeft, factor)
    imageRight = tf.subtract(imageRight, factor[:, :, -1::-1, :])
    img_apo = tf.concat([imageLeft, img_apo, imageRight], axis=2)

    return img_apo


def apodize3d(img, napodize=5):
    bs, ny, nx, nz, ch = img.get_shape().as_list()
    img_apo = img[:, napodize:ny-napodize, :, :, :]

    imageUp = img[:, 0:napodize, :, :, :]
    imageDown = img[:, ny-napodize:, :, :, :]
    diff = (imageDown[:, -1::-1, :, :, :] - imageUp) / 2
    l = np.arange(napodize)
    fact_raw = 1 - np.sin((l + 0.5) / napodize * np.pi / 2)
    fact = fact_raw[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
    fact = tf.convert_to_tensor(fact, dtype=tf.float32)
    fact = tf.tile(fact, [tf.shape(img)[0], 1, nx, nz, ch])
    factor = diff * fact
    imageUp = tf.add(imageUp, factor)
    imageDown = tf.subtract(imageDown, factor[:, -1::-1, :, :, :])
    img_apo = tf.concat([imageUp, img_apo, imageDown], axis=1)

    imageLeft = img_apo[:, :, 0:napodize, :, :]
    imageRight = img_apo[:, :, nx-napodize:, :, :]
    img_apo = img_apo[:, :, napodize:nx-napodize, :, :]
    diff = (imageRight[:, :, -1::-1, :, :] - imageLeft) / 2
    fact = fact_raw[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
    fact = tf.convert_to_tensor(fact, dtype=tf.float32)
    fact = tf.tile(fact, [tf.shape(img)[0], ny, 1, nz, ch])
    factor = diff * fact
    imageLeft = tf.add(imageLeft, factor)
    imageRight = tf.subtract(imageRight, factor[:, :, -1::-1, :, :])
    img_apo = tf.concat([imageLeft, img_apo, imageRight], axis=2)

    return img_apo


def pixel_shiffle(layer_in, scale):
    return tf.depth_to_space(layer_in, block_size=scale)


def global_average_pooling2d(layer_in):
    return tf.reduce_mean(layer_in, axis=(1, 2), keepdims=True)


def global_average_pooling3d(layer_in):
    return tf.reduce_mean(layer_in, axis=(1, 2, 3), keepdims=True)


def conv_block2d(input, channel_size):
    conv = Conv2D(channel_size[0], kernel_size=3, padding='same')(input)
    conv = LeakyReLU(alpha=0.1)(conv)
    conv = Conv2D(channel_size[1], kernel_size=3, padding='same')(conv)
    conv = LeakyReLU(alpha=0.1)(conv)
    return conv


def conv_block3d(input, channel_size):
    conv = Conv3D(channel_size[0], kernel_size=3, padding='same')(input)
    conv = LeakyReLU(alpha=0.1)(conv)
    conv = Conv3D(channel_size[1], kernel_size=3, padding='same')(conv)
    conv = LeakyReLU(alpha=0.1)(conv)
    return conv