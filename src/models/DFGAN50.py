from keras.models import Model
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Input, add, multiply, Lambda
from keras.layers.convolutional import AveragePooling2D, Conv2D
from keras.layers.advanced_activations import LeakyReLU
from .common import fft2d, fftshift2d, gelu, pixel_shiffle, conv_block2d, global_average_pooling2d


def FCALayer(input, channel, reduction=16, size_psc=128):
    absfft1 = Lambda(fft2d)(input)
    absfft1 = Lambda(fftshift2d, arguments={'size_psc': size_psc})(absfft1)
    absfft2 = Conv2D(channel, kernel_size=3, activation='relu', padding='same')(absfft1)
    W = Lambda(global_average_pooling2d)(absfft2)
    W = Conv2D(channel // reduction, kernel_size=1, activation='relu', padding='same')(W)
    W = Conv2D(channel, kernel_size=1, activation='sigmoid', padding='same')(W)
    mul = multiply([input, W])
    return mul


def FCAB(input, channel, size_psc=128):
    conv = Conv2D(channel, kernel_size=3, padding='same')(input)
    conv = Lambda(gelu)(conv)
    conv = Conv2D(channel, kernel_size=3, padding='same')(conv)
    conv = Lambda(gelu)(conv)
    att = FCALayer(conv, channel, reduction=16, size_psc=size_psc)
    output = add([att, input])
    return output


def ResidualGroup(input, channel, size_psc=128):
    conv = input
    n_RCAB = 10
    for _ in range(n_RCAB):
        conv = FCAB(conv, channel=channel, size_psc=size_psc)
    conv = add([conv, input])
    return conv


def Generator(input_shape, scale=2, channels=64, size_psc=128):
    inputs = Input(input_shape)
    conv = Conv2D(channels, kernel_size=3, padding='same')(inputs)
    conv = Lambda(gelu)(conv)
    n_ResGroup = 5
    for _ in range(n_ResGroup):
        conv = ResidualGroup(conv, channels, size_psc)
    conv = Conv2D(channels * (scale ** 2), kernel_size=3, padding='same')(conv)
    conv = Lambda(gelu)(conv)
    upsampled = Lambda(pixel_shiffle, arguments={'scale': scale})(conv)
    conv = Conv2D(1, kernel_size=3, padding='same')(upsampled)
    output = Activation('sigmoid')(conv)
    model = Model(inputs=inputs, outputs=output)
    return model


def Discriminator(input_shape):
    input_img = Input(shape=input_shape)
    x0 = Conv2D(32, kernel_size=3, padding='same')(input_img)
    x0 = LeakyReLU(alpha=0.1)(x0)
    x1 = conv_block2d(x0, (32, 32))
    x2 = conv_block2d(x1, (64, 64))
    x3 = conv_block2d(x2, (128, 128))
    x4 = conv_block2d(x3, (256, 256))
    x5 = conv_block2d(x4, (512, 512))
    x6 = AveragePooling2D(pool_size=(input_shape[0], input_shape[1]))(x5)
    y0 = Flatten(input_shape=(1, 1))(x6)
    y1 = Dense(256)(y0)
    y1 = LeakyReLU(alpha=0.1)(y1)
    output = Dense(1, activation='sigmoid')(y1)
    model = Model(inputs=input_img, outputs=output)
    return model