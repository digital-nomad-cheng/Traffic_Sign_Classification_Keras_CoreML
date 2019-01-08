"""
reimplement the simplenet V2 5M parameters version
"""
import keras
from keras.models import load_model, Model
from keras import regularizers, optimizers
from keras.layers import Input, Conv2D, Activation, Dense, Flatten
from keras.layers import BatchNormalization, Dropout
from keras.layers import MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.datasets import cifar10


def conv2d_bn_drop(x, filters, kernel_size=3, strides=1, padding='same', activation='relu', use_bias=False, dropout_rate=0, name=None):
    """Utility fucntion to apply conv + BN + dropout
    # Arguments:

    # Returns:
        Output tensor after applying 'Conv2D' and 'BatchNormalization' and "DropOut'
    """
    if name is not None:
        conv_name = name + '_conv'
        bn_name = name + '_bn'
        drop_name = name + '_dropout'
        ac_name = name + '_' + activation
    else:
        conv_name = None
        bn_name = None
        drop_name = name + '_dropout'
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias, name=conv_name)(x)
    x = BatchNormalization(axis=-1, scale=False, name=bn_name)(x)
    x = Activation(activation, name=ac_name)(x)
    x = Dropout(rate=dropout_rate, name=drop_name)(x)
    return x

def conv2d_bn_pooling_drop(x, filters, kernel_size=3, strides=1, padding='same', activation='relu', use_bias=False, pooling="max", dropout_rate=0, name=None):
    """Utility fucntion to apply conv + BN + dropout
    # Arguments:

    # Returns:
        Output tensor after applying 'Conv2D' and 'BatchNormalization' and "DropOut'
    """
    if name is not None:
        conv_name = name + '_conv'
        bn_name = name + '_bn'
        drop_name = name + '_dropout'
        ac_name = name + '_' + activation
    else:
        conv_name = None
        bn_name = None
        drop_name = name + '_dropout'
    x = Conv2D(filters, kernel_size, padding=padding, use_bias=use_bias, name=conv_name)(x)
    x = BatchNormalization(axis=-1, scale=False, name=bn_name)(x)
    if pooling == 'max':
        x = MaxPooling2D(pool_size=(2,2), strides=2, padding='valid')(x)
    else:
        x = AveragePooling2D(pool_size=(2,2), strides=2, padding='valid')(x)
    x = Activation(activation, name=ac_name)(x)
    x = Dropout(rate=dropout_rate, name=drop_name)(x)
    return x
def conv2d_pooling_bn_drop(x, filters, kernel_size=3, strides=1, padding='same', activation='relu', use_bias=False, pooling="max", dropout_rate=0, name=None):
    """Utility fucntion to apply conv + BN + dropout
    # Arguments:

    # Returns:
        Output tensor after applying 'Conv2D' and 'BatchNormalization' and "DropOut'
    """
    if name is not None:
        conv_name = name + '_conv'
        bn_name = name + '_bn'
        drop_name = name + '_dropout'
        ac_name = name + '_' + activation
    else:
        conv_name = None
        bn_name = None
        drop_name = name + '_dropout'
    x = Conv2D(filters, kernel_size, padding=padding, use_bias=use_bias, name=conv_name)(x)
    if pooling == 'max':
        x = MaxPooling2D(pool_size=(2,2), strides=2, padding='valid')(x)
    else:
        x = AveragePooling2D(pool_size=(2,2), strides=2, padding='valid')(x)
    x = BatchNormalization(axis=-1, scale=False, name=bn_name)(x)
    x = Activation(activation, name=ac_name)(x)
    x = Dropout(rate=dropout_rate, name=drop_name)(x)
    return x

def SimpleNet(input_tensor=None, stride=2, weight_decay=1e-2, pooling="Max", act='relu',
input_shape=(227,227,3), num_classes=10):
    s = stride
    act = 'relu' 
    
    if input_tensor is None:
        input_tensor = Input(shape=input_shape)   
    
    x = conv2d_bn_drop(input_tensor, 64, (7,7), strides=2, padding='same', activation='relu', name="block1_0")
    
    x = conv2d_bn_drop(x, 64, (3,3), padding='same', activation='relu', name="block1_1")
    
    x = conv2d_bn_drop(x, 96, (3,3), padding='same', activation='relu', name="block2_0")
    
    x = conv2d_bn_pooling_drop(x, 96, (3,3), padding='same', activation='relu', name="block2_1")
    
    x = conv2d_bn_drop(x, 96, (3,3), padding='same', activation='relu', name="block2_2")
    
    x = conv2d_bn_drop(x, 128, (3,3), padding='same', activation='relu', name="block3_0")
    
    x = conv2d_pooling_bn_drop(x, 128, (3,3), padding='same', activation='relu', name="block4_0")
    
    x = conv2d_bn_drop(x, 160, (3,3), padding='same', activation='relu', name="block4_1")
    
    x = conv2d_bn_pooling_drop(x, 160, (3,3), padding='same', activation='relu', dropout_rate=0.3, name="block4_2")

    x = Conv2D(filters=256, kernel_size=(3,3), strides=1, padding="same", activation='relu', name='block5_0_conv')(x)
    
    x = Conv2D(filters=512, kernel_size=(3,3), strides=1, padding="same", activation='relu', name='cccp5')(x)
    
    x = MaxPooling2D(pool_size=(2,2), strides=2, padding='valid', name='poolcp5')(x)
    
    x = Conv2D(filters=512, kernel_size=(3,3), strides=2, padding="same", activation='relu', name='cccp6')(x)
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes)(x) 
    x = Activation('softmax')(x)
    
    model = Model(inputs=input_tensor, outputs=x)
    model.summary()
    return model


if __name__ == '__main__':
    input_tensor = Input(shape=(227, 227,3))
    model = SimpleNet(input_tensor)

