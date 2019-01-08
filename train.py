"""
Easy to use train script for different kinds of networkds and dataset...
@author: Vincent

"""
import os
import glob
from collections import Counter
import numpy as np
import keras
from keras.optimizers import SGD
import keras.backend as K
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks
import argparse
from simplenet import SimpleNet
from learning_rate import create_lr_schedule


if __name__ == "__main__":
        ap = argparse.ArgumentParser()
        ap.add_argument(
                '--dataset',
                type=str,
                default='traffic_sign',
                help='directory name of dataset, which should have structure ./train ./val and according classes to suit flow from directory'
        )
        ap.add_argument(
                '--batch_size',
                type=int,
                default=16,
                help='training batch size'
        )
        ap.add_argument(
                '--input_shape',
                type=list,
                default=(112,112,3),
                help='input image shape',
        )
        ap.add_argument(
                '--epochs',
                type=int, 
                default=100,
                help='training epochs'
        )
        ap.add_argument(
                '--class_weight_balance_mode',
                type=bool,
                default=True,
                help='whether to enable class weights mode to deal with classs unbalance'
        )
        ap.add_argument(
                '--model',
                type=str,
                default="SimpleNet",
                help="which model to use to train"
        )

        args = vars(ap.parse_args())
        num_classes = len([f for f in os.listdir(os.path.join('/Users/yuhua.cheng/Opt/temp/traffic_sign/data/{0}'.format(args['dataset']),'train')) 
                        if os.path.isdir(os.path.join('/Users/yuhua.cheng/Opt/temp/traffic_sign/data/{0}/train/'.format(args['dataset']),f))])
        print("num_classes:", num_classes)
        num_train_samples = len(glob.glob('/Users/yuhua.cheng/Opt/temp/traffic_sign/data/{0}/train/*/*.ppm'.format(args['dataset'])))
        num_val_samples = len(glob.glob('/Users/yuhua.cheng/Opt/temp/traffic_sign/data/{0}/val/*/*.ppm'.format(args['dataset'])))
        if args['class_weight_balance_mode']:
                trained_model_path = './models/{0}_with_class_weights.h5'.format(args['dataset'])
        else:
                trained_model_path = './models/{0}_without_class_weights.h5'.format(args['dataset'])
         
        train_gen = ImageDataGenerator(
                    rescale = 1/255.,
                    samplewise_center=True,
                    samplewise_std_normalization=True,
                    rotation_range=15,
                    zoom_range=0.15,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    horizontal_flip=True,
                    )
        val_gen = ImageDataGenerator(
                    rescale = 1/255.,
                    samplewise_center=True,
                    samplewise_std_normalization=True
                    )

        train_iter = train_gen.flow_from_directory('/Users/yuhua.cheng/Opt/temp/traffic_sign/data/{0}/train'.format(args['dataset']), 
                            target_size=args['input_shape'][0:2], 
                            batch_size=args['batch_size'],
                            # color_mode='grayscale',
                            # save_to_dir='./aug_train',
                            class_mode='categorical', 
                            interpolation='bicubic')

        val_iter = train_gen.flow_from_directory('/Users/yuhua.cheng/Opt/temp/traffic_sign/data/{0}/val'.format(args['dataset']), 
                            target_size=args['input_shape'][0:2], 
                            batch_size=args['batch_size'],
                            # color_mode='grayscale',
                            # save_to_dir='./aug_val',
                            class_mode='categorical',
                            interpolation='bicubic')
        # 针对样本不均衡问题进行weight balance
        class_weight = {}
        counter = Counter(train_iter.classes)
        max_val = float(max(counter.values()))
        class_weights = {class_id:max_val/num_images for class_id, num_images in counter.items()}
        print("class_weights for samples:", class_weights)
        # 
        model = locals()[args['model']](input_shape=args['input_shape'], num_classes=num_classes)
        # sgd = SGD(lr=1e-1, decay=1e-6, momentum=0.9, nesterov=True)
        sgd = keras.optimizers.Adadelta()

        # create callbacks
        tensorboard = callbacks.TensorBoard(log_dir='./logs', write_graph=False)
        learning_rate = callbacks.LearningRateScheduler(create_lr_schedule(epochs=args['epochs'], lr_base=0.01, mode='progressive_drops'))
        callbacks = [tensorboard, learning_rate]

        # compile the model
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

        # train the model
        if args['class_weight_balance_mode']:
                history = model.fit_generator(
                    generator = train_iter,
                    steps_per_epoch = num_train_samples // args['batch_size'],
                    epochs=args['epochs'],
                    validation_data = val_iter,
                    validation_steps = num_val_samples // args['batch_size'],
                    class_weight = class_weights,
                    verbose = 1,
                    callbacks = callbacks)
        else:
                history = model.fit_generator(
                    generator = train_iter,
                    steps_per_epoch = num_train_samples // args['batch_size'],
                    epochs = args['epochs'],
                    validation_data = val_iter,
                    validation_steps = num_val_samples // args['batch_size'],
                    verbose = 1,
                    callbacks = callbacks)
                
             
        model.save(trained_model_path)
