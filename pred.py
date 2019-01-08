import cv2
import os
import glob
import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model
from imageio import imread

image_files = [f for f in os.listdir('./data/traffic_sign/test') if not f.startswith('.')]
classes = sorted(os.listdir('./data/traffic_sign/val'))
model = load_model('./models/traffic_sign_with_class_weights.h5')
model.summary()
for image_file in image_files:
	img = imread(os.path.join('./data/traffic_sign/test', image_file))
	plt.subplot(1,2,1)
	plt.imshow(img)
	plt.title("img")
	img = cv2.resize(img, (112,112))
	img = img.astype("float32")
	print(np.mean(img, axis=(0,1)))
	img = (img - np.mean(img)) / np.std(img)
	img = np.expand_dims(img, 0)
	prediction = model.predict(img)
	print("prediction:", prediction)
	label = np.argmax(prediction)
	label_image = imread(glob.glob('./data/traffic_sign/train/{0}/*.ppm'.format(classes[label]))[0])
	plt.subplot(1,2,2)
	plt.imshow(label_image)
	plt.title("predicted img")
	plt.show()

