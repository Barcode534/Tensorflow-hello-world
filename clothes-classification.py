from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__) # check tf running properly

fashion_mnist = keras.datasets.fashion_mnist #import dataset you can use 'keras.datasets.fashion_mnist' or 'keras.datasets.mnist'

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() # assign tuples images and labels, test images and labels.
#print(train_images) this is a list of lists. Image data stored as rows/cols of values between 0 and 255 I think?
#print(train_labels)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'] # define options of classification

#class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] #use numbers with mnist, and clothing class names with fashion_mnist
#
print(train_images.shape) # show shape of images
print(len(train_labels)) # show number of options
print(train_labels)
print(test_images.shape)
print(len(test_labels))

plt.figure() # initialise
plt.imshow(train_images[0]) # what you are going to show
plt.colorbar() # show color bar on right hand side
plt.grid(False) # hide grid lines.
plt.show() # actually make the pop up appear

train_images = train_images / 255.0 # standardise all values to between 0 and 1
test_images = test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(25): #we will show 25 images
    plt.subplot(5,5,i+1) #shape is 5x5 grid.
    plt.xticks([]) #empty list to disable xticks
    plt.yticks([])
    plt.grid(False) #hide grid lines
    plt.imshow(train_images[i], cmap=plt.cm.binary) #image, color map optional. I believe binary shows black and white
    plt.xlabel(class_names[train_labels[i]]) #show label class
plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), #flatten 28x28 into 1x 784
    keras.layers.Dense(128, activation=tf.nn.relu), #784 inputs to 128 neurons to 10 categories output
    keras.layers.Dense(10, activation=tf.nn.softmax) #softmax means all output values which could be negative, or not summing to 1, will sum to 1 and be in range(0,1)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']) #optimizer - the type of learning? i think. #loss = type. seemed to work ok swapping for categorical

model.fit(train_images, train_labels, epochs=5) #5 epochs, using training images and labels. One epoch is one sweep over all training data

test_loss, test_acc = model.evaluate(test_images, test_labels) #loss and acc based on test data.

print('Test accuracy:', test_acc) #show accuracy

predictions = model.predict(test_images) #make predictions for test images
print(predictions[0]) #show sample 1 predictions
print(np.argmax(predictions[0])) # show best prediction
print(test_labels[0]) #show true value


def plot_image(i, predictions_array, true_label, img): #helper function
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label): #helper function
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()

# Grab an image from the test dataset
img = test_images[0]

print(img.shape)

img = (np.expand_dims(img,0))

print(img.shape)

predictions_single = model.predict(img)

print(predictions_single)

plot_value_array(0, predictions_single, test_labels)
plt.xticks(range(10), class_names, rotation=45)
plt.show()

prediction_result = np.argmax(predictions_single[0])
print(prediction_result)