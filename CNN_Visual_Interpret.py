#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 18:04:14 2019

@author: christrombley
"""



drive.mount('/content/gdrive/')

##import google drive files
drive.mount('/content/gdrive/')
much_data = np.load('/content/gdrive/My Drive/imgFileNew.npy')
labels = np.load('/content/gdrive/My Drive/labelsNew.npy')


# scale the raw pixel intensities to the range [0, 1] bc pixels are 0 to 255
data = np.array(much_data, dtype="float") ##/ 4630.0
labels = np.array(labels)
data = data.reshape([-1, 50, 50])



(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.3, random_state = 48)

##first num should be number of images
trainX = trainX.reshape([-1, 50, 50])
testX = testX.reshape([-1,50, 50])


from keras.models import model_from_json

# Model reconstruction from JSON file
with open('/content/gdrive/My Drive/model.json', 'r') as f:
    model = model_from_json(f.read())
    

# Load weights into the new model
model.load_weights('/content/gdrive/My Drive/model.h5')

##model.summary()

imgRead = pydicom.dcmread('/content/gdrive/My Drive/1.dcm').pixel_array
#print(imgRead.shape)
img = cv2.resize(imgRead, dsize=(50, 50), interpolation=cv2.INTER_CUBIC)
img = img.reshape([50, 50,1])

model.summary()


print("plot of original image: ")
plt.imshow(imgRead, cmap=plt.cm.bone)
plt.show()


##VISUALIZE FILTERS

x1w = model.get_weights()[0][:,:,0,:]
for i in range(1,26):
  plt.subplot(5,5,i)
  plt.imshow(x1w[:,:,i],interpolation="nearest",cmap="gray")
plt.show()




##VISUALIZE FEATURE MAPS

from keras import models


img_tensor = np.expand_dims(img, axis=0)


layer_outputs = [layer.output for layer in model.layers[:]] 
# Extracts the outputs of the top 12 layers
activation_model = models.Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input

activations = activation_model.predict(img_tensor) 
# Returns a list of five Numpy arrays: one array per layer activation

first_layer_activation = activations[0]
print(first_layer_activation.shape)

plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')

layer_names = []
for layer in model.layers[:12]:
    layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
    
images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
    n_features = layer_activation.shape[-1] # Number of features in the feature map
    size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
    n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols): # Tiles each filter into a big horizontal grid
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, # Displays the grid
                         row * size : (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis') 
    
    
    
    
    
 ##Activation maps
 
 
 
 
 much_data = np.load('/content/gdrive/My Drive/imgFileNew.npy')
labels = np.load('/content/gdrive/My Drive/labelsNew.npy')
data = np.array(much_data, dtype="float") ##/ 4630.0
labels = np.array(labels)



(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.3, random_state = 48)

plt.figure(figsize=(6,6))
plt.imshow(trainX[1][:,:])
plt.title(trainY[1].argmax());



from keras.models import Model
layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(trainX[10].reshape(1,50,50,1))
 
def display_activation(activations, col_size, row_size, act_index): 
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1
            
display_activation(activations, 8, 8, 1)
display_activation(activations, 8, 8, 2)
display_activation(activations, 8, 8, 3)
display_activation(activations, 8, 8, 4)
display_activation(activations, 8, 8, 5)







##Image Occlusion to do



## Sal Maps

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.axis as ax

from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input

from vis.utils import utils
from keras.applications.vgg16 import VGG16, decode_predictions
from vis.visualization import visualize_cam



from vis.visualization import visualize_saliency
from vis.utils import utils
from keras import activations

# Utility to search for layer index by name. 
# Alternatively we can specify this as -1 since it corresponds to the last layer.
layer_idx = utils.find_layer_idx(model, 'dense_2')

# Swap softmax with linear
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

grads = visualize_saliency(model, layer_idx, filter_indices=0,seed_input=img)
# Plot with 'jet' colormap to visualize as a heatmap.
plt.imshow(grads, cmap='jet')
    
    


##CAM


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.axis as ax

from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input

from vis.utils import utils
from keras.applications.vgg16 import VGG16, decode_predictions
from vis.visualization import visualize_cam



from vis.visualization import visualize_saliency
from vis.utils import utils
from keras import activations

# Utility to search for layer index by name. 
# Alternatively we can specify this as -1 since it corresponds to the last layer.
layer_idx = utils.find_layer_idx(model, 'dense_2')

# Swap softmax with linear
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

##look in docs at args maybe? - doesnt look accurate
grads = visualize_cam(model, layer_idx, filter_indices=1,seed_input=img)
# Plot with 'jet' colormap to visualize as a heatmap.
plt.imshow(grads, cmap='jet')


##TSNE


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
%pylab inline



X_tsne = TSNE(learning_rate=100).fit_transform(img)
X_pca = PCA().fit_transform(img)

figure(figsize=(10, 5))
subplot(121)
scatter(X_tsne[:, 0], X_tsne[:, 1])
subplot(122)
scatter(X_pca[:, 0], X_pca[:, 1])

    
    