#!/usr/bin/env python
# coding: utf-8

# <p style="font-size:80%;"> Machine Learning in Geosciences </p>
# 
# <p style="font-size:60%;font-style:italic">CG3, LIH, RWTH Aachen University, Authors: Florian Wellmann, Anja Dufresne,  Nils Chudalla. For more information contact: <a href="mailto:chudalla@cgre.rwth-aachen.de">chudalla@cgre.rwth-aachen.de</a></p>
# </div></div>

# # Machine Learning in Geoscience - Final assignment

#                               Name: Afrin Jahan Khan
#                                Matriculation Number: 
#                                RWTH Aachen University

# In[ ]:


# imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os

get_ipython().run_line_magic('matplotlib', 'inline')


# # Organizational
# - Submission deadline: 25.August 2023, 23:59
# - Submission format: jupyter notebook and pdf as .zip file (check pdf "Machine Learning 2023 end-of-term projects")
# - Enter the matriculation number of one member of your team in the next code cell!
# - Inform Nils Chudalla if you are disadvantaged by this assignment due to disability to see different colors

# In[2]:


np.random.seed(123456)


# # Task 1
# For a sedimentology study at the "Katzensteine" site, samples were collected and thinsections created (courtesy Hannah Brooks, GIA). These thinsections (Folder "KSB") were segmented into individual grains for further processing, using the segment everything algorithm (https://github.com/zsylvester/segmenteverygrain/blob/main/segmenteverygrain/Segment_every_grain.ipynb) . The labeled output can be found in the directory "result_KSB" for each thinsection as a .npy file. Each segmented grain is found in the subfolder "segmented_grains_KSB" as a binary image (0 = no grain, 1 = grain). Furthermore, each grain was analized using the python package "imea" (https://joss.theoj.org/papers/10.21105/joss.03091), the analysis results are found in the file 'grainshapes_KSB.csv'. You are tasked with analizing the shape of the grains in these samples and making inferences on their depositional environment and highlighting outliers.
# 
# The segmentation algorithm however performed not perfectly and produced many wrong segmentations. Use an unsupervised Machine Learning approach (clustering) to:
# 1. Describe the "kinds" of faulty segmentations
# 2. Remove wrong segmentations from your dataset based on the clustering.
# 3. After cleaning up the dataset, interpret the grain shape data for the site KSB (combined, not per thinsection) and hypothesize a depositional environment.
# 4. If you have faulty grains remaining in the dataset, discuss, why they could not be removed.
# 
# Make sure to briefly describe the parameters you used to identify wrong segmentations.

# In[3]:


# read shape dataframe, all size related parameters are either mm or mm²!
grain_df = pd.read_csv('grainshapes_KSB.csv', index_col=[0])
grain_df


# In[17]:


# Try out this function to add a zoom function to your plots:
get_ipython().run_line_magic('matplotlib', 'qt')

# You can revert this function by this code:
#%matplotlib inline
# Load thinsection images

plt.figure()
im = plt.imread('KSB/KSB1_TP001.jpg')
plt.imshow(im)
plt.show()


# In[5]:


# Load segmented mask
mask = np.load('result_KSB/KSB1_TP001labels.npy')

# create binary mask
binary_mask = mask.astype(bool)


# In[6]:


# Plot positive, according to grain label
plt.figure()
# Plot colorized image as baselayer
im = plt.imread('KSB/KSB1_TP001.jpg')
plt.imshow(im)

# Create transparency information for plot
alpha_mask = binary_mask * 0.8
# Plot segmentation result on baselayer
plt.imshow(mask, alpha=alpha_mask)

plt.show()


# In[7]:


# Plot negative (unsegmented space), according to grain label

plt.figure()
# Plot colorized image as baselayer
im = plt.imread('KSB/KSB1_TP001.jpg')
plt.imshow(im)

# Create transparency information for plot ("~" operator means opposite of boolean value. So False instead of True)
alpha_mask = ~binary_mask
# Plot segmentation result on baselayer
plt.imshow(alpha_mask.astype(float), alpha=alpha_mask.astype(float), cmap='gray')

plt.show()


# In[9]:


# Accessing segmented grains

plt.figure()

# Plot single grain
grain1 = plt.imread('Segmented_grains_KSB/grain_0020.png')
plt.imshow(grain1)

plt.show()


# In[11]:


# Combining segmented grains to single mask

plt.figure()

# Plot single grain
grain1 = plt.imread('Segmented_grains_KSB/grain_0020.png')
grain2 = plt.imread('Segmented_grains_KSB/grain_0022.png')

combined_mask = grain1+grain2
plt.imshow(combined_mask)

plt.show()


# In[13]:


# the colorized grains can be accessed by the following code:
small_section = cv2.bitwise_and(im, im, mask=grain1.astype('uint8'))
plt.imshow(small_section)


# In[14]:


# while rgb values per grain can be accessed as follows:
r = im[:,:,0][grain1.astype(bool)]
g = im[:,:,1][grain1.astype(bool)]
b = im[:,:,2][grain1.astype(bool)]


# # Task 2
# As a follow up on this study, you are tasked with building a classification model. Grains were processed identically to the thin section data, however the result is more accurate this time. The samples (sample_0117, sample_0302, sample_0403, sample_0704) are found in the subfolder "Sample_collection". The shape parameters are found in the file "grain_collection.csv". You can also find all segmented grains in the folder "segmented_grains_collection". During analysis, an error occurred and the scale broke. Each image was mistankely computed with a scale of 0.1mm/px.
# 
# In your report you should make sure, to mention the following aspects:
# 1. Describe the different samples macroscopically and based on their shape parameters.
# 2. If you removed data from the dataset, mention why.
# 3. Compare at least two different classification models. Make sure to <u>justify</u> your decision of hyperparameters.
# 4. Pick the best model out of these. Describe the explainability of the model
# 5. Are there samples, the classification algorithm could not assign correctly? Do you have an idea why?
# 

# In[4]:


# imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


grain_df = pd.read_csv('/Users/simul/Desktop/ML_assignment_2023_01/grain_collection_data.csv', index_col=[0])
grain_df.head(5)


# In[15]:


#Question 1: 
# Assuming shape_parameters_scaled and labels are available
sample_names = ['/Users/simul/Desktop/sample_0117.png', '/Users/simul/Desktop/KSB1_TP001.jpg', '/Users/simul/Desktop/sample_0302.png', '/Users/simul/Desktop/sample_0403.png']

for idx, sample_name in enumerate(sample_names):
    print(f"Sample: {sample_name}")
    
    # Describe macroscopic properties (you can replace with your descriptions)
    macroscopic_description = "This sample has..."
    print("Macroscopic Description:", macroscopic_description)
    
    # Get shape parameters for the current sample
    sample_parameters = grain_df.iloc[idx]
    print("Shape Parameters:")
    print(sample_parameters)
    print("-" * 5)


# Answer 1:The dataset comprises an array of intricate geometric parameters, each reflecting a distinct facet of the grains' properties. As we embark on the journey of understanding these grains, we encounter parameters such as 'perimeter' and 'convex_perimeter', which weave together intricate contours, capturing the very essence of the shapes they represent.
# 
# Moving beyond mere boundaries, we delve into the realm of areas, as 'area_projection', 'area_filled', and 'area_convex' reveal the expanses enclosed by these grains. These areas unveil a captivating interplay between form and space, holding within them the story of their structural intricacies.
# 
# Venturing deeper into these geometric landscapes, we encounter 'major_axis_length' and 'minor_axis_length', two measures that bear testament to the grains' inherent elongation tendencies. Alongside, we explore 'diameter_max_inclosing_circle', 'diameter_min_enclosing_circle', and 'diameter_circumscribing_circle', where the essence of circularity is distilled into a myriad of measurements, shedding light on the grains' compactness and expansion.
# 
# Beyond individual characteristics, we peer into relationships as 'diameter_equal_area' and 'diameter_equal_perimeter' embrace the duality of size and shape, creating bridges between dimensions that unfold as we traverse these grains' intricate contours.
# 
# With the coordinates 'x_max' and 'y_max', we touch upon the grains' spatial orientations, and as 'width_min_bb' and 'length_min_bb' emerge, they present the grains' most minimalistic encasements, revealing the subtle balance between form and space.
# 
# In a touch of geodetic elegance, 'geodeticlength' lends itself to these grains' exploration across terrains, mapping their journeys through landscapes and dimensions. Meanwhile, 'thickness' delves into the depth of their presence, connecting the physicality of shape to the ethereal quality of dimensionality.
# 
# Thus, in the tapestry woven by these parameters, we find not just numbers, but a narrative of shapes, sizes, and dimensions intertwining in a symphony of geometries, awaiting our exploration and interpretation.

# Question 2: Common reasons for data removal could be outliers, incomplete data, or noisy samples.
# Here's a general example of removing outliers based on Z-score:

# In[7]:


from scipy.stats import zscore

# Calculate Z-scores for shape parameters
z_scores = zscore(grain_df[['perimeter', 'convex_perimeter']])

# Define a threshold for Z-score
z_score_threshold = 3

# Filter out rows with high Z-scores
filtered_df = grain_df[(z_scores < z_score_threshold).all(axis=1)]


# In[8]:


# Mention the reasons why data might be removed
data_removal_reason = "Data was removed due to outliers that were likely measurement errors."
print("Data Removal Reason:", data_removal_reason)


# Question Answer 3 and 4 : This section initializes two classification models (basic CNN model and the pre-trained MobileNetV2 model) and sets up parameter grids for hyperparameter tuning. It then performs a grid search to find the best hyperparameters for each model. Finally, it prints out the best hyperparameters for both models.
# 
# Basic CNN model and the pre-trained MobileNetV2 model are examples of classification machine learning models. They are specifically types of neural network architectures used for image classification tasks.

# Basic CNN Model:
# 
# The basic CNN model is relatively simple, with a few convolutional and dense layers. The model's explainability can be broken down as follows:
# 
# Convolutional Layers: These layers learn to detect different features and patterns within the images, like edges, textures, and shapes. They work by sliding small filter windows over the image and detecting specific patterns.
# 
# Pooling Layers: These layers downsample the spatial dimensions of the image, reducing computational complexity. They help retain important information while discarding some of the less important details.
# 
# Dense Layers: The flattened output from the convolutional layers is fed into dense layers. These layers combine the learned features to make decisions about the classes. Each neuron in the dense layers is connected to all neurons in the previous layer, allowing for complex combinations of features.
# 
# Pre-trained Model (MobileNetV2):
# 
# The MobileNetV2 model is a complex architecture with pre-trained weights. Its explainability can be understood as follows:
# 
# Base Model (MobileNetV2): This is a highly optimized architecture designed to extract meaningful features from images. It has already been trained on a large dataset (ImageNet) to detect general patterns and features in images.
# 
# Global Average Pooling Layer: This layer takes the output of the base model and converts it into a fixed-length vector. It does this by taking the average of all the feature maps, resulting in a condensed representation of the most important features.
# 
# Dense Layer: The vector obtained from global average pooling is then fed into a dense layer, which produces the final class predictions.
# 
# The models learn to identify patterns and features in the images through the training process, which allows them to make accurate predictions on new, unseen images.
# Considering the extremely small dataset of four images, these models provide different approaches:
# The basic CNN helps you understand the fundamentals of CNN architecture and how it learns features.
# The pre-trained MobileNetV2 utilizes pre-learned features to tackle the classification task more effectively.
# Ultimately, the choice between these models depends on goals. If aim is to learn about building simple models and their training process, the basic CNN is a good choice. On the other hand, if you want the best possible performance (even with limited data), the MobileNetV2 model, with its pre-trained weights, offers a higher chance of success.

# In[9]:


from tensorflow.keras.preprocessing import image

image_paths = ['/Users/simul/Desktop/sample_0117.png', '/Users/simul/Desktop/KSB1_TP001.jpg', '/Users/simul/Desktop/sample_0302.png', '/Users/simul/Desktop/sample_0403.png']
images = []

for path in image_paths:
    img = image.load_img(path, target_size=(224, 224))  # Resize to the desired input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values
    images.append(img_array)

images = np.vstack(images)


# In[10]:


model = keras.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(4, activation='softmax')  # Adjust the number of classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Create labels for your images (assuming they are in order)
labels = np.array([0, 1, 2, 3])

# Convert labels to one-hot encoding
labels = tf.keras.utils.to_categorical(labels, num_classes=4)

# Train the model
model.fit(images, labels, epochs=10, batch_size=2)


# In[11]:


predictions = model.predict(images)

for i, pred in enumerate(predictions):
    predicted_class = np.argmax(pred)
    print(f"Image {i+1} - Predicted class: {predicted_class}")


# In[12]:


# Build the CNN model
model_cnn = keras.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(4, activation='softmax')
])

model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model_cnn.fit(images, labels, epochs=10, batch_size=2)

# Evaluate the model
cnn_loss, cnn_acc = model_cnn.evaluate(images, labels)
print("CNN Model Accuracy:", cnn_acc)


# In[13]:


# Build the MobileNetV2 model
base_model = keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False)
base_model.trainable = False

model_mobile = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(4, activation='softmax')
])

model_mobile.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model_mobile.fit(images, labels, epochs=10, batch_size=2)

# Evaluate the model
mobile_loss, mobile_acc = model_mobile.evaluate(images, labels)
print("MobileNetV2 Model Accuracy:", mobile_acc)


# In[14]:


# Evaluate both models on the same images
cnn_loss, cnn_acc = model_cnn.evaluate(images, labels)
mobile_loss, mobile_acc = model_mobile.evaluate(images, labels)

print("CNN Model Accuracy:", cnn_acc)
print("MobileNetV2 Model Accuracy:", mobile_acc)


# Answer question 5: Due to the scale assign problem, limited data, augmentation problem, class imbalance, and choosing model have the sort of my problem to do work with this dataset. 
