#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/AAliArslan/AAliArslan/blob/main/CarDetection_v1.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


# Import required packages
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import time


# In[ ]:

#Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="/home/funtastic/Downloads/Car-Detection-MobileNetV2-main/lite_V3.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Load the MobileNetV2 model
#model = MobileNetV2(weights="imagenet")


# In[ ]:


# Load the input image
def load_image(image_path):
  image = load_img(image_path, target_size=(224, 224))
  return image


# In[ ]:


# Define an Ind array that holds the important indexes for us
Ind =  [407, 408, 436, 444, 450, 468, 479, 475, 511, 517, 537, 555, 561, 565, 
        569, 573, 575, 581, 586, 595, 603, 609, 612, 621, 627, 654, 656, 661, 
        665, 670, 671, 675, 690, 705, 717, 730, 734, 751, 757, 779, 802, 803, 
        817, 829, 847, 856, 864, 866, 867, 870, 874, 880]

Threshold = 0.10


# In[ ]:


# Preprocess the input image
def preprocess_image(image):
  image = img_to_array(image)
  image = preprocess_input(image)
  # Add a new axis to the tensor
  image = tf.expand_dims(image, axis=0)
  return image


# In[ ]:


# Classify the input image using the MobileNetV2 model
def classify_image(image):
  interpreter.set_tensor(input_details[0]["index"],image)
  interpreter.invoke()
  
  output_data = interpreter.get_tensor(output_details[0]["index"])
  preds = output_data
  return preds
  


# In[ ]:


def detect_car(image):
  processed_image = preprocess_image(image)
  preds = classify_image(processed_image)
  # Find the index of the highest value in the predictions array
  indexes = np.argmax(preds)

  # convert the list to a NumPy array
  my_array = np.array(preds)

  # find the index of every element with value higher than 0.1
  indices = np.where(my_array > 0.05)

  # We add up all the probabilities that corresponds to desired indexes in Prob
  Prob = 0

  for i in Ind:
    Prob = Prob + preds[0][i]

    # Return true if the probability is higher than %15
  return Prob > Threshold


# In[ ]:


# Test the car detection module
image = load_image("test_image.jpg")
time1 = time.time()
if detect_car(image):
  print("A vehicle is present in the image")
  time2 = time.time() - time1
  print(time2)
else:
  print("No vehicle is present in the image")
  time2 = time.time() - time1
  print(time2)


# # Yeni Bölüm
