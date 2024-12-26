import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns
# Define paths
train_dir = '/kaggle/input/guava-fruit-disease-dataset/Guava Fruit Disease Dataset/GuavaDiseaseDataset/GuavaDiseaseDataset/train'
test_dir = '/kaggle/input/guava-fruit-disease-dataset/Guava Fruit Disease Dataset/GuavaDiseaseDataset/GuavaDiseaseDataset/test'
val_dir = '/kaggle/input/guava-fruit-disease-dataset/Guava Fruit Disease Dataset/GuavaDiseaseDataset/GuavaDiseaseDataset/val'
# Image dimensions
img_height = 224
img_width = 224
batch_size = 32 
# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(img_height, img_width),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(val_dir,
                                                        target_size=(img_height, img_width),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')

test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(img_height, img_width),
                                                  batch_size=batch_size,
                                                  class_mode='categorical',
                                                  shuffle=False)
# Define a custom CNN block with fusion of DenseNet and Residual connections
def custom_cnn_block(x, filters):
    input_tensor = x
    # DenseNet-like block
    x1 = layers.Conv2D(filters, (1,1), padding='same', activation='relu')(x)
    x1 = layers.Conv2D(filters, (3,3), padding='same', activation='relu')(x1)
    
    x2 = layers.Conv2D(filters, (1,1), padding='same', activation='relu')(x)
    x2 = layers.Conv2D(filters, (5,5), padding='same', activation='relu')(x2)
    
    x = layers.Concatenate()([x1, x2])
    x = layers.Conv2D(filters, (1,1), padding='same', activation='relu')(x) 
    
    # Residual connection
    shortcut = layers.Conv2D(filters, (1,1), padding='same')(input_tensor)
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

# Build the model using Functional API
inputs = layers.Input(shape=(img_height, img_width, 3))

# Initial layers
x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D(pool_size=(2,2))(x)
x = inputs
# # Custom CNN blocks with increasing filters
x = custom_cnn_block(x, 64)
x = layers.MaxPooling2D(pool_size=(2,2))(x)

x = custom_cnn_block(x, 128)
x = layers.MaxPooling2D(pool_size=(2,2))(x)

x = custom_cnn_block(x, 256)
x = layers.MaxPooling2D(pool_size=(2,2))(x)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dense(128, activation='relu')(x)
# Output layer
outputs = layers.Dense(3, activation='softmax')(x)
# Compile the model
model = models.Model(inputs, outputs)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# Print model summary
model.summary()
