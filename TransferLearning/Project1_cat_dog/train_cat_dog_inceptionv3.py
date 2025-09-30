import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras import layers, models

# Paths
original_data_dir = '/tf/code/dataset/PetImages'
model_output_dir = '/tf/code/models'
os.makedirs(model_output_dir, exist_ok=True)

# Data generators (note target_size 299×299)
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    original_data_dir,
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    original_data_dir,
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

print("Class Indices:", train_generator.class_indices)

# Build model
base_model = InceptionV3(
    weights='imagenet',
    include_top=False,
    input_shape=(299, 299, 3)
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    train_generator,
    epochs=6,
    validation_data=val_generator,
    verbose=1
)

# Save model
model_path = os.path.join(model_output_dir, 'cat_dog_transfer_InceptionV3_model.h5')
model.save(model_path)

# Plot & save accuracy
plt.figure(figsize=(8,6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('InceptionV3 Transfer Learning Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(model_output_dir, 'accuracy_plot_InceptionV3.png'))
plt.close()

# Confusion matrix
val_generator.reset()
preds = (model.predict(val_generator) > 0.5).astype('int32').flatten()
labels = val_generator.classes

cm = confusion_matrix(labels, preds)
plt.figure(figsize=(6,6))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=train_generator.class_indices.keys(),
    yticklabels=train_generator.class_indices.keys()
)
plt.title('Confusion Matrix (InceptionV3)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(os.path.join(model_output_dir, 'confusion_matrix_InceptionV3.png'))
plt.close()
