from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from tensorflow.keras.models import Sequential


from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dropout

import tensorflow as tf
from matplotlib import pyplot as plt


dataset_directory = r"flower_photos"
#test_directory = "path_to_test_folder"

# training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory=dataset_directory,
    image_size=(180, 180),
    batch_size=32,
    validation_split=0.2,
    subset="training",
    seed=123,
    labels='inferred',
    label_mode='categorical',
)
# validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    directory=dataset_directory,
    image_size=(180, 180),
    batch_size=32,
    validation_split=0.2,
    subset="validation",
    seed=123,
    labels='inferred',
    label_mode='categorical',
)

# Load and preprocess the test dataset
"""test_ds = tf.keras.utils.image_dataset_from_directory(
    directory=test_directory,
    image_size=(180, 180),
    batch_size=32,
    seed=123,
    labels='inferred',
    label_mode='categorical',
)"""

# model architecture
model = Sequential()
model.add(Rescaling(1./255, input_shape=(180, 180, 3)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))


# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model
history = model.fit(
    train_ds,
    epochs=20,
    validation_data=val_ds
)

def plot_learning_curve(history):
    plt.figure(figsize=(12, 6))

    # Plot training and validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout()
    plt.show()


plot_learning_curve(history)

# Evaluate the model on the test dataset
"""test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")"""