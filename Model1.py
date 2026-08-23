import os
import shutil
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
original_data_dir = os.path.join(SCRIPT_DIR, "Resized")
base_dir = os.path.join(SCRIPT_DIR, "SplitDataset")
model_path = os.path.join(SCRIPT_DIR, "final_custom_model.keras")

if os.path.exists(model_path):
    print("Loading saved model...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")


def split_dataset():
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    test_dir = os.path.join(base_dir, "test")

    for dir_path in [train_dir, val_dir, test_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    for class_name in os.listdir(original_data_dir):
        class_path = os.path.join(original_data_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        for split_dir in [train_dir, val_dir, test_dir]:
            class_split_path = os.path.join(split_dir, class_name)
            if not os.path.exists(class_split_path):
                os.makedirs(class_split_path)

        # Shuffle and split data
        images = os.listdir(class_path)
        random.shuffle(images)
        train_size = int(len(images) * 0.7)
        val_size = int(len(images) * 0.15)

        train_images = images[:train_size]
        val_images = images[train_size : train_size + val_size]
        test_images = images[train_size + val_size :]

        for img in train_images:
            shutil.copy(
                os.path.join(class_path, img), os.path.join(train_dir, class_name, img)
            )
        for img in val_images:
            shutil.copy(
                os.path.join(class_path, img), os.path.join(val_dir, class_name, img)
            )
        for img in test_images:
            shutil.copy(
                os.path.join(class_path, img), os.path.join(test_dir, class_name, img)
            )

    print("Dataset split complete.")


if not os.path.exists(base_dir):
    split_dataset()

train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

N_CLASSES = len(
    [
        d
        for d in os.listdir(original_data_dir)
        if os.path.isdir(os.path.join(original_data_dir, d))
    ]
)

# Train model if not already loaded
if not os.path.exists(model_path):
    print("No pre-trained model found. Starting training from scratch...")

    BATCH_SIZE = 32
    IMG_HEIGHT, IMG_WIDTH = 224, 224

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=40,
        width_shift_range=0.25,
        height_shift_range=0.25,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        channel_shift_range=30.0,
        fill_mode="nearest",
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )

    WEIGHT_DECAY = 1e-4
    model = Sequential(
        [
            # Block 1: 224x224x3 -> 112x112x64
            Conv2D(
                64,
                (3, 3),
                padding="same",
                activation="relu",
                kernel_regularizer=l2(WEIGHT_DECAY),
                input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
            ),
            BatchNormalization(),
            Conv2D(
                64,
                (3, 3),
                padding="same",
                activation="relu",
                kernel_regularizer=l2(WEIGHT_DECAY),
            ),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            # Block 2: 112x112x64 -> 56x56x128
            Conv2D(
                128,
                (3, 3),
                padding="same",
                activation="relu",
                kernel_regularizer=l2(WEIGHT_DECAY),
            ),
            BatchNormalization(),
            Conv2D(
                128,
                (3, 3),
                padding="same",
                activation="relu",
                kernel_regularizer=l2(WEIGHT_DECAY),
            ),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            # Block 3: 56x56x128 -> 28x28x256
            Conv2D(
                256,
                (3, 3),
                padding="same",
                activation="relu",
                kernel_regularizer=l2(WEIGHT_DECAY),
            ),
            BatchNormalization(),
            Conv2D(
                256,
                (3, 3),
                padding="same",
                activation="relu",
                kernel_regularizer=l2(WEIGHT_DECAY),
            ),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.3),
            # Block 4: 28x28x256 -> 14x14x512
            Conv2D(
                512,
                (3, 3),
                padding="same",
                activation="relu",
                kernel_regularizer=l2(WEIGHT_DECAY),
            ),
            BatchNormalization(),
            Conv2D(
                512,
                (3, 3),
                padding="same",
                activation="relu",
                kernel_regularizer=l2(WEIGHT_DECAY),
            ),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.3),
            # Classification Head — GlobalAveragePooling reduces overfitting
            GlobalAveragePooling2D(),
            Dense(512, activation="relu", kernel_regularizer=l2(WEIGHT_DECAY)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation="relu", kernel_regularizer=l2(WEIGHT_DECAY)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(N_CLASSES, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    # Callbacks — ReduceLROnPlateau helps find optimal min
    checkpoint = ModelCheckpoint(
        model_path, save_best_only=True, monitor="val_accuracy", mode="max", verbose=1
    )
    early_stop = EarlyStopping(
        monitor="val_accuracy", patience=10, restore_best_weights=True, mode="max"
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
    )

    # Train Model
    history = model.fit(
        train_gen,
        epochs=50,
        validation_data=val_gen,
        callbacks=[checkpoint, early_stop, reduce_lr],
        verbose=1,
    )

    print(f"Model trained and saved at {model_path}.")

# Test Data Generator
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False,
)

test_loss, test_accuracy = model.evaluate(test_gen, verbose=1)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")
