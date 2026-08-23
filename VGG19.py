import os
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Parameters
BATCH_SIZE = 32
INPUT_SHAPE = (224, 224, 3)
N_CLASSES = 3
LEARNING_RATE_BASE = 0.001  # Learning rate for initial training
LEARNING_RATE_FT = 0.0001  # Learning rate for fine-tuning
N_EPOCHS_BASE = 10
N_EPOCHS_FT = 10
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "Resized")
CLASS_NAMES = ["amla", "herda", "bherda"]

# Paths for saving models
BASE_MODEL_PATH = "vgg19_base_model.keras"
FINE_TUNED_MODEL_PATH = "vgg19_fine_tuned_model.keras"

# Ensure dataset structure
for cls in CLASS_NAMES:
    if not os.path.exists(os.path.join(DATASET_DIR, cls)):
        raise FileNotFoundError(
            f"Class directory not found: {os.path.join(DATASET_DIR, cls)}"
        )

# Data Generators
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.15,  # 15% of training data for validation
)

train_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    classes=CLASS_NAMES,
    class_mode="categorical",
    subset="training",
)

valid_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    classes=CLASS_NAMES,
    class_mode="categorical",
    subset="validation",
)


# Model Creation
def create_vgg19_model(input_shape, n_classes, optimizer, fine_tune=0):
    base_model = VGG19(include_top=False, weights="imagenet", input_shape=input_shape)

    # Unfreeze last `fine_tune` layers for fine-tuning
    if fine_tune > 0:
        for layer in base_model.layers[:-fine_tune]:
            layer.trainable = False
        for layer in base_model.layers[-fine_tune:]:
            layer.trainable = True
    else:
        for layer in base_model.layers:
            layer.trainable = False

    # Add custom layers on top
    x = Flatten()(base_model.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(n_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


# Check if a fine-tuned model already exists
if os.path.exists(FINE_TUNED_MODEL_PATH):
    model = load_model(FINE_TUNED_MODEL_PATH)
    print("Loaded fine-tuned model.")
else:
    # Initial Training (Freeze All Pre-trained Layers)
    if not os.path.exists(BASE_MODEL_PATH):
        optimizer_base = Adam(learning_rate=LEARNING_RATE_BASE)
        model = create_vgg19_model(INPUT_SHAPE, N_CLASSES, optimizer_base, fine_tune=0)

        # Callbacks for Base Training
        checkpoint_base = ModelCheckpoint(
            BASE_MODEL_PATH,
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=1,
        )
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, mode="min"
        )

        # Train the Model (Base)
        model.fit(
            train_gen,
            validation_data=valid_gen,
            epochs=N_EPOCHS_BASE,
            callbacks=[checkpoint_base, early_stopping],
            verbose=1,
        )

        print("Base model training completed and saved.")
    else:
        print("Base model found. Loading base model.")
        model = load_model(BASE_MODEL_PATH)

    # Fine-Tuning (Unfreeze Top Layers)
    optimizer_ft = Adam(learning_rate=LEARNING_RATE_FT)
    model = create_vgg19_model(
        INPUT_SHAPE, N_CLASSES, optimizer_ft, fine_tune=4
    )  # Unfreeze top 4 layers

    # Callbacks for Fine-Tuning
    checkpoint_ft = ModelCheckpoint(
        FINE_TUNED_MODEL_PATH,
        save_best_only=True,
        monitor="val_loss",
        mode="min",
        verbose=1,
    )
    early_stopping_ft = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True, mode="min"
    )

    # Train the Model (Fine-Tuning)
    model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=N_EPOCHS_FT,
        callbacks=[checkpoint_ft, early_stopping_ft],
        verbose=1,
    )

    print("Fine-tuned model training completed and saved.")

# Evaluate on Validation Set
val_loss, val_acc = model.evaluate(valid_gen)
print(f"Validation Accuracy After Fine-Tuning: {val_acc * 100:.2f}%")
