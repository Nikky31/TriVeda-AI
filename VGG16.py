import os
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
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
CLASS_NAMES = ['amla', 'herda', 'bherda']

# Model Save Paths
BASE_MODEL_PATH = 'vgg16_base_model.keras'
FINE_TUNED_MODEL_PATH = 'vgg16_fine_tuned_model.keras'

# Ensure dataset structure
for cls in CLASS_NAMES:
    if not os.path.exists(os.path.join(DATASET_DIR, cls)):
        raise FileNotFoundError(f"Class directory not found: {os.path.join(DATASET_DIR, cls)}")

# Data Generators
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.15  # 15% of training data for validation
)

train_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    classes=CLASS_NAMES,
    class_mode='categorical',
    subset='training'
)

valid_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    classes=CLASS_NAMES,
    class_mode='categorical',
    subset='validation'
)

# Model Creation Function
def create_model(input_shape, n_classes, optimizer, fine_tune=0):
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)

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
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Base Training
if os.path.exists(BASE_MODEL_PATH):
    print("Loading previously trained base model...")
    model = load_model(BASE_MODEL_PATH)
else:
    print("Training base model...")
    optimizer_base = Adam(learning_rate=LEARNING_RATE_BASE)
    model = create_model(INPUT_SHAPE, N_CLASSES, optimizer_base, fine_tune=0)

    # Callbacks for Base Training
    checkpoint_base = ModelCheckpoint(BASE_MODEL_PATH, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, mode='min')

    # Train the Model (Base)
    model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=N_EPOCHS_BASE,
        callbacks=[checkpoint_base, early_stopping],
        verbose=1
    )
    print("Base model training complete. Saved to:", BASE_MODEL_PATH)

# Fine-Tuning
if os.path.exists(FINE_TUNED_MODEL_PATH):
    print("Loading previously fine-tuned model...")
    model = load_model(FINE_TUNED_MODEL_PATH)
else:
    print("Fine-tuning model...")
    optimizer_ft = Adam(learning_rate=LEARNING_RATE_FT)
    model = create_model(INPUT_SHAPE, N_CLASSES, optimizer_ft, fine_tune=4)  # Unfreeze top 4 layers

    # Callbacks for Fine-Tuning
    checkpoint_ft = ModelCheckpoint(FINE_TUNED_MODEL_PATH, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, mode='min')

    # Train the Model (Fine-Tuning)
    model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=N_EPOCHS_FT,
        callbacks=[checkpoint_ft, early_stopping],
        verbose=1
    )
    print("Fine-tuning complete. Saved to:", FINE_TUNED_MODEL_PATH)

# Evaluate the Model
val_loss, val_acc = model.evaluate(valid_gen)
print(f"Validation Accuracy After Fine-Tuning: {val_acc * 100:.2f}%")
