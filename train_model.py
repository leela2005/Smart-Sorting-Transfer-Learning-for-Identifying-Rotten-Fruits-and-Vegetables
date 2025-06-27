import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Ensure model directory exists
os.makedirs('model', exist_ok=True)

# Data preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    r'C:\smart_sorting',
    target_size=(150, 150),
    batch_size=64,
    class_mode='sparse',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    r'C:\smart_sorting',
    target_size=(150, 150),
    batch_size=64,
    class_mode='sparse',
    subset='validation'
)

print("Class indices:", train_generator.class_indices)
print("Number of classes:", len(train_generator.class_indices))

# Load pre-trained VGG16 model
vgg = VGG16(weights='imagenet', include_top=False, input_shape=(150,150, 3))
for layer in vgg.layers:
    layer.trainable = False

# Build model
model = Sequential([
    vgg,
    Flatten(),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint('model/healthy_vs_rotten.h5', save_best_only=True, monitor='val_accuracy', mode='max')
earlystop = EarlyStopping(monitor='val_loss', patience=3)

# Train model
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5,
    callbacks=[checkpoint, earlystop]
)
