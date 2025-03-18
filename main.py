import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from fastapi import FastAPI, File, UploadFile
import uvicorn
from PIL import Image
import numpy as np
import io
import os

# Initialize FastAPI
app = FastAPI()

# Load dataset (Modify path)
dataset_path = "path_to_your_dataset"  # Update this
batch_size = 32
img_size = (150, 150)

# Load dataset for training
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

class_names = train_ds.class_names  # Get class labels

# Define model
model = models.Sequential([
    layers.Rescaling(1./255),
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model if not exists
model_path = "trained_model.h5"
if not os.path.exists(model_path):
    print("Training model...")
    model.fit(train_ds, epochs=5)  # Adjust epochs as needed
    model.save(model_path)
    print("Model trained and saved!")

# Load trained model
model = tf.keras.models.load_model(model_path)

# API Endpoint for predictions
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image
    image = Image.open(io.BytesIO(await file.read()))
    image = image.resize(img_size)  # Resize to match model input
    img_array = np.array(image) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    return {"prediction": predicted_class}

# Run API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
