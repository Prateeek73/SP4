#!/usr/bin/env python3
"""
GPU Training Task - TensorFlow tf.data Streaming Pipeline
Loads training data incrementally using tf.data API to avoid memory issues
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import json
import os

# Configure GPU
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

print("\n" + "="*70, flush=True)
print("GPU Training Task - TensorFlow tf.data Streaming Pipeline", flush=True)
print("="*70, flush=True)

# Load batch metadata
with open('/opt/spark/data/training/batch_info.json') as f:
    batch_info = json.load(f)

train_labels = np.load('/opt/spark/data/training/train_labels.npy')

with open('/opt/spark/data/training/config.json') as f:
    config = json.load(f)

print(f"\nBatch Info:", flush=True)
print(f"  Total images: {batch_info['total_images']}", flush=True)
print(f"  Number of batches: {batch_info['num_batches']}", flush=True)
print(f"  Batch directory: {batch_info['batch_dir']}", flush=True)
print(f"  Labels shape: {train_labels.shape}", flush=True)
print("="*70, flush=True)

# Get batch files
batch_dir = batch_info['batch_dir']
batch_files = sorted([os.path.join(batch_dir, f) for f in os.listdir(batch_dir) if f.startswith('batch_')])
print(f"\nFound {len(batch_files)} batch files:", flush=True)
for bf in batch_files:
    print(f"  - {bf}", flush=True)

# Create tf.data pipeline that streams data from disk
def load_batch_file(batch_file_path, start_idx, end_idx):
    """Load a batch file and return images with their corresponding labels"""
    images = np.load(batch_file_path.numpy().decode('utf-8'))
    start = start_idx.numpy()
    end = end_idx.numpy()
    labels = train_labels[start:end]
    return images, labels

def create_streaming_dataset():
    """Create a tf.data.Dataset that streams batches from disk"""
    
    # Calculate index ranges for each batch file
    batch_ranges = []
    current_idx = 0
    for batch_file in batch_files:
        batch_data = np.load(batch_file)
        num_samples = len(batch_data)
        batch_ranges.append((batch_file, current_idx, current_idx + num_samples))
        current_idx += num_samples
        del batch_data  # Free memory
    
    print(f"\nBatch index ranges:", flush=True)
    for bf, start, end in batch_ranges:
        print(f"  {os.path.basename(bf)}: indices {start}-{end-1} ({end-start} samples)", flush=True)
    
    # Create dataset from batch file paths and ranges
    def generator():
        for batch_file, start_idx, end_idx in batch_ranges:
            images = np.load(batch_file)
            labels = train_labels[start_idx:end_idx]
            
            # Yield each sample individually
            for i in range(len(images)):
                yield images[i], labels[i]
            
            del images  # Free memory after each batch
    
    # Create dataset from generator
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(config['num_classes'],), dtype=tf.float32)
        )
    )
    
    return dataset

print("\n" + "="*70, flush=True)
print("Creating TensorFlow tf.data streaming pipeline...", flush=True)
print("="*70, flush=True)

# Create streaming dataset
train_dataset = create_streaming_dataset()

# Configure dataset for training
train_dataset = train_dataset.repeat()  # Repeat dataset to use all steps
train_dataset = train_dataset.shuffle(buffer_size=1000)  # Shuffle with reasonable buffer
train_dataset = train_dataset.batch(config['batch_size'])  # Batch the data
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)  # Prefetch for performance

print(f"✓ Streaming dataset created", flush=True)
print(f"  - Buffer size: 1000 samples", flush=True)
print(f"  - Batch size: {config['batch_size']}", flush=True)
print(f"  - Prefetching enabled", flush=True)
print("="*70, flush=True)

# Build model
print("\nBuilding VGG model...", flush=True)
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Conv2D(32, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(2),
    layers.Conv2D(64, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(2),
    layers.Conv2D(128, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(config['num_classes'], activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=config['learning_rate']),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"✓ Model built: {model.count_params():,} parameters", flush=True)
print("="*70, flush=True)

# Calculate steps per epoch
steps_per_epoch = int(np.ceil(batch_info['total_images'] / config['batch_size']))
print(f"\nTraining configuration:", flush=True)
print(f"  Total images: {batch_info['total_images']}", flush=True)
print(f"  Batch size: {config['batch_size']}", flush=True)
print(f"  Steps per epoch: {steps_per_epoch}", flush=True)
print(f"  Epochs: {config['epochs']}", flush=True)
print("="*70, flush=True)
print("Starting training with streaming data pipeline...", flush=True)
print("", flush=True)

# Train with streaming dataset
os.makedirs('/opt/spark/data/models', exist_ok=True)

history = model.fit(
    train_dataset,
    steps_per_epoch=steps_per_epoch,
    epochs=config['epochs'],
    verbose=1
)

model.save('/opt/spark/data/models/vgg_final.keras')

print("", flush=True)
print("="*60, flush=True)
print(f"Training complete! Final accuracy: {history.history['accuracy'][-1]:.4f}", flush=True)
print(f"Model saved: /opt/spark/data/models/vgg_final.keras", flush=True)
print("="*60, flush=True)
