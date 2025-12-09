#!/usr/bin/env python3
"""
Model Evaluation on Validation Set
Loads trained VGG model and evaluates on validation split (5 classes)
Uses preprocessed validation data from data/processed/validation
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
from collections import Counter

# Configure GPU
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

def load_validation_data(target_classes=['person', 'cat', 'dog', 'car', 'chair']):
    """Load validation data for 5 classes"""
    print("="*60)
    print("Loading Validation Data")
    print("="*60)
    
    data_dir = "/opt/spark/data/processed/validation"
    val_images, val_labels = [], []
    
    chunk_id = 0
    while os.path.exists(f"{data_dir}/images_chunk_{chunk_id}.npy"):
        imgs = np.load(f"{data_dir}/images_chunk_{chunk_id}.npy")
        labs = np.load(f"{data_dir}/labels_chunk_{chunk_id}.npy", allow_pickle=True)
        val_images.append(imgs)
        val_labels.extend(labs)
        print(f"  Loaded chunk {chunk_id}: {len(imgs)} images")
        chunk_id += 1
    
    val_images = np.concatenate(val_images, axis=0)
    val_labels = np.array(val_labels)
    
    # Filter to target classes
    mask = np.isin(val_labels, target_classes)
    val_images = val_images[mask]
    val_labels = val_labels[mask]
    
    print(f"\nFiltered to {len(val_images)} images")
    
    # Show distribution
    counts = Counter(val_labels)
    for cls in target_classes:
        print(f"  {cls}: {counts.get(cls, 0)} images")
    
    print("="*60)
    return val_images, val_labels

def encode_labels(test_labels, label_mapping):
    """Encode labels to one-hot"""
    label_to_int = {label: int(idx) for label, idx in label_mapping.items()}
    test_labels_int = np.array([label_to_int[label] for label in test_labels])
    test_labels_cat = np.eye(len(label_mapping))[test_labels_int]
    return test_labels_cat, test_labels_int

def evaluate_model(model_path, test_images, test_labels_cat):
    """Evaluate model"""
    print("\n" + "="*60)
    print("Model Evaluation")
    print("="*60)
    
    model = keras.models.load_model(model_path)
    print(f"Model: {model.count_params():,} parameters")
    
    test_loss, test_acc = model.evaluate(test_images, test_labels_cat, batch_size=32, verbose=1)
    
    print("\n" + "="*60)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print("="*60)
    
    return test_acc, test_loss

def main():
    print("="*60)
    print("VGG Model Validation Evaluation")
    print("="*60)
    
    try:
        # Load label mapping
        with open('/opt/spark/data/training/label_mapping.json', 'r') as f:
            label_mapping = json.load(f)
        
        print("\nLabel mapping:")
        for label, idx in label_mapping.items():
            print(f"  {idx}: {label}")
        
        # Load validation data
        val_images, val_labels_str = load_validation_data()
        
        # Encode labels
        val_labels_cat, _ = encode_labels(val_labels_str, label_mapping)
        
        # Evaluate
        model_path = '/opt/spark/data/models/vgg_final.keras'
        evaluate_model(model_path, val_images, val_labels_cat)
        
        print("\nValidation Evaluation Complete!")
        return 0
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
