import subprocess
import sys
import os

print("=" * 60)
print("CNN Image Classification with Apache Spark")
print("=" * 60)
print("Note: Running in offline mode. Skipping auto-install.")
print("=" * 60)

import numpy as np
try:
    import cv2
except ImportError:
    print("WARNING: OpenCV not available. Image loading will fail.")
    cv2 = None

from pathlib import Path
from typing import List, Tuple, Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from pyspark.sql import SparkSession
except ImportError:
    print("WARNING: PySpark not available. Will use local processing only.")
    SparkSession = None

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.optimizers import Adam
except ImportError:
    print("WARNING: TensorFlow not available. Model training will fail.")
    keras = None

try:
    import fiftyone.zoo as foz
    FIFTYONE_AVAILABLE = True
except ImportError:
    FIFTYONE_AVAILABLE = False
    print("INFO: FiftyOne not available. Will use local images only.")


class ImagePreprocessor:
    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        self.image_size = image_size
    
    def load_image(self, image_path: str):
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return cv2.resize(image, self.image_size)
        except Exception as e:
            logger.error(f"Error loading {image_path}: {e}")
            return None
    
    def normalize_image(self, image: np.ndarray):
        return image.astype(np.float32) / 255.0
    
    def segment_image(self, image: np.ndarray):
        try:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            objects = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:
                    x, y, w, h = cv2.boundingRect(contour)
                    objects.append({'bbox': [x, y, w, h], 'area': float(area)})
            
            markers = cv2.watershed(image.astype(np.uint8), np.zeros(gray.shape, dtype=np.int32))
            return {'contours': len(contours), 'objects': objects, 'markers': markers}
        except Exception as e:
            logger.error(f"Segmentation error: {e}")
            return None
    
    def extract_features(self, image: np.ndarray):
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        features = {
            'mean': float(gray.mean()),
            'std': float(gray.std()),
            'min': float(gray.min()),
            'max': float(gray.max())
        }
        edges = cv2.Canny(gray, 100, 200)
        features['edge_density'] = float(edges.sum() / (edges.shape[0] * edges.shape[1]))
        return features


class DatasetLoader:
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        self.label_map = {}
    
    def load_from_fiftyone(self, dataset_name: str = "coco-2017", max_samples: Dict[str, int] = None):
        if not FIFTYONE_AVAILABLE:
            logger.error("FiftyOne not available")
            return {}, {}
        
        if max_samples is None:
            max_samples = {'train': 1000, 'validation': 300, 'test': 200}
        
        dataset_splits = {}
        labels_splits = {}
        for split in ['train', 'validation', 'test']:
            logger.info(f"Loading {split}...")
            try:
                dataset = foz.load_zoo_dataset(dataset_name, split=split, max_samples=max_samples.get(split, 100))
                dataset_splits[split], labels_splits[split] = self._save_dataset_locally(dataset, split, max_samples.get(split, 100))
            except Exception as e:
                logger.error(f"Error loading {split}: {e}")
                dataset_splits[split] = []
                labels_splits[split] = []
        
        return dataset_splits, labels_splits
    
    def _save_dataset_locally(self, dataset, split_name: str, max_samples: int):
        file_paths = []
        labels = []
        try:
            for idx, sample in enumerate(dataset):
                if idx >= max_samples:
                    break
                image_path = sample.filepath
                output_path = os.path.join(self.data_dir, split_name, f"{idx:06d}.jpg")
                Path(self.data_dir, split_name).mkdir(parents=True, exist_ok=True)
                
                if os.path.exists(image_path):
                    img = cv2.imread(image_path)
                    if img is not None:
                        cv2.imwrite(output_path, img)
                        file_paths.append(output_path)
                        
                        label = 0
                        if hasattr(sample, 'ground_truth') and sample.ground_truth:
                            detections = sample.ground_truth.detections
                            if detections:
                                label = min(len(detections), 79)
                        elif hasattr(sample, 'label') and sample.label:
                            label = hash(str(sample.label)) % 80
                        
                        labels.append(label)
            
            logger.info(f"Saved {len(file_paths)} images to {split_name} with labels")
            return file_paths, labels
        except Exception as e:
            logger.error(f"Error saving {split_name}: {e}")
            return file_paths, labels
    
    def load_from_local(self, split: str = "train"):
        split_dir = os.path.join(self.data_dir, split)
        if not os.path.exists(split_dir):
            return [], []
        
        image_files = list(Path(split_dir).glob("*.jpg")) + list(Path(split_dir).glob("*.png"))
        image_files = sorted([str(f) for f in image_files])
        logger.info(f"Found {len(image_files)} images in {split}")
        
        labels = list(range(len(image_files)))
        return image_files, labels


class CNNImageClassifier:
    def __init__(self, num_classes: int = 80, image_size: Tuple[int, int] = (224, 224)):
        self.num_classes = num_classes
        self.image_size = image_size
        self.model = None
    
    def build_model(self, model_type: str = "mobilenetv2"):
        if model_type == "mobilenetv2":
            logger.info("Building MobileNetV2 model...")
            base_model = MobileNetV2(input_shape=(*self.image_size, 3), include_top=False, weights='imagenet')
            base_model.trainable = False
            
            self.model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(self.num_classes, activation='softmax')
            ])
        else:
            logger.info("Building custom CNN model...")
            self.model = models.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.image_size, 3)),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.Flatten(),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes, activation='softmax')
            ])
        
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        return self.model
    
    def preprocess_batch(self, image_paths: List[str], preprocessor):
        images = []
        for image_path in image_paths:
            image = preprocessor.load_image(image_path)
            if image is not None:
                images.append(preprocessor.normalize_image(image))
        return np.array(images) if images else np.array([])
    
    def train(self, train_images: np.ndarray, train_labels: np.ndarray, val_images: np.ndarray, val_labels: np.ndarray, epochs: int = 20, batch_size: int = 32):
        logger.info(f"Training for {epochs} epochs with batch_size={batch_size}...")
        history = self.model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=epochs, batch_size=batch_size, verbose=1)
        
        logger.info("\n" + "=" * 60)
        logger.info("Training Completed - Final Metrics:")
        logger.info("=" * 60)
        logger.info(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
        logger.info(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
        logger.info(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
        logger.info(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
        
        best_val_acc = max(history.history['val_accuracy'])
        best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
        logger.info(f"Best Validation Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
        logger.info("=" * 60 + "\n")
        
        return history.history
    
    def predict(self, images: np.ndarray):
        return self.model.predict(images)
    
    def save_model(self, filepath: str):
        if self.model:
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")


class SparkImageProcessor:
    def __init__(self, app_name: str = "CNN-Image-Classification"):
        try:
            self.spark = SparkSession.builder.appName(app_name).config("spark.driver.maxResultSize", "2g").getOrCreate()
            self.spark.sparkContext.setLogLevel("WARN")
            logger.info("Spark session initialized")
            self.available = True
        except Exception as e:
            logger.warning(f"Spark initialization failed: {e}. Using local processing.")
            self.spark = None
            self.available = False
    
    def process_images_distributed(self, image_paths: List[str], preprocessor):
        if not self.available or len(image_paths) == 0:
            return []
        
        logger.info(f"Processing {len(image_paths)} images...")
        
        try:
            rdd = self.spark.sparkContext.parallelize(image_paths, numPartitions=4)
            
            def process_image(image_path):
                image = preprocessor.load_image(image_path)
                if image is None:
                    return None
                normalized = preprocessor.normalize_image(image)
                seg = preprocessor.segment_image(normalized)
                features = preprocessor.extract_features(normalized)
                return {'path': image_path, 'image': normalized.tolist(), 'segmentation': seg, 'features': features}
            
            results = rdd.map(process_image).filter(lambda x: x is not None).collect()
            logger.info(f"Processed {len(results)} images")
            return results
        except Exception as e:
            logger.error(f"Distributed processing failed: {e}")
            return self._process_local(image_paths, preprocessor)
    
    def _process_local(self, image_paths: List[str], preprocessor):
        results = []
        for image_path in image_paths:
            image = preprocessor.load_image(image_path)
            if image is not None:
                normalized = preprocessor.normalize_image(image)
                seg = preprocessor.segment_image(normalized)
                features = preprocessor.extract_features(normalized)
                results.append({'path': image_path, 'image': normalized.tolist(), 'segmentation': seg, 'features': features})
        return results
    
    def stop(self):
        if self.spark:
            self.spark.stop()
            logger.info("Spark session stopped")


def main():
    logger.info("=" * 60)
    logger.info("CNN Image Classification with Apache Spark")
    logger.info("=" * 60)
    
    config = {
        'data_dir': './data',
        'model_dir': './models',
        'image_size': (224, 224),
        'num_classes': 80,
        'batch_size': 32,
        'epochs': 10,
        'max_samples': {'train': 1000, 'validation': 300, 'test': 200}
    }
    
    Path(config['model_dir']).mkdir(parents=True, exist_ok=True)
    
    preprocessor = ImagePreprocessor(image_size=config['image_size'])
    dataset_loader = DatasetLoader(data_dir=config['data_dir'])
    classifier = CNNImageClassifier(num_classes=config['num_classes'], image_size=config['image_size'])
    spark_processor = SparkImageProcessor()
    
    try:
        logger.info("\nStep 1: Loading datasets...")
        dataset_splits, labels_splits = dataset_loader.load_from_fiftyone(dataset_name="coco-2017", max_samples=config['max_samples'])
        
        if not dataset_splits or all(len(v) == 0 for v in dataset_splits.values()):
            logger.info("FiftyOne loading failed. Using local loading...")
            dataset_splits, labels_splits = {}, {}
            for split in ['train', 'validation', 'test']:
                imgs, labels = dataset_loader.load_from_local(split)
                dataset_splits[split] = imgs
                labels_splits[split] = labels
        
        logger.info("\nStep 2: Preprocessing images...")
        train_data = spark_processor.process_images_distributed(dataset_splits['train'][:min(100, len(dataset_splits['train']))], preprocessor)
        val_data = spark_processor.process_images_distributed(dataset_splits['validation'][:min(30, len(dataset_splits['validation']))], preprocessor)
        test_data = spark_processor.process_images_distributed(dataset_splits['test'][:min(20, len(dataset_splits['test']))], preprocessor)
        
        logger.info(f"Processed - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        if len(train_data) > 0 and len(val_data) > 0:
            train_images = np.array([np.array(item['image']) for item in train_data])
            train_labels_real = labels_splits['train'][:len(train_data)]
            train_labels = keras.utils.to_categorical(train_labels_real, num_classes=config['num_classes'])
            
            val_images = np.array([np.array(item['image']) for item in val_data])
            val_labels_real = labels_splits['validation'][:len(val_data)]
            val_labels = keras.utils.to_categorical(val_labels_real, num_classes=config['num_classes'])
            
            test_images = np.array([np.array(item['image']) for item in test_data]) if test_data else np.array([])
            test_labels_real = labels_splits['test'][:len(test_data)]
            
            logger.info(f"\nDataset Summary:")
            logger.info(f"  Train samples: {len(train_images)} with {len(set(train_labels_real))} unique labels")
            logger.info(f"  Val samples: {len(val_images)} with {len(set(val_labels_real))} unique labels")
            logger.info(f"  Test samples: {len(test_images)}")
            
            logger.info("\nStep 3: Building and training CNN model...")
            classifier.build_model(model_type="mobilenetv2")
            classifier.train(train_images, train_labels, val_images, val_labels, epochs=config['epochs'], batch_size=config['batch_size'])
            
            if len(test_images) > 0:
                logger.info("\nStep 4: Making predictions on test set...")
                predictions = classifier.predict(test_images)
                logger.info(f"Predictions shape: {predictions.shape}")
                
                pred_classes = np.argmax(predictions, axis=1)
                correct = np.sum(pred_classes == test_labels_real)
                test_accuracy = correct / len(test_labels_real)
                
                logger.info(f"\nTest Set Metrics:")
                logger.info(f"  Test Accuracy: {test_accuracy:.4f} ({correct}/{len(test_labels_real)} correct)")
                logger.info(f"\nSample Predictions:")
                for i in range(min(5, len(predictions))):
                    pred_class = pred_classes[i]
                    true_class = test_labels_real[i]
                    confidence = predictions[i][pred_class]
                    match = "✓" if pred_class == true_class else "✗"
                    logger.info(f"  {match} Image {i}: Predicted={pred_class}, True={true_class}, Confidence={confidence:.4f}")
            
            model_path = os.path.join(config['model_dir'], 'cnn_classifier.h5')
            classifier.save_model(model_path)
            logger.info(f"\nCompleted! Model saved to: {model_path}")
        else:
            logger.error("No training data available")
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    
    finally:
        spark_processor.stop()


if __name__ == "__main__":
    main()

