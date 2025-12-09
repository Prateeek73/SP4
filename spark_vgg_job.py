#!/usr/bin/env python3
"""
Apache Spark Job: VGG-Style CNN Training
Distributed data loading via Spark RDD, GPU training on spark-worker-gpu
Submit with: spark-submit --master spark://spark-master:7077 spark_vgg_job.py
"""
import os
import sys
import json
import logging
import numpy as np
from collections import Counter
from pyspark.sql import SparkSession
from pyspark import SparkContext

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_spark():
    """Initialize Spark session for distributed processing"""
    logger.info("="*70)
    logger.info("Initializing Apache Spark Session")
    logger.info("="*70)
    
    spark = SparkSession.builder \
        .appName("VGG_CNN_Training") \
        .config("spark.executor.memory", "1400m") \
        .config("spark.executor.cores", "1") \
        .config("spark.cores.max", "3") \
        .config("spark.driver.memory", "2g") \
        .config("spark.driver.maxResultSize", "1g") \
        .config("spark.default.parallelism", "3") \
        .config("spark.executor.memoryOverhead", "512m") \
        .config("spark.memory.fraction", "0.6") \
        .config("spark.memory.storageFraction", "0.3") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
        .config("spark.kryoserializer.buffer.max", "256m") \
        .config("spark.rpc.message.maxSize", "256") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    logger.info(f"Spark Master: {spark.sparkContext.master}")
    logger.info(f"Spark App ID: {spark.sparkContext.applicationId}")
    logger.info(f"Available Executors: 3 workers")
    logger.info("="*70)
    
    return spark

def load_data_distributed_batched(spark, num_batches=4, chunks_per_batch=10, num_classes=5):
    """Load data in batches to avoid OutOfMemoryError"""
    logger.info("\n" + "="*70)
    logger.info("STEP 1: Batch-wise Distributed Data Loading via Spark RDD")
    logger.info(f"Loading {num_batches} batches × {chunks_per_batch} chunks = {num_batches * chunks_per_batch} total chunks")
    logger.info("="*70)
    
    def load_chunk_task(task_info):
        """Worker function to load a single chunk"""
        import numpy as np
        split, chunk_id = task_info
        data_dir = "/opt/spark/data/processed"
        
        img_file = f"{data_dir}/{split}/images_chunk_{chunk_id}.npy"
        lbl_file = f"{data_dir}/{split}/labels_chunk_{chunk_id}.npy"
        
        if not os.path.exists(img_file):
            return None
        
        imgs = np.load(img_file)
        labs = np.load(lbl_file, allow_pickle=True)
        
        return (split, imgs, labs, len(imgs), chunk_id)
    
    # Target classes for filtering
    target_classes = ['person', 'cat', 'dog', 'car', 'chair']
    
    all_train_labels = []
    total_images = 0
    
    # Load data in batches to avoid OOM
    for batch_num in range(num_batches):
        start_chunk = batch_num * chunks_per_batch
        end_chunk = start_chunk + chunks_per_batch
        
        logger.info(f"\n--- Batch {batch_num + 1}/{num_batches}: Loading chunks {start_chunk}-{end_chunk-1} ---")
        
        tasks = [('train', i) for i in range(start_chunk, end_chunk)]
        
        # Parallelize loading across workers
        rdd = spark.sparkContext.parallelize(tasks, numSlices=3)
        results = rdd.map(load_chunk_task).filter(lambda x: x is not None).collect()
        
        # Process batch results
        batch_images = []
        batch_labels = []
        
        for split, imgs, labs, count, chunk_id in results:
            batch_images.append(imgs)
            batch_labels.extend(labs)
        
        if batch_images:
            batch_images = np.concatenate(batch_images, axis=0)
            batch_labels = np.array(batch_labels)
            
            # Filter to target classes for this batch
            mask = np.isin(batch_labels, target_classes)
            filtered_images = batch_images[mask]
            filtered_labels = batch_labels[mask]
            
            # Save batch immediately to avoid memory buildup
            batch_dir = '/opt/spark/data/batches'
            os.makedirs(batch_dir, exist_ok=True)
            np.save(f'{batch_dir}/batch_{batch_num}.npy', filtered_images)
            
            all_train_labels.extend(filtered_labels)
            total_images += len(filtered_images)
            
            logger.info(f"  Batch {batch_num + 1}: Loaded {len(batch_images)} → Filtered to {len(filtered_images)} images → Saved")
            
            # Clear batch memory immediately
            del batch_images, batch_labels, filtered_images, filtered_labels
    
    # Save labels
    train_labels = np.array(all_train_labels)
    np.save(f'{batch_dir}/labels.npy', train_labels)
    
    logger.info(f"\n✓ Total data prepared: {total_images} images from {num_batches * chunks_per_batch} chunks")
    logger.info(f"✓ All batches saved to {batch_dir}")
    logger.info(f"Classes in dataset: {target_classes}")
    
    return batch_dir, total_images, train_labels, target_classes

def load_data_distributed(spark, num_classes=5, max_train=100):
    """Load and filter data using Spark distributed processing"""
    logger.info("\n" + "="*70)
    logger.info("STEP 1: Distributed Data Loading via Spark RDD")
    logger.info("="*70)
    
    def load_chunk_task(task_info):
        """Worker function to load a single chunk"""
        import numpy as np
        split, chunk_id = task_info
        data_dir = "/opt/spark/data/processed"
        
        img_file = f"{data_dir}/{split}/images_chunk_{chunk_id}.npy"
        lbl_file = f"{data_dir}/{split}/labels_chunk_{chunk_id}.npy"
        
        if not os.path.exists(img_file):
            return None
        
        imgs = np.load(img_file)
        labs = np.load(lbl_file, allow_pickle=True)
        
        return (split, imgs, labs, len(imgs), chunk_id)
    
    # Create tasks for parallel loading (training only)
    # Safe limit: 10 chunks to avoid OutOfMemoryError during collect()
    # Each chunk = 50 images × 224×224×3×4 bytes ≈ 30MB
    # 10 chunks ≈ 300MB serialized data (within driver maxResultSize limit)
    tasks = [('train', i) for i in range(10)]
    
    logger.info(f"Distributing {len(tasks)} data loading tasks across Spark cluster...")
    
    # Parallelize loading across workers using Spark RDD
    rdd = spark.sparkContext.parallelize(tasks, numSlices=3)
    results = rdd.map(load_chunk_task).filter(lambda x: x is not None).collect()
    
    # Aggregate results
    train_images = []
    train_labels = []
    
    for split, imgs, labs, count, chunk_id in results:
        logger.info(f"  Worker loaded {split} chunk {chunk_id}: {count} images")
        train_images.append(imgs)
        train_labels.extend(labs)
    
    train_images = np.concatenate(train_images, axis=0)
    train_labels = np.array(train_labels)
    
    logger.info(f"\nTotal data loaded: {train_images.shape}")
    
    # Use specific 5 classes: person, cat, dog, car, chair
    target_classes = ['person', 'cat', 'dog', 'car', 'chair']
    logger.info(f"\nFiltering to {len(target_classes)} specific classes: {', '.join(target_classes)}")
    
    # Filter to target classes
    label_counts = Counter(train_labels)
    available_classes = [cls for cls in target_classes if label_counts[cls] > 0]
    
    logger.info(f"\nClass distribution:")
    for cls in available_classes:
        logger.info(f"  {cls}: {label_counts[cls]} images")
    
    train_mask = np.isin(train_labels, available_classes)
    train_images = train_images[train_mask][:max_train]
    train_labels = train_labels[train_mask][:max_train]
    
    logger.info(f"\nFiltered dataset: {train_images.shape}")
    logger.info("="*70)
    
    return train_images, train_labels, available_classes

def prepare_training_data(train_images, train_labels, target_classes, config):
    """Prepare and save data for GPU training"""
    logger.info("\n" + "="*70)
    logger.info("STEP 2: Preparing Training Data")
    logger.info("="*70)
    
    # Create label mapping
    label_to_int = {label: idx for idx, label in enumerate(target_classes)}
    int_to_label = {idx: label for label, idx in label_to_int.items()}
    
    logger.info(f"Label mapping for {len(label_to_int)} classes:")
    for label, idx in label_to_int.items():
        logger.info(f"  {idx}: {label}")
    
    # Encode labels
    train_labels_int = np.array([label_to_int[label] for label in train_labels])
    
    # Convert to categorical
    num_classes = len(label_to_int)
    train_labels_cat = np.eye(num_classes)[train_labels_int]
    
    # Save data
    logger.info("\nSaving training data to shared storage...")
    os.makedirs('/opt/spark/data/training', exist_ok=True)
    
    np.save('/opt/spark/data/training/train_images.npy', train_images)
    np.save('/opt/spark/data/training/train_labels.npy', train_labels_cat)
    
    with open('/opt/spark/data/training/config.json', 'w') as f:
        json.dump(config, f)
    
    with open('/opt/spark/data/training/label_mapping.json', 'w') as f:
        json.dump({k: int(v) for k, v in label_to_int.items()}, f)

def prepare_training_data_from_batches(batch_dir, total_images, train_labels, target_classes, config):
    """Prepare training data from saved batches without loading all into memory"""
    logger.info("\n" + "="*70)
    logger.info("STEP 2: Preparing Training Data from Batches")
    logger.info("="*70)
    
    num_classes = config['num_classes']
    
    # Create label mappings
    label_to_int = {label: idx for idx, label in enumerate(target_classes)}
    int_to_label = {idx: label for label, idx in label_to_int.items()}
    
    logger.info(f"Label mapping for {len(label_to_int)} classes:")
    for label, idx in label_to_int.items():
        logger.info(f"  {idx}: {label}")
    
    # Convert labels to categorical
    train_labels_int = np.array([label_to_int[label] for label in train_labels])
    train_labels_cat = np.eye(num_classes)[train_labels_int]
    
    logger.info(f"Total training images: {total_images}")
    logger.info(f"Training labels shape: {train_labels_cat.shape}")
    
    # Save batch directory path and labels for GPU worker
    training_dir = '/opt/spark/data/training'
    os.makedirs(training_dir, exist_ok=True)
    
    # Save metadata about batches
    batch_files = sorted([f for f in os.listdir(batch_dir) if f.startswith('batch_')])
    with open(f'{training_dir}/batch_info.json', 'w') as f:
        json.dump({
            'batch_dir': batch_dir,
            'total_images': total_images,
            'num_batches': len(batch_files)
        }, f)
    
    np.save(f'{training_dir}/train_labels.npy', train_labels_cat)
    
    with open(f'{training_dir}/config.json', 'w') as f:
        json.dump(config, f)
    
    with open(f'{training_dir}/label_mapping.json', 'w') as f:
        json.dump({k: int(v) for k, v in label_to_int.items()}, f)
    
    logger.info(f"✓ Batch metadata saved to {training_dir}")
    logger.info(f"✓ {len(batch_files)} batch files available in {batch_dir}")
    logger.info("="*70)
    
    return int_to_label
    
    logger.info("Data saved to /opt/spark/data/training/")
    logger.info("="*70)
    
    return int_to_label

def submit_training_to_gpu_worker(spark):
    """Submit training job to GPU worker via Spark with batch loading"""
    logger.info("\n" + "="*70)
    logger.info("STEP 3: Training on GPU Worker (TF.Data Streaming)")
    logger.info("="*70)
    
    # Execute on GPU worker - directly use the batched script
    def run_training():
        import subprocess
        import sys
        
        # Run the batched training script directly
        result = subprocess.run(
            [sys.executable, '/opt/spark/train_gpu_task_batched.py'],
            capture_output=True,
            text=True,
            cwd='/opt/spark'
        )
        
        return result.stdout + result.stderr
    
    logger.info("Submitting TF.Data streaming training to GPU worker...")
    rdd = spark.sparkContext.parallelize([1], numSlices=1)
    result = rdd.map(lambda x: run_training()).collect()
    
    logger.info("\n" + "="*70)
    logger.info("GPU Worker Training Output:")
    logger.info("="*70)
    for output in result:
        print(output)  # Print directly to show training progress
    logger.info("="*70)


def main():
    """Main Spark job execution"""
    import tensorflow as tf
    
    # Configure GPU at the very beginning (for driver if needed)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    
    logger.info("="*70)
    logger.info("Apache Spark Job: VGG-Style CNN Training")
    logger.info("="*70)
    logger.info("\nJob Configuration:")
    logger.info("  - Framework: Apache Spark + TensorFlow")
    logger.info("  - Architecture: VGG-style CNN (3 conv blocks)")
    logger.info("  - Dataset: COCO subset, 5 classes")
    logger.info("  - GPU: NVIDIA GTX 1050 (CUDA 11.8)")
    logger.info("  - Distribution: Spark RDD-based parallel processing")
    logger.info("="*70)
    
    try:
        # Initialize Spark
        spark = initialize_spark()
        
        # Training configuration
        config = {
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 0.001,
            'num_classes': 5,
            'num_batches': 2,        # Load data in 2 batches for 1000 images
            'chunks_per_batch': 10   # 10 chunks per batch = 20 chunks total (~1000 images)
        }
        
        # Step 1: Load data in batches across Spark workers (40 chunks = ~2000 images)
        batch_dir, total_images, train_labels, available_classes = load_data_distributed_batched(
            spark, 
            num_batches=config['num_batches'],
            chunks_per_batch=config['chunks_per_batch'],
            num_classes=config['num_classes']
        )
        
        # Step 2: Prepare training data from saved batches
        int_to_label = prepare_training_data_from_batches(
            batch_dir, total_images, train_labels, available_classes, config
        )
        
        # Step 3: Train on GPU worker
        submit_training_to_gpu_worker(spark)
        
        logger.info("\n" + "="*70)
        logger.info("Spark Job Completed Successfully!")
        logger.info("Model saved: /opt/spark/data/models/vgg_final.keras")
        logger.info("="*70)
        
        # Stop Spark session
        spark.stop()
        return 0
        
    except Exception as e:
        logger.error("\n" + "="*70)
        logger.error(f"Spark Job Error: {e}")
        logger.error("="*70)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
