import os, json, logging, sys
from pathlib import Path
import numpy as np, cv2
from pyspark.sql import SparkSession

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LOCAL_RAW_PATH = "/opt/spark/data/raw"
LOCAL_PROCESSED_PATH = "/opt/spark/data/processed"

class HDFSImagePreprocessor:
    def __init__(self):
        try:
            import socket
            hostname = socket.gethostname()
            master_url = "spark://spark-master:7077"
            logger.info(f"Environment: hostname={hostname}, using master={master_url}")
            self.spark = (
                SparkSession.builder
                .appName("ImagePreprocessing")
                .master(master_url)
                .config("spark.driver.memory", "1500m")
                .config("spark.driver.cores", "2")
                .config("spark.executor.memory", "900m")
                .config("spark.executor.cores", "2")
                .config("spark.cores.max", "4")
                .config("spark.driver.maxResultSize", "400m")
                .config("spark.task.maxFailures", "2")
                .config("spark.shuffle.service.enabled", "false")
                .config("spark.python.worker.memory", "300m")
                .config("spark.python.worker.reuse", "false")
                .config("spark.scheduler.mode", "FIFO")
                .config("spark.default.parallelism", "4")
                .config("spark.rpc.message.maxSize", "128")
                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                .config("spark.kryoserializer.buffer.max", "128m")
                .config("spark.sql.execution.arrow.pyspark.enabled", "false")
                .getOrCreate()
            )
            self.spark.sparkContext.setLogLevel("WARN")
            logger.info("="*70)
            logger.info("Spark Preprocessing Job - CPU Workers (Cluster Mode)")
            logger.info("-"*70)
            logger.info("Configuration:")
            logger.info("  - Worker 1: 172.18.0.3 (2 cores, 1024 MiB)")
            logger.info("  - Worker 2: 172.18.0.4 (2 cores, 1024 MiB)")
            logger.info("  - Executor Memory: 900m per worker")
            logger.info("  - Total Cores: 4 (parallelism: 4)")
            logger.info("  - Chunk size: 50 images")
            logger.info("  - Preprocessing: ImageNet normalization (Mean=[0.485,0.456,0.406], Std=[0.229,0.224,0.225])")
            logger.info("  - Output: Local " + LOCAL_PROCESSED_PATH)
            logger.info("="*70)
            self.avail = True
            self.sc = self.spark.sparkContext
        except Exception as e:
            logger.error(f"Spark initialization failed: {e}")
            self.avail = False
            self.sc = None
    
    def load_local_images(self, data_dir, split, max_count, target_classes=None):
        """Load image paths from raw data directory prepared by data_loader.py"""
        # Use LOCAL_RAW_PATH if data_dir is default, otherwise use provided path
        if data_dir == './data':
            sp_dir = os.path.join(LOCAL_RAW_PATH, split)
            logger.info(f"Loading from raw data storage: {sp_dir}")
        else:
            sp_dir = os.path.join(data_dir, split)
        
        if not os.path.exists(sp_dir):
            logger.warning(f"Directory not found: {sp_dir}")
            return [], []
        files = sorted(list(Path(sp_dir).glob("coco_*.jpg")) + list(Path(sp_dir).glob("coco_*.png")))
        
        labels = []
        try:
            if os.path.exists(lf := os.path.join(sp_dir, 'labels.json')):
                with open(lf, 'r') as f:
                    ld = json.load(f)
                    labels = ld.get('labels', []) if isinstance(ld, dict) else ld
        except Exception as e:
            logger.warning(f"Failed to load labels: {e}")
        if not labels:
            labels = [i % 80 for i in range(len(files))]
        
        # Filter by target classes if specified
        if target_classes:
            logger.info(f"Filtering to target classes: {target_classes}")
            logger.info(f"Sample labels before filtering: {labels[:20]}")
            filtered_paths = []
            filtered_labels = []
            for i, (file, label) in enumerate(zip(files, labels)):
                if label in target_classes:
                    filtered_paths.append(str(file))
                    filtered_labels.append(label)
                    if len(filtered_paths) >= max_count:
                        break
            paths = filtered_paths
            labels = filtered_labels
            logger.info(f"After filtering: {len(paths)} images from target classes")
            if len(paths) > 0:
                logger.info(f"Sample filtered labels: {labels[:10]}")
        else:
            paths = [str(f) for f in files[:max_count]]
            labels = labels[:len(paths)]
        
        logger.info(f"Loaded {split}: {len(paths)} images")
        return paths, labels
    
    def process_and_save_to_hdfs(self, paths, labels, split_name, target_size=(224, 224)):
        """Process images using Spark workers by broadcasting minimal data"""
        if not self.avail or not paths:
            logger.error("Cannot process: Spark unavailable or no paths")
            return False
        
        logger.info(f"\nProcessing {split_name}: {len(paths)} images")
        logger.info(f"Target size: {target_size}, Output: {LOCAL_PROCESSED_PATH}/{split_name}/")
        logger.info(f"Mode: DISTRIBUTED processing across spark-worker-1 and spark-worker-2")
        
        # Process in chunks to save memory
        chunk_size = 50
        num_chunks = (len(paths) + chunk_size - 1) // chunk_size
        logger.info(f"Processing in {num_chunks} chunks of {chunk_size} images")
        
        processed_count = 0
        
        for chunk_id in range(num_chunks):
            start_idx = chunk_id * chunk_size
            end_idx = min(start_idx + chunk_size, len(paths))
            chunk_paths = paths[start_idx:end_idx]
            chunk_labels = labels[start_idx:end_idx]
            
            if (chunk_id + 1) % 10 == 0 or chunk_id == 0:
                logger.info(f"Progress: Processing chunk {chunk_id+1}/{num_chunks}")
            
            # Create simple index list and broadcast paths/labels
            indices = list(range(len(chunk_paths)))
            bc_paths = self.sc.broadcast(chunk_paths)
            bc_labels = self.sc.broadcast(chunk_labels)
            bc_size = self.sc.broadcast(target_size)
            
            def process_by_index(idx):
                """Process image by index on worker"""
                import cv2 as cv2_local
                import numpy as np_local
                import os as os_local
                try:
                    path = bc_paths.value[idx]
                    label = bc_labels.value[idx]
                    size = bc_size.value
                    
                    if not os_local.path.exists(path):
                        return None
                    
                    img = cv2_local.imread(path)
                    if img is None:
                        return None
                    
                    img_rgb = cv2_local.cvtColor(img, cv2_local.COLOR_BGR2RGB)
                    resized = cv2_local.resize(img_rgb, size)
                    
                    # ImageNet normalization
                    normalized = resized.astype(np_local.float32) / 255.0
                    mean = np_local.array([0.485, 0.456, 0.406], dtype=np_local.float32)
                    std = np_local.array([0.229, 0.224, 0.225], dtype=np_local.float32)
                    normalized = (normalized - mean) / std
                    
                    return (normalized.tobytes(), label)
                except Exception:
                    return None
            
            # Create RDD with 4 partitions (utilize all cores)
            rdd = self.sc.parallelize(indices, 4)
            results = rdd.map(process_by_index).filter(lambda x: x is not None).collect()
            
            # Cleanup broadcasts
            bc_paths.unpersist()
            bc_labels.unpersist()
            bc_size.unpersist()
            
            if not results:
                logger.warning(f"Chunk {chunk_id+1} produced no results")
                continue
            
            # Reconstruct images
            processed_images = []
            processed_labels = []
            for img_bytes, label in results:
                try:
                    img_array = np.frombuffer(img_bytes, dtype=np.float32).reshape(*target_size, 3)
                    processed_images.append(img_array)
                    processed_labels.append(label)
                except Exception as e:
                    logger.warning(f"Failed to deserialize: {str(e)[:50]}")
            
            if not processed_images:
                continue
            
            # Save chunk to local storage
            chunk_array = np.array(processed_images, dtype=np.float32)
            chunk_labels_array = np.array(processed_labels, dtype=object)
            
            output_dir = f"{LOCAL_PROCESSED_PATH}/{split_name}"
            os.makedirs(output_dir, exist_ok=True)
            
            output_images = f"{output_dir}/images_chunk_{chunk_id}.npy"
            output_labels = f"{output_dir}/labels_chunk_{chunk_id}.npy"
            
            np.save(output_images, chunk_array)
            np.save(output_labels, chunk_labels_array, allow_pickle=True)
            
            processed_count += len(processed_images)
            
            if (chunk_id + 1) % 10 == 0 or chunk_id == num_chunks - 1:
                logger.info(f"Chunk {chunk_id+1}/{num_chunks}: Saved {len(processed_images)} images (Total: {processed_count}/{len(paths)})")
        
        logger.info(f"Completed {split_name}: {processed_count}/{len(paths)} images saved to {LOCAL_PROCESSED_PATH}/{split_name}/")
        return processed_count > 0
    
    def stop(self):
        if self.avail:
            try:
                logger.info("Stopping Spark session...")
                self.spark.sparkContext.stop()
                import time
                time.sleep(2)
                self.spark.stop()
                logger.info("Spark stopped successfully")
            except Exception as e:
                logger.warning(f"Error stopping Spark: {e}")

def main():
    logger.info("="*70)
    logger.info("IMAGE PREPROCESSING JOB - Distributed CPU Processing")
    logger.info("="*70)
    
    # Configuration - Filter to 5 specific classes
    TARGET_CLASSES = ['person', 'cat', 'dog', 'car', 'chair']
    
    config = {
        'data_dir': './data',
        'target_size': (224, 224),
        'target_classes': TARGET_CLASSES,
        'max_samples': {
            'train': 64000,
            'validation': 8000,
            'test': 8000
        }
    }
    
    logger.info(f"Configuration:")
    logger.info(f"  - Data directory: {config['data_dir']}")
    logger.info(f"  - Target size: {config['target_size']}")
    logger.info(f"  - Target classes: {TARGET_CLASSES} (FILTERING ENABLED)")
    logger.info(f"  - Preprocessing: ImageNet normalization (for ResNet/EfficientNet/MobileNetV3)")
    logger.info(f"  - Max samples: train={config['max_samples']['train']}, val={config['max_samples']['validation']}, test={config['max_samples']['test']}")
    logger.info("="*70)
    
    preprocessor = HDFSImagePreprocessor()
    
    if not preprocessor.avail:
        logger.error("Preprocessing failed: Spark unavailable")
        sys.exit(1)
    
    try:
        # Process each split
        for split in ['train', 'validation', 'test']:
            logger.info(f"\n{'='*70}")
            logger.info(f"Processing {split.upper()} split")
            logger.info(f"{'='*70}")
            
            paths, labels = preprocessor.load_local_images(
                config['data_dir'], 
                split, 
                config['max_samples'][split],
                target_classes=config['target_classes']
            )
            
            if not paths:
                logger.warning(f"No images found for {split} split, skipping...")
                continue
            
            success = preprocessor.process_and_save_to_hdfs(
                paths, 
                labels, 
                split, 
                config['target_size']
            )
            
            if not success:
                logger.error(f"Failed to process {split} split")
            else:
                logger.info(f"Successfully processed {split} split")
        
        logger.info("\n" + "="*70)
        logger.info("PREPROCESSING COMPLETE")
        logger.info("="*70)
        logger.info(f"Processed data available at: {LOCAL_PROCESSED_PATH}/")
        logger.info("Run training.py on GPU worker to train models")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"Preprocessing error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        preprocessor.stop()

if __name__ == "__main__":
    main()