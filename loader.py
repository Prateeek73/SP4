import os
import json
import logging
import sys
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import fiftyone as fo
    import fiftyone.zoo as foz
except ImportError:
    logger.error("FiftyOne not installed. Run: pip install fiftyone")
    sys.exit(1)

OUTPUT_DIR = "/opt/spark/data/raw"

def download_and_organize_split(split, max_samples):
    """Download COCO images and organize them with labels"""
    logger.info(f"{'='*70}")
    logger.info(f"Processing {split.upper()} split ({max_samples} samples)")
    logger.info(f"{'='*70}")
    
    output_split_dir = os.path.join(OUTPUT_DIR, split)
    os.makedirs(output_split_dir, exist_ok=True)
    
    # Download with FiftyOne
    logger.info(f"Downloading COCO-2017 {split} split...")
    try:
        dataset = foz.load_zoo_dataset(
            "coco-2017",
            split=split,
            max_samples=max_samples
        )
        logger.info(f"Downloaded {len(dataset)} samples")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False
    
    # Extract images and labels
    logger.info("Copying images and extracting labels...")
    labels_list = []
    copied_count = 0
    
    for idx, sample in enumerate(dataset):
        if idx >= max_samples:
            break
            
        # Get source image path
        src_path = sample.filepath
        if not os.path.exists(src_path):
            continue
        
        # Copy image to output directory
        dst_filename = f"coco_{idx:06d}.jpg"
        dst_path = os.path.join(output_split_dir, dst_filename)
        shutil.copy2(src_path, dst_path)
        
        # Extract label
        label = "unknown"
        if hasattr(sample, 'ground_truth') and sample.ground_truth:
            detections = getattr(sample.ground_truth, 'detections', [])
            if detections and len(detections) > 0:
                label = str(detections[0].label).lower().strip()
        
        labels_list.append(label)
        copied_count += 1
        
        if (idx + 1) % 500 == 0:
            logger.info(f"Progress: {idx + 1}/{max_samples} images")
    
    # Save labels.json
    labels_path = os.path.join(output_split_dir, "labels.json")
    with open(labels_path, 'w') as f:
        json.dump({"labels": labels_list}, f, indent=2)
    
    logger.info(f"Completed: {copied_count} images saved to {output_split_dir}")
    logger.info(f"Labels saved to {labels_path}")
    
    # Cleanup FiftyOne dataset
    dataset.delete()
    
    return True

def main():
    logger.info("="*70)
    logger.info("COCO-2017 DATA LOADER")
    logger.info("="*70)
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("="*70)
    
    # Configuration
    splits_config = {
        'train': 64000,
        'validation': 8000,
        'test': 8000
    }
    
    logger.info(f"Downloading: train={splits_config['train']}, val={splits_config['validation']}, test={splits_config['test']}")
    logger.info("="*70)
    
    # Process each split
    for split, max_samples in splits_config.items():
        success = download_and_organize_split(split, max_samples)
        if not success:
            logger.error(f"Failed to process {split} split")
            sys.exit(1)
    
    logger.info("\n" + "="*70)
    logger.info("DATA LOADING COMPLETE")
    logger.info("="*70)
    logger.info(f"Images saved to: {OUTPUT_DIR}/{{train,validation,test}}/")
    logger.info("Each split contains:")
    logger.info("  - coco_XXXXXX.jpg (images)")
    logger.info("  - labels.json (class names)")
    logger.info("="*70)
    logger.info("Next: Run preprocessing.py")
    logger.info("="*70)

if __name__ == "__main__":
    main()
