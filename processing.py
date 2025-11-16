import subprocess, sys, os
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from pyspark.sql import SparkSession
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
try:
    import fiftyone as fo
except: fo = None


class ImagePreprocessor:
    def __init__(self, size=(224,224)): self.size = size
    
    def load_image(self, path):
        try:
            img = cv2.imread(path)
            if img is None: return None
            return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), self.size)
        except: return None
    def normalize(self, img): return img.astype(np.float32) / 255.0
    def segment(self, img):
        try:
            gray = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            objs = [{'bbox': cv2.boundingRect(c)[:4], 'area': cv2.contourArea(c)} for c in contours if cv2.contourArea(c) > 100]
            return {'contours': len(contours), 'objects': objs}
        except: return None
    
    def features(self, img):
        gray = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return {'mean': float(gray.mean()), 'std': float(gray.std()), 'edge_density': float(edges.sum()/(edges.shape[0]*edges.shape[1]))}

class DatasetLoader:
    def __init__(self, dir="hdfs://localhost:9000/data"):
        self.dir = dir
        self.use_hdfs = dir.startswith("hdfs://")
    
    def check_hdfs_exists(self, sp_name, expected_count):
        if not self.use_hdfs: return False
        hdfs_path = f"{self.dir}/{sp_name}"
        result = subprocess.run(f"hdfs dfs -count {hdfs_path} 2>/dev/null", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            try:
                parts = result.stdout.strip().split()
                file_count = int(parts[-1])
                if file_count >= expected_count * 0.8: return True
            except: pass
        return False
    
    def load_fiftyone(self, name="coco-2017", max_samp=None):
        max_samp = max_samp or {'train': 10000, 'val': 3000, 'test': 2000}
        splits, labels = {}, {}
        for sp in ['train', 'validation', 'test']:
            sp_name = sp if sp != 'validation' else 'val'
            actual_sp = 'validation' if sp_name == 'val' else sp_name
            hdfs_sp = sp_name
            exp_count = max_samp.get(sp_name, 100)
            
            if self.check_hdfs_exists(hdfs_sp, exp_count):
                logger.info(f"Found {hdfs_sp} in HDFS, loading from there...")
                splits[actual_sp], labels[actual_sp] = self.load_local(hdfs_sp)
            elif fo:
                logger.info(f"Loading {actual_sp} from FiftyOne and saving to HDFS...")
                try:
                    ds = fo.load_zoo_dataset(name, split=actual_sp, max_samples=exp_count)
                    splits[actual_sp], labels[actual_sp] = self._save_to_hdfs(ds, hdfs_sp, exp_count)
                except Exception as e:
                    logger.warning(f"FiftyOne load failed {actual_sp}: {e}")
                    splits[actual_sp], labels[actual_sp] = self.load_local(hdfs_sp)
            else:
                logger.info(f"FiftyOne unavailable, loading {actual_sp} from HDFS...")
                splits[actual_sp], labels[actual_sp] = self.load_local(hdfs_sp)
            
            if not splits[actual_sp]:
                logger.error(f"CRITICAL: No data found for {actual_sp} in HDFS or FiftyOne!")
        return splits, labels
    
    def _save_to_hdfs(self, ds, sp_name, mx):
        paths, labs = [], []
        try:
            for i, sample in enumerate(ds):
                if i >= mx: break
                path = sample.filepath
                if not os.path.exists(path): continue
                img = cv2.imread(path)
                if img is None: continue
                _, img_enc = cv2.imencode('.jpg', img)
                hdfs_path = f"{self.dir}/{sp_name}/{i:06d}.jpg"
                subprocess.run(f"hdfs dfs -mkdir -p {self.dir}/{sp_name}", shell=True, capture_output=True)
                subprocess.run(f"echo '{img_enc.tobytes()}' | hdfs dfs -put -f - {hdfs_path}", shell=True, capture_output=True)
                paths.append(hdfs_path)
                lab = 0
                if hasattr(sample, 'ground_truth') and sample.ground_truth:
                    labs_list = sample.ground_truth.detections if hasattr(sample.ground_truth, 'detections') else []
                    lab = min(len(labs_list), 79) if labs_list else 0
                labs.append(lab)
            logger.info(f"Saved {len(paths)} to HDFS:{sp_name}")
        except Exception as e:
            logger.error(f"Save error {sp_name}: {e}")
        return paths, labs
    
    def load_local(self, sp):
        if self.use_hdfs:
            hdfs_path = f"{self.dir}/{sp}"
            result = os.popen(f"hdfs dfs -ls {hdfs_path} 2>/dev/null | awk '{{print $NF}}'").read().strip().split('\n')
            files = [f for f in result if f and (f.endswith('.jpg') or f.endswith('.png'))]
            logger.info(f"Found {len(files)} in HDFS:{sp}")
        else:
            sp_dir = os.path.join(self.dir, sp)
            if not os.path.exists(sp_dir): return [], []
            files = sorted([str(f) for f in list(Path(sp_dir).glob("*.jpg")) + list(Path(sp_dir).glob("*.png"))])
            logger.info(f"Found {len(files)} in {sp}")
        return files, list(range(len(files)))

class CNNClassifier:
    def __init__(self, nc=80, size=(224,224)):
        self.nc, self.size, self.model = nc, size, None
    
    def build(self, typ="mobile"):
        if typ == "mobile":
            logger.info("Building MobileNetV2...")
            base = MobileNetV2(input_shape=(*self.size, 3), include_top=False, weights=None)
            base.trainable = False
            self.model = models.Sequential([base, layers.GlobalAveragePooling2D(), layers.Dense(256, activation='relu'), 
                                           layers.Dropout(0.5), layers.Dense(128, activation='relu'), layers.Dropout(0.3), 
                                           layers.Dense(self.nc, activation='softmax')])
        else:
            self.model = models.Sequential([layers.Conv2D(32,(3,3),activation='relu',input_shape=(*self.size,3)), 
                                           layers.MaxPooling2D((2,2)), layers.Conv2D(64,(3,3),activation='relu'), 
                                           layers.MaxPooling2D((2,2)), layers.Flatten(), layers.Dense(64,activation='relu'),
                                           layers.Dropout(0.5), layers.Dense(self.nc, activation='softmax')])
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    def train(self, tx, tl, vx, vl, ep=10, bs=32):
        logger.info(f"Training {ep} epochs, bs={bs}...")
        try:
            h = self.model.fit(tx, tl, validation_data=(vx,vl), epochs=ep, batch_size=bs, verbose=0)
            logger.info("="*60)
            logger.info(f"Loss: {h.history['loss'][-1]:.4f} | Acc: {h.history['accuracy'][-1]:.4f}")
            logger.info(f"Val Loss: {h.history['val_loss'][-1]:.4f} | Val Acc: {h.history['val_accuracy'][-1]:.4f}")
            best = max(h.history['val_accuracy'])
            logger.info(f"Best Val Acc: {best:.4f} (Epoch {h.history['val_accuracy'].index(best)+1})")
            logger.info("="*60)
            return h.history
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return None
    
    def predict(self, x): return self.model.predict(x)
    def save(self, path):
        if self.model: self.model.save(path); logger.info(f"Saved to {path}")
    def load(self, path):
        self.model = keras.models.load_model(path); logger.info(f"Loaded from {path}")

class SparkProcessor:
    def __init__(self, app="CNN"):
        try:
            self.spark = SparkSession.builder.appName(app).master("local[*]").config("spark.driver.memory","2g")\
                .config("spark.executor.memory","1g").config("spark.driver.maxResultSize","1g")\
                .config("spark.executor.maxResultSize","1g").config("spark.executor.heartbeatInterval","60s")\
                .config("spark.network.timeout","120s").config("spark.python.worker.memory","512m")\
                .config("spark.shuffle.compress","true").config("spark.shuffle.spill.compress","true").getOrCreate()
            self.spark.sparkContext.setLogLevel("WARN")
            logger.info("Spark initialized with HDFS optimizations")
            self.avail = True
        except Exception as e:
            logger.warning(f"Spark failed: {e}. Local mode.")
            self.avail = False
    
    def process(self, paths, prep):
        if not self.avail or not paths: return self._local(paths, prep)
        logger.info(f"Processing {len(paths)} images via Spark RDD...")
        try:
            rdd = self.spark.sparkContext.binaryFiles(f"{paths[0].rsplit('/',1)[0]}/*") if paths and paths[0].startswith('hdfs://') else self.spark.sparkContext.parallelize(paths, min(8, len(paths)))
            def proc_bin(kv):
                import numpy as np_local
                import cv2 as cv2_local
                try:
                    path, data = kv
                    nparr = np_local.frombuffer(data, np_local.uint8)
                    img = cv2_local.imdecode(nparr, cv2_local.IMREAD_COLOR)
                    if img is None: return None
                    img = cv2_local.cvtColor(img, cv2_local.COLOR_BGR2RGB)
                    img = cv2_local.resize(img, (224, 224))
                    norm = img.astype(np_local.float32) / 255.0
                    gray = cv2_local.cvtColor((norm*255).astype(np_local.uint8), cv2_local.COLOR_RGB2GRAY)
                    edges = cv2_local.Canny(gray, 100, 200)
                    edge_d = float(edges.sum()/(edges.shape[0]*edges.shape[1]))
                    return {'path': path, 'image': norm.tolist(), 'edge_density': edge_d}
                except: return None
            res = rdd.map(proc_bin).filter(lambda x: x is not None).collect()
            logger.info(f"Processed {len(res)} images successfully")
            return res
        except Exception as e:
            logger.error(f"Spark processing failed: {e}")
            return self._local(paths, prep)
    
    def _local(self, paths, prep):
        res = []
        for p in paths:
            try:
                result = subprocess.run(f"hdfs dfs -cat {p}", shell=True, capture_output=True)
                if result.returncode == 0:
                    nparr = np.frombuffer(result.stdout, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (224, 224))
                        norm = img.astype(np.float32) / 255.0
                        res.append({'path': p, 'image': norm.tolist()})
            except: pass
        return res
    
    def stop(self):
        if self.avail: self.spark.stop(); logger.info("Spark stopped")

def main():
    logger.info("="*60); 
    logger.info("CNN + Spark + HDFS - No Local FS"); 
    logger.info("="*60)
    
    cfg = {
        'dir': 'hdfs://localhost:9000/data', 
        'size': (224,224), 
        'nc': 80, 
        'bs': 128, 
        'ep': 5,
        'max': {'train': 500, 'val': 150, 'test': 100}
        }
    
    subprocess.run(f"hdfs dfs -mkdir -p {cfg['dir']}", shell=True, capture_output=True)
    prep = ImagePreprocessor(cfg['size'])
    loader = DatasetLoader(cfg['dir'])
    clf = CNNClassifier(cfg['nc'], cfg['size'])
    spark = SparkProcessor()

    try:
        logger.info("\nStep 1: Loading datasets (HDFS or FiftyOne - mandatory)...")
        splits, labs = loader.load_fiftyone(max_samp=cfg['max'])
        
        hdfs_status = {sp: (len(splits.get(sp, [])), len(labs.get(sp, []))) for sp in ['train', 'validation', 'test']}
        logger.info(f"Data status: train={hdfs_status['train'][0]} val={hdfs_status['validation'][0]} test={hdfs_status['test'][0]}")
        
        logger.info("\nStep 2: Distributed preprocessing from HDFS...")
        tr_dat = spark.process(splits['train'][:cfg['max']['train']], prep)
        vl_dat = spark.process(splits['validation'][:cfg['max']['val']], prep)
        ts_dat = spark.process(splits['test'][:cfg['max']['test']], prep)
        logger.info(f"Loaded - Train: {len(tr_dat)}, Val: {len(vl_dat)}, Test: {len(ts_dat)}")
        
        if len(tr_dat) > 0 and len(vl_dat) > 0:
            logger.info("\nStep 3: Building arrays...")
            tr_img = np.array([np.array(d['image']) for d in tr_dat])
            tr_lab_real = labs['train'][:len(tr_dat)]
            tr_lab = keras.utils.to_categorical(tr_lab_real, cfg['nc'])
            vl_img = np.array([np.array(d['image']) for d in vl_dat])
            vl_lab_real = labs['validation'][:len(vl_dat)]
            vl_lab = keras.utils.to_categorical(vl_lab_real, cfg['nc'])
            ts_img = np.array([np.array(d['image']) for d in ts_dat]) if ts_dat else np.array([])
            ts_lab_real = labs['test'][:len(ts_dat)]
            
            logger.info(f"Train={len(tr_img)}({len(set(tr_lab_real))}cls) Val={len(vl_img)} Test={len(ts_img)}")
            logger.info("\nStep 4: Training CNN...")
            clf.build("mobile")
            clf.train(tr_img, tr_lab, vl_img, vl_lab, cfg['ep'], cfg['bs'])
            
            if len(ts_img) > 0:
                logger.info("\nStep 5: Testing...")
                pred = clf.predict(ts_img)
                pred_cls = np.argmax(pred, axis=1)
                acc = np.sum(pred_cls == ts_lab_real) / len(ts_lab_real)
                logger.info(f"Test Acc: {acc:.4f} ({np.sum(pred_cls == ts_lab_real)}/{len(ts_lab_real)})")
                for i in range(min(3, len(pred))):
                    m = "✓" if pred_cls[i] == ts_lab_real[i] else "✗"
                    logger.info(f"  {m} Img {i}: Pred={pred_cls[i]}, True={ts_lab_real[i]}")
            logger.info("\n✓ Complete - No model saved, all from HDFS!")
        else:
            logger.error("Insufficient data")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        spark.stop()

if __name__ == "__main__": main()