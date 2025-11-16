import subprocess, sys, os
import numpy as np
try: import cv2
except: cv2 = None
from pathlib import Path
from typing import List, Tuple, Dict
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
try: from pyspark.sql import SparkSession
except: SparkSession = None
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.optimizers import Adam
except: keras = None
try:
    import fiftyone.zoo as foz
    FIFTYONE_AVAILABLE = True
except: FIFTYONE_AVAILABLE = False

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
    def __init__(self, dir="./data"):
        self.dir = dir
        Path(self.dir).mkdir(exist_ok=True)
    def load_fiftyone(self, name="coco-2017", max_samp=None):
        if not FIFTYONE_AVAILABLE: return {}, {}
        max_samp = max_samp or {'train': 10000, 'val': 3000, 'test': 2000}
        splits, labels = {}, {}
        for sp in ['train', 'validation', 'test']:
            logger.info(f"Loading {sp}...")
            try:
                ds = foz.load_zoo_dataset(name, split=sp, max_samples=max_samp.get(sp, 100))
                splits[sp], labels[sp] = self._save_local(ds, sp, max_samp.get(sp, 100))
            except Exception as e:
                logger.error(f"Error {sp}: {e}")
                splits[sp], labels[sp] = [], []
        return splits, labels
    def _save_local(self, ds, sp_name, mx):
        paths, labs = [], []
        try:
            for i, sample in enumerate(ds):
                if i >= mx: break
                path, out = sample.filepath, os.path.join(self.dir, sp_name, f"{i:06d}.jpg")
                Path(self.dir, sp_name).mkdir(exist_ok=True)
                if os.path.exists(path):
                    img = cv2.imread(path)
                    if img is not None:
                        cv2.imwrite(out, img)
                        paths.append(out)
                        lab = 0
                        if hasattr(sample, 'ground_truth') and sample.ground_truth:
                            labs_list = sample.ground_truth.detections if hasattr(sample.ground_truth, 'detections') else []
                            lab = min(len(labs_list), 79) if labs_list else 0
                        labs.append(lab)
            logger.info(f"Saved {len(paths)} to {sp_name}")
        except Exception as e:
            logger.error(f"Save error {sp_name}: {e}")
        return paths, labs
    def load_local(self, sp):
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
            base = MobileNetV2(input_shape=(*self.size, 3), include_top=False, weights='imagenet')
            base.trainable = False
            self.model = models.Sequential([base, layers.GlobalAveragePooling2D(), layers.Dense(256, activation='relu'), 
                                           layers.Dropout(0.5), layers.Dense(128, activation='relu'), layers.Dropout(0.3), 
                                           layers.Dense(self.nc, activation='softmax')])
        else:
            self.model = models.Sequential([layers.Conv2D(32,(3,3),activation='relu',input_shape=(*self.size,3)), 
                                           layers.MaxPooling2D((2,2)), layers.Conv2D(64,(3,3),activation='relu'), 
                                           layers.MaxPooling2D((2,2)), layers.Flatten(), layers.Dense(64,activation='relu'),
                                           layers.Dropout(0.5), layers.Dense(self.nc, activation='softmax')])
        self.model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    def train(self, tx, tl, vx, vl, ep=10, bs=32):
        logger.info(f"Training {ep} epochs, bs={bs}...")
        h = self.model.fit(tx, tl, validation_data=(vx,vl), epochs=ep, batch_size=bs, verbose=1)
        logger.info("="*60)
        logger.info(f"Loss: {h.history['loss'][-1]:.4f} | Acc: {h.history['accuracy'][-1]:.4f}")
        logger.info(f"Val Loss: {h.history['val_loss'][-1]:.4f} | Val Acc: {h.history['val_accuracy'][-1]:.4f}")
        best = max(h.history['val_accuracy'])
        logger.info(f"Best Val Acc: {best:.4f} (Epoch {h.history['val_accuracy'].index(best)+1})")
        logger.info("="*60)
        return h.history
    def predict(self, x): return self.model.predict(x)
    def save(self, path):
        if self.model: self.model.save(path); logger.info(f"Saved to {path}")
    def load(self, path):
        self.model = keras.models.load_model(path); logger.info(f"Loaded from {path}")

class SparkProcessor:
    def __init__(self, app="CNN"):
        try:
            self.spark = SparkSession.builder.appName(app).config("spark.driver.maxResultSize","1g")\
                .config("spark.executor.heartbeatInterval","60s").config("spark.network.timeout","120s")\
                .config("spark.python.worker.memory","512m").getOrCreate()
            self.spark.sparkContext.setLogLevel("WARN")
            logger.info("Spark initialized")
            self.avail = True
        except Exception as e:
            logger.warning(f"Spark failed: {e}. Local mode.")
            self.avail = False
    def process(self, paths, prep):
        if not self.avail or not paths: return self._local(paths, prep)
        logger.info(f"Processing {len(paths)} images...")
        try:
            rdd = self.spark.sparkContext.parallelize(paths, min(8, len(paths)))
            def proc(p):
                try:
                    img = prep.load_image(p)
                    if img is None: return None
                    norm = prep.normalize(img)
                    return {'path': p, 'image': norm.tolist(), 'seg': prep.segment(norm), 'feat': prep.features(norm)}
                except: return None
            res = rdd.map(proc).filter(lambda x: x is not None).collect()
            logger.info(f"Processed {len(res)} images")
            return res
        except Exception as e:
            logger.error(f"Spark failed: {e}. Local fallback.")
            return self._local(paths, prep)
    def _local(self, paths, prep):
        res = []
        for p in paths:
            img = prep.load_image(p)
            if img: norm = prep.normalize(img); res.append({'path': p, 'image': norm.tolist(), 'seg': prep.segment(norm), 'feat': prep.features(norm)})
        return res
    def stop(self):
        if self.avail: self.spark.stop(); logger.info("Spark stopped")

def main():
    logger.info("="*60); logger.info("CNN Classification with Spark"); logger.info("="*60)
    cfg = {'dir': './data', 'mdir': './models', 'size': (224,224), 'nc': 80, 'bs': 64, 'ep': 10,
           'max': {'train': 10000, 'val': 3000, 'test': 2000}}
    Path(cfg['mdir']).mkdir(exist_ok=True)
    prep, loader, clf, spark = ImagePreprocessor(cfg['size']), DatasetLoader(cfg['dir']), CNNClassifier(cfg['nc'], cfg['size']), SparkProcessor()
    
    try:
        logger.info("\nStep 1: Loading...")
        splits, labs = loader.load_fiftyone(max_samp=cfg['max'])
        if not splits or all(not v for v in splits.values()):
            logger.info("Local fallback...")
            for sp in ['train', 'validation', 'test']:
                f, l = loader.load_local(sp)
                splits[sp], labs[sp] = f, l
        
        logger.info("\nStep 2: Preprocessing...")
        tr_dat = spark.process(splits['train'][:cfg['max']['train']], prep)
        vl_dat = spark.process(splits['validation'][:cfg['max']['val']], prep)
        ts_dat = spark.process(splits['test'][:cfg['max']['test']], prep)
        logger.info(f"Train: {len(tr_dat)}, Val: {len(vl_dat)}, Test: {len(ts_dat)}")
        
        if len(tr_dat) > 0 and len(vl_dat) > 0:
            tr_img = np.array([np.array(d['image']) for d in tr_dat])
            tr_lab_real = labs['train'][:len(tr_dat)]
            tr_lab = keras.utils.to_categorical(tr_lab_real, cfg['nc'])
            vl_img = np.array([np.array(d['image']) for d in vl_dat])
            vl_lab_real = labs['validation'][:len(vl_dat)]
            vl_lab = keras.utils.to_categorical(vl_lab_real, cfg['nc'])
            ts_img = np.array([np.array(d['image']) for d in ts_dat]) if ts_dat else np.array([])
            ts_lab_real = labs['test'][:len(ts_dat)]
            
            logger.info(f"\nDataset: Train={len(tr_img)}({len(set(tr_lab_real))}cls) Val={len(vl_img)} Test={len(ts_img)}")
            logger.info("\nStep 3: Training...")
            clf.build("mobile")
            clf.train(tr_img, tr_lab, vl_img, vl_lab, cfg['ep'], cfg['bs'])
            
            if len(ts_img) > 0:
                logger.info("\nStep 4: Testing...")
                pred = clf.predict(ts_img)
                pred_cls = np.argmax(pred, axis=1)
                acc = np.sum(pred_cls == ts_lab_real) / len(ts_lab_real)
                logger.info(f"Test Acc: {acc:.4f} ({np.sum(pred_cls == ts_lab_real)}/{len(ts_lab_real)})")
                for i in range(min(5, len(pred))):
                    m = "✓" if pred_cls[i] == ts_lab_real[i] else "✗"
                    logger.info(f"  {m} Img {i}: Pred={pred_cls[i]}, True={ts_lab_real[i]}, Conf={pred[i][pred_cls[i]]:.4f}")
            
            clf.save(os.path.join(cfg['mdir'], 'cnn.h5'))
            logger.info("\nDone!")
        else:
            logger.error("No data")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        spark.stop()

if __name__ == "__main__": main()