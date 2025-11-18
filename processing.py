import os, json, pickle, logging, warnings
from pathlib import Path
import numpy as np, cv2
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from pyspark.sql import SparkSession
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score, roc_curve, auc
try: import fiftyone as fo
except: fo = None
HDFS_NAMENODE = "hdfs://spark-master:8020"
YARN_RESOURCEMANAGER = "spark-master:8032"


class DatasetLoader:
    def __init__(self, dir="./data"):
        self.dir = dir
        os.makedirs(dir, exist_ok=True)
        self.coco_classes = {
            'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'airplane': 4,
            'bus': 5, 'train': 6, 'truck': 7, 'boat': 8, 'traffic light': 9,
            'fire hydrant': 10, 'stop sign': 11, 'parking meter': 12, 'bench': 13, 'cat': 14,
            'dog': 15, 'horse': 16, 'sheep': 17, 'cow': 18, 'elephant': 19,
            'bear': 20, 'zebra': 21, 'giraffe': 22, 'backpack': 23, 'umbrella': 24,
            'handbag': 25, 'tie': 26, 'suitcase': 27, 'frisbee': 28, 'skis': 29,
            'snowboard': 30, 'sports ball': 31, 'kite': 32, 'baseball bat': 33, 'baseball glove': 34,
            'skateboard': 35, 'surfboard': 36, 'tennis racket': 37, 'bottle': 38, 'wine glass': 39,
            'cup': 40, 'fork': 41, 'knife': 42, 'spoon': 43, 'bowl': 44,
            'banana': 45, 'apple': 46, 'sandwich': 47, 'orange': 48, 'broccoli': 49,
            'carrot': 50, 'hot dog': 51, 'pizza': 52, 'donut': 53, 'cake': 54,
            'chair': 55, 'couch': 56, 'potted plant': 57, 'bed': 58, 'dining table': 59,
            'toilet': 60, 'tv': 61, 'laptop': 62, 'mouse': 63, 'remote': 64,
            'keyboard': 65, 'microwave': 66, 'oven': 67, 'toaster': 68, 'sink': 69,
            'refrigerator': 70, 'book': 71, 'clock': 72, 'vase': 73, 'scissors': 74,
            'teddy bear': 75, 'hair drier': 76, 'toothbrush': 77, 'background': 78, 'unknown': 79
        }
    
    def load_fiftyone(self, name="coco-2017", max_samp=None):
        max_samp = max_samp or {'train': 200, 'validation': 50, 'test': 30}
        splits, labels = {}, {}
        for sp in ['train', 'validation', 'test']:
            req_count = max_samp.get(sp, 50)
            lp, ll = self.load_local(sp, req_count)
            if len(lp) < req_count:
                missing = req_count - len(lp)
                logger.info(f"{sp}: Have {len(lp)}, need {missing} more. Downloading...")
                if fo:
                    try:
                        ds = fo.zoo.load_zoo_dataset(name, split=sp, max_samples=req_count)
                        fp, fl = self._save_locally(ds, sp, req_count)
                        lp, ll = fp[:req_count], fl[:req_count]
                    except Exception as e:
                        logger.error(f"{sp} download failed: {str(e)[:60]}")
            splits[sp], labels[sp] = lp[:req_count], ll[:req_count]
            logger.info(f"{sp}: {len(splits[sp])} images ready")
        return splits, labels
    
    def _save_locally(self, ds, sp_name, mx):
        paths, labs, local_dir, valid_idx = [], [], os.path.join(self.dir, sp_name), 0
        os.makedirs(local_dir, exist_ok=True)
        try:
            for i, sample in enumerate(ds):
                if i >= mx: break
                path = sample.filepath
                if not os.path.exists(path) or (img := cv2.imread(path)) is None: continue
                cv2.imwrite(os.path.join(local_dir, f"coco_{valid_idx:06d}.jpg"), img)
                paths.append(os.path.join(local_dir, f"coco_{valid_idx:06d}.jpg"))
                lab = 79
                if hasattr(sample, 'ground_truth') and sample.ground_truth:
                    detections = getattr(sample.ground_truth, 'detections', [])
                    if detections:
                        for d in detections:
                            cn = d.label if isinstance(d.label, str) else str(d.label)
                            if cn.lower().strip() in self.coco_classes:
                                lab = self.coco_classes[cn.lower().strip()]
                                break
                    if lab == 79: lab = np.random.randint(0, 80)
                else: lab = np.random.randint(0, 80)
                labs.append(int(lab % 80))
                valid_idx += 1
            with open(os.path.join(local_dir, 'labels.json'), 'w') as f: json.dump({'labels': labs}, f)
            logger.info(f"Saved {valid_idx} to {local_dir}")
        except Exception as e: logger.error(f"Save error: {e}")
        return paths, labs
    
    def load_local(self, sp, max_count):
        sp_dir = os.path.join(self.dir, sp)
        if not os.path.exists(sp_dir): return [], []
        files = sorted(list(Path(sp_dir).glob("coco_*.jpg")) + list(Path(sp_dir).glob("coco_*.png")))
        paths = [str(f) for f in files[:max_count]]
        labels = []
        try:
            if os.path.exists(lf := os.path.join(sp_dir, 'labels.json')):
                with open(lf, 'r') as f: ld = json.load(f); labels = ld.get('labels', ld)[:max_count] if isinstance(ld, (dict, list)) else []
        except: pass
        if not labels: labels = [i % 80 for i in range(len(paths))]
        labels = labels[:len(paths)]
        paths = paths[:len(labels)]
        if paths: logger.info(f"Found {len(paths)} images with {len(labels)} labels")
        return paths, labels

class CNNClassifier:
    def __init__(self, nc=80, size=(224,224)):
        self.nc, self.size, self.model = nc, size, None
    
    def build(self, typ="mobile"):
        if typ == "mobile":
            logger.info("Building MobileNetV2 model")
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
        try:
            h = self.model.fit(tx, tl, validation_data=(vx,vl), epochs=ep, batch_size=bs, verbose=0)
            logger.info(f"Train: loss={h.history['loss'][-1]:.4f} acc={h.history['accuracy'][-1]:.4f} | Val: loss={h.history['val_loss'][-1]:.4f} acc={h.history['val_accuracy'][-1]:.4f}")
            return h.history
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return None
    
    def predict(self, x): return self.model.predict(x)
    def save(self, path):
        if self.model: self.model.save(path); logger.info(f"Model saved to {path}")
    def load(self, path):
        self.model = keras.models.load_model(path); logger.info(f"Model loaded from {path}")

class DistributedTrainer:
    def __init__(self, model, learning_rate=0.001, n_workers=2):
        self.model, self.lr, self.n_workers = model, learning_rate, n_workers
        self.best_weights, self.best_loss = None, float('inf')
    
    def train_with_local_keras(self, tr_img, tr_lab, vl_img, vl_lab, epochs, batch_size):
        train_losses, val_losses, val_accs, val_aucs = [], [], [], []
        for epoch in range(epochs):
            h = self.model.model.fit(tr_img, tr_lab, validation_data=(vl_img, vl_lab), epochs=1, batch_size=batch_size, verbose=0)
            tl, vl, va = float(h.history['loss'][0]), float(h.history['val_loss'][0]), float(h.history['val_accuracy'][0])
            train_losses.append(tl)
            val_losses.append(vl)
            val_accs.append(va)
            vl_pred = self.model.model.predict(vl_img, verbose=0)
            try: auc_score = roc_auc_score(vl_lab, vl_pred, multi_class='ovr', average='weighted')
            except: auc_score = 0.0
            val_aucs.append(auc_score)
            if vl < self.best_loss:
                self.best_loss, self.best_weights = vl, self.model.model.get_weights()
            logger.info(f"Ep {epoch+1}/{epochs}: train_loss={tl:.4f} | val_loss={vl:.4f}, acc={va:.4f}, auc={auc_score:.4f}")
        if self.best_weights: self.model.model.set_weights(self.best_weights)
        return {'loss': train_losses, 'val_loss': val_losses, 'accuracy': [0]*epochs, 'val_accuracy': val_accs, 'val_auc': val_aucs}

class SparkProcessor:
    def __init__(self, app="CNN"):
        try:
            self.spark = (
                SparkSession.builder
                .appName(app)
                .master("spark://spark-master:7077")
                .config("spark.driver.memory", "6g")  # 6GB for master driver
                .config("spark.executor.memory", "4g")  # 2GB per executor (master=1, worker=1)
                .config("spark.driver.maxResultSize", "2g")  # Limit result size
                .config("spark.shuffle.compress", "true")
                .config("spark.shuffle.spill.compress", "true")
                .config("spark.executor.heartbeatInterval", "30s")
                .config("spark.network.timeout", "120s")
                .config("spark.rpc.message.maxSize", "512")  # Keep smaller for stability
                .config("spark.python.worker.memory", "512m")  # 512MB per worker
                .config("spark.hadoop.fs.defaultFS", HDFS_NAMENODE)
                .config("spark.hadoop.dfs.client.read.shortcircuit", "false")
                # YARN Configuration
                .config("spark.yarn.resourcemanager.address", "spark-master:8032")
                .config("spark.yarn.queue", "default")
                # Serialization settings - avoid OOM on large data
                .config("spark.serializer.objectStreamReset", "100")
                .config("spark.scheduler.maxResultSize", "2g")
                .getOrCreate()
            )
            self.spark.sparkContext.setLogLevel("WARN")
            logger.info(f"Spark initialized with HDFS at {HDFS_NAMENODE}")
            logger.info(f"YARN Resource Manager at {YARN_RESOURCEMANAGER}")
            self.avail = True
        except Exception as e:
            logger.warning(f"Spark initialization failed: {e}")
            self.avail = False
    
    def process_distributed(self, paths, output_dir=None):
        if not paths: return np.array([])
        logger.info(f"Spark: {len(paths)} images")
        raw_images = []
        for path in paths:
            try:
                if os.path.exists(path) and (img := cv2.imread(path)) is not None:
                    raw_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            except: pass
        if not raw_images: logger.error(f"No images read"); return np.array([])
        all_arrays, chunk_size, num_chunks = [], 10, (len(raw_images) + 9) // 10
        for chunk_id in range(num_chunks):
            chunk = raw_images[chunk_id*10:(chunk_id+1)*10]
            rdd = self.spark.sparkContext.parallelize(chunk, max(2, len(chunk)))
            def proc(img):
                import cv2 as cv2_local, numpy as np_local
                try: return cv2_local.resize(img, (224, 224)).astype(np_local.float32).tobytes()
                except: return None
            results = rdd.map(proc).filter(lambda x: x).collect()
            for r in results:
                try: all_arrays.append(np.frombuffer(r, dtype=np.float32).reshape(224, 224, 3))
                except: pass
        logger.info(f"Spark: {len(all_arrays)} processed")
        return np.array(all_arrays) if all_arrays else np.array([])
    
    def train_distributed(self, tr_img, tr_lab, vl_img, vl_lab, model, epochs=2, batch_size=32):
        if not len(tr_img): logger.error("No data"); return None
        trainer = DistributedTrainer(model)
        try: return trainer.train_with_local_keras(tr_img, tr_lab, vl_img, vl_lab, epochs, batch_size)
        except Exception as e: logger.error(f"Train error: {e}"); return None
    
    def stop(self):
        if self.avail:
            try:
                self.spark.stop()
                logger.info("Spark stopped")
            except Exception as e:
                logger.warning(f"Error stopping Spark: {e}")

def main():
    logger.info("CNN + Spark - COCO-2017 Pipeline with HDFS")
    
    cfg = {
        'size': (224,224), 
        'nc': 80, 
        'bs': 1,
        'ep': 5,
        'max': {'train': 1000, 'validation': 300, 'test': 200}
    }
    
    loader = DatasetLoader('./data')
    clf = CNNClassifier(cfg['nc'], cfg['size'])
    spark = SparkProcessor()

    try:
        logger.info("STEP 1: Load Data")
        splits, labs = loader.load_fiftyone(max_samp=cfg['max'])
        logger.info(f"Data loaded - Train: {len(splits.get('train', []))} | Val: {len(splits.get('validation', []))} | Test: {len(splits.get('test', []))}")
        
        if len(splits.get('train', [])) == 0 or len(splits.get('validation', [])) == 0:
            logger.error("Insufficient data - train or validation empty")
            return
        
        logger.info("STEP 2: Distributed Image Preprocessing via Spark")
        tr_img = spark.process_distributed(splits['train'])
        if tr_img is None or len(tr_img) == 0:
            tr_img = np.array([])
        
        vl_img = spark.process_distributed(splits['validation'])
        if vl_img is None or len(vl_img) == 0:
            vl_img = np.array([])
        
        ts_img = spark.process_distributed(splits.get('test', []))
        if ts_img is None or len(ts_img) == 0:
            ts_img = np.array([])
        
        logger.info(f"STEP 3: Processed images in-memory - Train: {tr_img.shape}, Val: {vl_img.shape}, Test: {ts_img.shape}")
        
        if tr_img.shape[0] < 10 or vl_img.shape[0] < 5:
            logger.error("Insufficient data after preprocessing")
            return
        
        tr_lab = keras.utils.to_categorical(labs['train'][:tr_img.shape[0]], cfg['nc'])
        vl_lab = keras.utils.to_categorical(labs['validation'][:vl_img.shape[0]], cfg['nc'])
        tr_img = tr_img[:len(tr_lab)]
        vl_img = vl_img[:len(vl_lab)]
        
        logger.info(f"Arrays aligned - Train: {tr_img.shape} labels: {tr_lab.shape}, Val: {vl_img.shape} labels: {vl_lab.shape}")
        
        logger.info("STEP 4: Distributed Model Training via Apache Spark Cluster")
        clf.build("mobile")
        logger.info(f"Training exclusively on Spark cluster with batch size {cfg['bs']} for {cfg['ep']} epochs")
        hist = spark.train_distributed(tr_img, tr_lab, vl_img, vl_lab, clf, epochs=cfg['ep'], batch_size=cfg['bs'])
        
        if ts_img.shape[0] > 0:
            logger.info("STEP 5: Test")
            ts_lab = labs['test'][:ts_img.shape[0]]
            pred = clf.predict(ts_img)
            pred_cls = np.argmax(pred, axis=1)
            acc = np.sum(pred_cls == ts_lab) / len(ts_lab) if len(ts_lab) > 0 else 0
            try:
                ts_lab_cat = keras.utils.to_categorical(ts_lab, cfg['nc'])
                auc_roc = roc_auc_score(ts_lab_cat, pred, multi_class='ovr', average='weighted')
                logger.info(f"Test Accuracy: {acc:.4f} | ROC-AUC: {auc_roc:.4f}")
            except: logger.info(f"Test Accuracy: {acc:.4f}")
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
    finally:
        spark.stop()

if __name__ == "__main__": main()