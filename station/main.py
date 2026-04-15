#!/usr/bin/env python3
# =============================================================
#  Central Station — Main Loop  (Raspberry Pi 4)
#  PFE TUNSA — Interférométrie IoT
#
#  Reçoit telemetry T1 + T2 via WiFi HTTP POST
#  Pipeline AI à chaque cycle :
#    ① Fringe Analyzer  (CNN+BiLSTM) → |V|, φ, SNR
#    ② Sky Reconstructor (U-Net)     → image sky 64×64
#    ③ Source Classifier (RF)        → classe + confiance
#  Stocke dans SQLite (IoT time-series)
#  Expose résultats via Flask REST API port 5000
# =============================================================

import json, logging, os, signal, sys, time, threading, sqlite3, base64
from datetime import datetime
from pathlib import Path

import numpy as np
from flask import Flask, request, jsonify

import config, lora_receiver, fso_receiver, camera, gps_parser, data_logger

# ── Logging ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-12s] %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("station")

# ── Chemins ──────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR.parent / "ai_pipeline" / "models"
DB_PATH    = BASE_DIR / "telemetry.db"
LAMBDA_NM  = 625e-9

# =============================================================
#  A — PIPELINE AI
# =============================================================

class FringeAnalyzer:
    """CNN+BiLSTM — extrait |V|, φ, SNR depuis signal BPW34."""
    def __init__(self):
        self.ready = False
        try:
            try:    import tflite_runtime.interpreter as tflite
            except: import tensorflow.lite as tflite
            path = MODELS_DIR / "fringe_analyzer.tflite"
            self.interp = tflite.Interpreter(model_path=str(path))
            self.interp.allocate_tensors()
            self.inp_idx = self.interp.get_input_details()[0]["index"]
            self.out_idx = self.interp.get_output_details()[0]["index"]
            self.ready = True
            logger.info("  AI FringeAnalyzer    — OK")
        except Exception as e:
            logger.warning("  AI FringeAnalyzer    — DISABLED (%s)", e)

    def analyze(self, samples: list) -> dict:
        if not self.ready:
            return {"visibility": None, "phase_rad": None, "snr_db": None}
        try:
            sig = np.array(samples, dtype=np.float32)
            if len(sig) < 2:
                t = np.linspace(0, 1, 1024)
                sig = float(sig[0]) if len(sig) > 0 else 0.5
                sig = np.full(1024, sig) + 0.1 * np.sin(2*np.pi*10*t)
                sig = sig.astype(np.float32)
            elif len(sig) < 1024:
                from scipy.interpolate import interp1d
                f = interp1d(np.linspace(0,1,len(sig)), sig, kind="linear")
                sig = f(np.linspace(0,1,1024)).astype(np.float32)
            else:
                sig = sig[:1024]
            sig = np.clip(sig, 0, None) / (sig.max() + 1e-10)
            self.interp.set_tensor(self.inp_idx, sig.reshape(1,1024,1))
            self.interp.invoke()
            out = self.interp.get_tensor(self.out_idx)[0]
            return {"visibility": round(float(np.clip(out[0],0,1)),4),
                    "phase_rad":  round(float(out[1]*np.pi),4),
                    "snr_db":     round(float(np.clip(out[2],0,1)*40),2)}
        except Exception as e:
            logger.error("  FringeAnalyzer: %s", e)
            return {"visibility": None, "phase_rad": None, "snr_db": None}


class SourceClassifier:
    """Random Forest — classifie la source (7 classes)."""
    CLASSES = {0:"Étoile ponctuelle",1:"Binaire serrée",2:"Binaire large",
               3:"Nébuleuse étendue",4:"Disque circumstellaire",
               5:"Objet compact",6:"Sources multiples"}

    def __init__(self):
        self.ready = False
        try:
            import pickle
            with open(MODELS_DIR/"random_forest.pkl","rb") as f:
                b = pickle.load(f)
            self.rf, self.scaler = b["model"], b["scaler"]
            self.ready = True
            logger.info("  AI SourceClassifier  — OK")
        except Exception as e:
            logger.warning("  AI SourceClassifier  — DISABLED (%s)", e)

    def classify(self, fringe: dict, t1: dict, t2: dict) -> dict:
        if not self.ready:
            return {"class_id":None,"class_name":"N/A","confidence":None,"probabilities":None}
        try:
            feat = self._features(fringe, t1 or {}, t2 or {})
            proba = self.rf.predict_proba(self.scaler.transform([feat]))[0]
            cid   = int(np.argmax(proba))
            return {"class_id":cid,"class_name":self.CLASSES[cid],
                    "confidence":round(float(proba[cid]),4),
                    "probabilities":{self.CLASSES[i]:round(float(p),3)
                                     for i,p in enumerate(proba)}}
        except Exception as e:
            logger.error("  Classifier: %s", e)
            return {"class_id":None,"class_name":"ERROR","confidence":None,"probabilities":None}

    def _features(self, fr, t1, t2):
        V   = fr.get("visibility") or 0.5
        phi = fr.get("phase_rad")  or 0.0
        snr = fr.get("snr_db")     or 15.0
        d1  = float(t1.get("distance",0) or 0)/1000.0
        d2  = float(t2.get("distance",0) or 0)/1000.0
        B   = abs(d1-d2) if (d1 and d2) else 0.25
        ang = (float(t1.get("pitch",0) or 0)+float(t2.get("pitch",0) or 0))/2
        l1  = float(t1.get("light_intensity",500) or 500)/4095.0
        l2  = float(t2.get("light_intensity",500) or 500)/4095.0
        Bp  = B*np.cos(np.radians(ang))
        u   = Bp/LAMBDA_NM
        return [
            (l1+l2)/2, abs(l1-l2), V*2*np.sqrt(l1*l2+1e-10), V*0.5,
            V*10, abs(l1-l2)/2, max(l1,l2), (l1**2+l2**2)/2,
            u*2e-6, V*max(l1,l2), (l1+l2)/2, V*2,
            V**2, V*0.1, V**2*0.1, V*0.05,
            u*2e-6*0.8, V*max(l1,l2)*1.1, V*(l1+l2)/2, V*0.2,
            V, snr, V**2, u*2e-6, 0.05,
            1.0/(u*2e-6+1e-10), 0.0, -V*np.log(V+1e-10),
            B, ang, (d1+d2)/2, u
        ]


class SkyReconstructor:
    """U-Net 64×64 — reconstruit image sky depuis visibilités."""
    def __init__(self):
        self.ready = False
        try:
            try:    import tflite_runtime.interpreter as tflite
            except: import tensorflow.lite as tflite
            path = MODELS_DIR / "unet_reconstructor.tflite"
            self.interp = tflite.Interpreter(model_path=str(path))
            self.interp.allocate_tensors()
            self.inp_idx = self.interp.get_input_details()[0]["index"]
            self.out_idx = self.interp.get_output_details()[0]["index"]
            self.ready = True
            logger.info("  AI SkyReconstructor  — OK")
        except Exception as e:
            logger.warning("  AI SkyReconstructor  — DISABLED (%s)", e)

    def reconstruct(self, V, phi, B, ang) -> np.ndarray:
        if not self.ready: return None
        try:
            dirty = self._dirty(V, phi, B, ang)
            self.interp.set_tensor(self.inp_idx, dirty.reshape(1,64,64,1))
            self.interp.invoke()
            return self.interp.get_tensor(self.out_idx)[0,:,:,0]
        except Exception as e:
            logger.error("  SkyReconstructor: %s", e); return None

    def _dirty(self, V, phi, B, ang, sz=64):
        from scipy.fft import ifft2, fftshift, ifftshift
        cx = sz//2
        Bp  = B*np.cos(np.radians(ang))
        u_p = int((Bp/LAMBDA_NM)/1e7*(sz//4)+cx) % sz
        V_c = V*np.exp(1j*phi)
        g   = np.zeros((sz,sz),dtype=complex)
        g[cx,u_p] = V_c
        g[sz-cx-1,sz-u_p-1] = np.conj(V_c)
        d = np.real(fftshift(ifft2(ifftshift(g))))
        d = np.clip(d,0,None)
        return (d/d.max() if d.max()>0 else d).astype(np.float32)

    def to_b64(self, sky: np.ndarray) -> str:
        import io
        try:
            from PIL import Image
            img = Image.fromarray((sky*255).astype(np.uint8),"L").resize((256,256),Image.NEAREST)
            buf = io.BytesIO(); img.save(buf,"PNG")
            return base64.b64encode(buf.getvalue()).decode()
        except: return ""


# Instances globales
fringe_analyzer   = None
source_classifier = None
sky_reconstructor = None

latest_ai   = {"timestamp":None,"visibility":None,"phase_rad":None,
               "snr_db":None,"class_id":None,"class_name":"—",
               "confidence":None,"probabilities":None,"sky_image_b64":None}
ai_lock = threading.Lock()


def run_ai_pipeline(t1, t2, samples):
    global latest_ai
    t1, t2 = t1 or {}, t2 or {}

    fr  = fringe_analyzer.analyze(samples)
    logger.info("  AI① V=%.3f φ=%.3f SNR=%.1fdB",
                fr.get("visibility") or 0, fr.get("phase_rad") or 0,
                fr.get("snr_db") or 0)

    V   = fr.get("visibility") or 0.5
    phi = fr.get("phase_rad")  or 0.0
    d1  = float(t1.get("distance",250) or 250)/1000.0
    d2  = float(t2.get("distance",250) or 250)/1000.0
    B   = max(abs(d1-d2), 0.05)
    ang = (float(t1.get("pitch",0) or 0)+float(t2.get("pitch",0) or 0))/2

    sky = sky_reconstructor.reconstruct(V, phi, B, ang)
    b64 = sky_reconstructor.to_b64(sky) if sky is not None else None
    logger.info("  AI② Sky: %s", "OK 64×64" if sky is not None else "disabled")

    clf = source_classifier.classify(fr, t1, t2)
    logger.info("  AI③ %s (%.2f)", clf.get("class_name"), clf.get("confidence") or 0)

    res = {"timestamp":datetime.now().isoformat(),
           "visibility":fr.get("visibility"), "phase_rad":fr.get("phase_rad"),
           "snr_db":fr.get("snr_db"), "class_id":clf.get("class_id"),
           "class_name":clf.get("class_name"), "confidence":clf.get("confidence"),
           "probabilities":clf.get("probabilities"), "sky_image_b64":b64,
           "baseline_m":round(B,4), "imu_angle_deg":round(ang,2)}

    with ai_lock:
        latest_ai = res
    return res


# =============================================================
#  B — SQLITE
# =============================================================
_db = None
_db_lock = threading.Lock()


def db_setup():
    global _db
    try:
        _db = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        c   = _db.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL,
            node_id TEXT, pitch REAL, roll REAL, yaw REAL,
            accel_x REAL, accel_y REAL, accel_z REAL,
            distance_mm REAL, light REAL, temperature REAL,
            counter INTEGER, rssi REAL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS ai_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL,
            visibility REAL, phase_rad REAL, snr_db REAL,
            class_id INTEGER, class_name TEXT, confidence REAL,
            baseline_m REAL, imu_angle REAL)""")
        _db.commit()
        logger.info("  SQLite DB            — OK (%s)", DB_PATH.name)
        return True
    except Exception as e:
        logger.error("  SQLite DB            — FAILED (%s)", e); return False


def db_log_tel(node, data):
    if not _db or not data: return
    try:
        with _db_lock:
            _db.execute(
                "INSERT INTO telemetry (ts,node_id,pitch,roll,yaw,"
                "accel_x,accel_y,accel_z,distance_mm,light,temperature,counter,rssi) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (time.time(), node,
                 data.get("pitch"), data.get("roll"), data.get("yaw"),
                 data.get("accel_x"), data.get("accel_y"), data.get("accel_z"),
                 data.get("distance"), data.get("light_intensity"),
                 data.get("temperature"), data.get("counter"), data.get("rssi")))
            _db.commit()
    except Exception as e: logger.error("  DB tel: %s", e)


def db_log_ai(r):
    if not _db or not r: return
    try:
        with _db_lock:
            _db.execute(
                "INSERT INTO ai_results (ts,visibility,phase_rad,snr_db,"
                "class_id,class_name,confidence,baseline_m,imu_angle) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (time.time(), r.get("visibility"), r.get("phase_rad"),
                 r.get("snr_db"), r.get("class_id"), r.get("class_name"),
                 r.get("confidence"), r.get("baseline_m"), r.get("imu_angle_deg")))
            _db.commit()
    except Exception as e: logger.error("  DB ai: %s", e)


def db_query(table, node=None, limit=100):
    if not _db: return []
    try:
        with _db_lock:
            if node:
                rows = _db.execute(
                    f"SELECT * FROM {table} WHERE node_id=? ORDER BY ts DESC LIMIT ?",
                    (node, limit)).fetchall()
            else:
                rows = _db.execute(
                    f"SELECT * FROM {table} ORDER BY ts DESC LIMIT ?",
                    (limit,)).fetchall()
            cols = [d[0] for d in _db.execute(f"PRAGMA table_info({table})").fetchall()]
            return [dict(zip(cols, r)) for r in rows]
    except Exception as e:
        logger.error("  DB query: %s", e); return []


# =============================================================
#  FLASK
# =============================================================
app = Flask(__name__)
app.logger.setLevel(logging.WARNING)

lat_t1, lat_t2 = None, None
tel_lock = threading.Lock()


@app.route("/telemetry", methods=["POST"])
def recv_telemetry():
    global lat_t1, lat_t2
    try:
        data = request.get_json(force=True)
        data["_received_at"] = datetime.now().isoformat()
        node = data.get("node","UNKNOWN")
        with tel_lock:
            if node == "T1": lat_t1 = data
            else:            lat_t2 = data
        db_log_tel(node, data)
        logger.info("  HTTP [%s] counter=%s pitch=%.1f", node,
                    data.get("counter","?"), float(data.get("pitch",0) or 0))
        return jsonify({"status":"ok"}), 200
    except Exception as e:
        return jsonify({"status":"error","msg":str(e)}), 400


@app.route("/api/status")
def api_status():
    with tel_lock:  t1,t2 = lat_t1, lat_t2
    with ai_lock:   ai = {k:v for k,v in latest_ai.items() if k!="sky_image_b64"}
    return jsonify({"status":"running","timestamp":datetime.now().isoformat(),
                    "t1_online":t1 is not None,"t2_online":t2 is not None,
                    "ai_ready":fringe_analyzer.ready if fringe_analyzer else False,
                    "last_ai":ai})

@app.route("/api/telemetry/latest")
def api_tel_latest():
    with tel_lock: return jsonify({"T1":lat_t1,"T2":lat_t2})

@app.route("/api/telemetry/history")
def api_tel_history():
    node  = request.args.get("node")
    limit = int(request.args.get("limit",100))
    rows  = db_query("telemetry", node=node, limit=limit)
    return jsonify({"count":len(rows),"data":rows})

@app.route("/api/ai/latest")
def api_ai_latest():
    with ai_lock: return jsonify(latest_ai)

@app.route("/api/ai/history")
def api_ai_history():
    limit = int(request.args.get("limit",50))
    return jsonify({"count":0,"data":db_query("ai_results",limit=limit)})

@app.route("/api/sky")
def api_sky():
    with ai_lock: b64 = latest_ai.get("sky_image_b64")
    if b64: return jsonify({"sky_image_b64":b64,"format":"png","size":"256x256"})
    return jsonify({"sky_image_b64":None}), 204


def start_flask():
    app.run(host="0.0.0.0", port=5000, use_reloader=False, threaded=True)


# =============================================================
#  SHUTDOWN
# =============================================================
running = True
def _sig(s,f):
    global running; logger.info("Shutdown"); running = False
signal.signal(signal.SIGINT, _sig)
signal.signal(signal.SIGTERM,_sig)


# =============================================================
#  SETUP + LOOP
# =============================================================
AI_EVERY = 3   # pipeline AI tous les N cycles


def setup():
    global fringe_analyzer, source_classifier, sky_reconstructor
    logger.info("="*55)
    logger.info("  Central Station — PFE TUNSA")
    logger.info("="*55)

    threading.Thread(target=start_flask, daemon=True).start()
    logger.info("  Flask              — port 5000")

    lora_ok = lora_receiver.setup()
    fso_ok  = fso_receiver.setup()
    cam_ok  = camera.setup()
    gps_ok  = gps_parser.setup()
    log_ok  = data_logger.setup()
    db_ok   = db_setup()

    fringe_analyzer   = FringeAnalyzer()
    source_classifier = SourceClassifier()
    sky_reconstructor = SkyReconstructor()

    logger.info("-"*55)
    for name,ok in [("LoRa",lora_ok),("FSO",fso_ok),("Camera",cam_ok),
                    ("GPS",gps_ok),("CSV",log_ok),("SQLite",db_ok)]:
        logger.info("  %-10s — %s", name, "OK" if ok else "DISABLED")
    logger.info("-"*55)
    if not log_ok: sys.exit(1)


def loop():
    cycle = 0
    while running:
        cycle += 1
        t0 = time.time()
        logger.info("── Cycle %d (%s) ──", cycle, datetime.now().strftime("%H:%M:%S"))

        with tel_lock: t1, t2 = lat_t1, lat_t2

        for node, data in [("T1",t1),("T2",t2)]:
            if data:
                logger.info("  %s: pitch=%.1f dist=%s light=%s", node,
                            float(data.get("pitch",0) or 0),
                            data.get("distance","—"), data.get("light_intensity","—"))
            else:
                logger.info("  %s: no data", node)

        # FSO fallback
        try:
            t2_fso = fso_receiver.receive()
            if t2_fso and not t2:
                t2 = t2_fso; db_log_tel("T2_FSO", t2_fso)
        except: pass

        # Camera
        img = None
        try: img = camera.capture()
        except Exception as e: logger.error("  CAM: %s", e)

        # GPS
        gps = None
        try: gps = gps_parser.read()
        except: pass

        # CSV log
        try: data_logger.log(t1_data=t1,t2_data=t2,gps_data=gps,image_path=img)
        except Exception as e: logger.error("  CSV: %s", e)

        # AI Pipeline
        if cycle % AI_EVERY == 0 and (t1 or t2):
            try:
                l1  = float((t1 or {}).get("light_intensity",500) or 500)
                l2  = float((t2 or {}).get("light_intensity",500) or 500)
                p1  = float((t1 or {}).get("pitch",0) or 0)
                p2  = float((t2 or {}).get("pitch",0) or 0)
                d1  = float((t1 or {}).get("distance",250) or 250)/1000.0
                d2  = float((t2 or {}).get("distance",250) or 250)/1000.0
                B   = max(abs(d1-d2), 0.05)
                Bp  = B*np.cos(np.radians((p1+p2)/2))
                t_a = np.linspace(0,1,1024)
                f_f = (Bp/LAMBDA_NM)*2e-6
                buf = (l1/4095.0 + l2/4095.0
                       + 2*np.sqrt(l1/4095.0*l2/4095.0+1e-10)*0.7
                       * np.cos(2*np.pi*f_f*t_a)).tolist()
                db_log_ai(run_ai_pipeline(t1, t2, buf))
            except Exception as e:
                logger.error("  AI: %s", e)

        sleep = max(0, config.LOOP_INTERVAL_S - (time.time()-t0))
        if sleep: time.sleep(sleep)


def main():
    try:
        setup(); loop()
    except KeyboardInterrupt: pass
    finally:
        logger.info("Shutting down...")
        camera.shutdown(); gps_parser.shutdown(); data_logger.shutdown()
        if _db: _db.close()
        logger.info("Goodbye.")


if __name__ == "__main__":
    main()
