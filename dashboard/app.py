#!/usr/bin/env python3
"""
Dashboard — PFE TUNSA  (port 5001)
Proxy vers la station API (port 5000) + rendu HTML.
"""
import logging, requests
from flask import Flask, jsonify, render_template

STATION = "http://127.0.0.1:5000"
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [dashboard] %(levelname)s %(message)s")
logger = logging.getLogger("dashboard")

app = Flask(__name__)
app.logger.setLevel(logging.WARNING)

def _get(path, fallback=None):
    try:
        r = requests.get(f"{STATION}{path}", timeout=2)
        return r.json()
    except Exception as e:
        logger.warning("Station unreachable (%s): %s", path, e)
        return fallback

@app.route("/")
def index():
    return render_template("index.html", station_url=STATION)

# ── Proxy endpoints ──────────────────────────────────────────

@app.route("/api/status")
def status():
    return jsonify(_get("/api/status", {"status":"offline"}))

@app.route("/api/telemetry/latest")
def tel_latest():
    return jsonify(_get("/api/telemetry/latest", {"T1":None,"T2":None}))

@app.route("/api/telemetry/history")
def tel_history():
    from flask import request
    node  = request.args.get("node","")
    limit = request.args.get("limit","60")
    path  = f"/api/telemetry/history?limit={limit}"
    if node: path += f"&node={node}"
    return jsonify(_get(path, {"count":0,"data":[]}))

@app.route("/api/ai/latest")
def ai_latest():
    return jsonify(_get("/api/ai/latest", {}))

@app.route("/api/ai/history")
def ai_history():
    from flask import request
    limit = request.args.get("limit","40")
    return jsonify(_get(f"/api/ai/history?limit={limit}", {"count":0,"data":[]}))

@app.route("/api/sky")
def sky():
    return jsonify(_get("/api/sky", {"sky_image_b64":None}))

if __name__ == "__main__":
    logger.info("Dashboard → %s  (port 5001)", STATION)
    app.run(host="0.0.0.0", port=5001, use_reloader=False)
