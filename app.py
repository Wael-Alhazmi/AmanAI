import sqlite3
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import joblib
import random

# ============================
# تحميل النموذج
# ============================
model = joblib.load("model.pkl")
RISK_MAP = {0: "منخفض", 1: "متوسط", 2: "مرتفع"}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# إنشاء قاعدة البيانات
# ============================
def init_db():
    conn = sqlite3.connect("incidents.db")
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT,
            lat REAL,
            lng REAL,
            day INTEGER,
            hour INTEGER,
            traffic TEXT,
            predicted_risk TEXT,
            observed_risk TEXT,
            incident_type TEXT,
            recommendation TEXT,
            source TEXT
        )
    """
    )
    conn.commit()
    conn.close()


init_db()

# ============================
# نماذج البيانات
# ============================


class PredictInput(BaseModel):
    lat: float
    lng: float
    day: int
    hour: int
    traffic: int


class IncidentManual(BaseModel):
    incident_type: str
    observed_risk: str
    recommendation: str
    lat: float
    lng: float


class DeleteInput(BaseModel):
    id: int


# ============================
# دوال مساعدة
# ============================


def risk_to_traffic_num(observed: str) -> int:
    m = {"منخفض": 0, "متوسط": 1, "مرتفع": 2}
    return m.get(observed, 1)


def make_recommendation(risk: str) -> str:
    if risk == "مرتفع":
        return "🚨 نوصي بإرسال دوريات فورًا ومتابعة الموقع بدقة."
    if risk == "متوسط":
        return "⚠️ متابعة الموقع خلال 10 دقائق والاستعداد للتصعيد."
    return "✓ الوضع مستقر ولا يتطلب إجراء فوري."


# ============================
# 1) API التنبؤ (يُستخدم داخليًا فقط)
# ============================


@app.post("/predict")
def predict(data: PredictInput):
    X = [[data.lat, data.lng, data.day, data.hour, data.traffic]]
    pred = int(model.predict(X)[0])
    proba = float(max(model.predict_proba(X)[0]))
    risk = RISK_MAP.get(pred, "غير محدد")

    return {
        "prediction": risk,
        "confidence": round(proba * 100, 2),
        "recommendation": make_recommendation(risk),
    }


# ============================
# 2) حفظ بلاغ يدوي (مع استخدام AI في الخلفية)
# ============================


@app.post("/save-incident")
def save_manual(data: IncidentManual):
    now = datetime.now()
    day = now.weekday()
    hour = now.hour
    traffic_num = risk_to_traffic_num(data.observed_risk)
    traffic_label = data.observed_risk  # نخزنها نصًا

    # تنبؤ AI بالخطر المتوقَّع
    X = [[data.lat, data.lng, day, hour, traffic_num]]
    pred = int(model.predict(X)[0])
    predicted_risk = RISK_MAP.get(pred, "غير محدد")

    conn = sqlite3.connect("incidents.db")
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO incidents
        (time, lat, lng, day, hour, traffic,
         predicted_risk, observed_risk, incident_type,
         recommendation, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            now.strftime("%Y-%m-%d %H:%M:%S"),
            data.lat,
            data.lng,
            day,
            hour,
            traffic_label,
            predicted_risk,
            data.observed_risk,
            data.incident_type,
            data.recommendation,
            "Manual",
        ),
    )
    conn.commit()
    conn.close()

    return {"status": "saved"}


# ============================
# 3) قائمة البلاغات
# ============================


@app.get("/incidents")
def get_incidents():
    conn = sqlite3.connect("incidents.db")
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM incidents ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ============================
# 4) حذف بلاغ واحد
# ============================


@app.post("/delete-incident")
def delete_incident(data: DeleteInput):
    conn = sqlite3.connect("incidents.db")
    conn.execute("DELETE FROM incidents WHERE id = ?", (data.id,))
    conn.commit()
    conn.close()
    return {"status": "deleted"}


# ============================
# 5) مسح جميع البلاغات
# ============================


@app.post("/clear-incidents")
def clear_incidents():
    conn = sqlite3.connect("incidents.db")
    conn.execute("DELETE FROM incidents")
    conn.commit()
    conn.close()
    return {"status": "cleared"}


# ============================
# 6) إحصائيات الـ Dashboard
# ============================


@app.get("/dashboard-stats")
def dashboard_stats():
    conn = sqlite3.connect("incidents.db")
    c = conn.cursor()

    total = c.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
    high = c.execute(
        "SELECT COUNT(*) FROM incidents WHERE predicted_risk = 'مرتفع'"
    ).fetchone()[0]
    this_hour = datetime.now().hour
    last_hour = c.execute(
        "SELECT COUNT(*) FROM incidents WHERE hour = ?", (this_hour,)
    ).fetchone()[0]

    pct = round((high / total * 100), 1) if total > 0 else 0.0
    conn.close()

    return {"total": total, "high": high, "last_hour": last_hour, "high_pct": pct}


# ============================
# 7) Heatmap من البلاغات
# ============================


@app.get("/heatmap")
def heatmap():
    conn = sqlite3.connect("incidents.db")
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT lat, lng, predicted_risk FROM incidents").fetchall()
    conn.close()

    points = []
    for r in rows:
        if r["lat"] is None or r["lng"] is None:
            continue
        if r["predicted_risk"] == "مرتفع":
            w = 3
        elif r["predicted_risk"] == "متوسط":
            w = 2
        else:
            w = 1
        points.append({"lat": r["lat"], "lng": r["lng"], "weight": w})

    return {"points": points}


# ============================
# 8) طبقة المرور (Hotspots)
# ============================


@app.get("/traffic-hotspots")
def traffic_hotspots():
    conn = sqlite3.connect("incidents.db")
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT lat, lng, predicted_risk FROM incidents"
    ).fetchall()
    conn.close()

    if not rows:
        # نقاط افتراضية حول الحرم
        base = (24.47, 39.61)
        return [
            {"lat": base[0] + 0.01, "lng": base[1], "level": "منخفض"},
            {"lat": base[0], "lng": base[1] + 0.01, "level": "متوسط"},
            {"lat": base[0] - 0.01, "lng": base[1] - 0.01, "level": "مرتفع"},
        ]

    # تجميع بسيط حسب الإحداثيات (تقريب لأربع منازل عشرية)
    buckets = {}
    for r in rows:
        key = (round(r["lat"], 4), round(r["lng"], 4))
        buckets.setdefault(key, {"high": 0, "med": 0, "low": 0})
        if r["predicted_risk"] == "مرتفع":
            buckets[key]["high"] += 1
        elif r["predicted_risk"] == "متوسط":
            buckets[key]["med"] += 1
        else:
            buckets[key]["low"] += 1

    result = []
    for (lat, lng), counts in buckets.items():
        if counts["high"] > 0:
            level = "مرتفع"
        elif counts["med"] > 0:
            level = "متوسط"
        else:
            level = "منخفض"
        result.append({"lat": lat, "lng": lng, "level": level})

    return result


# ============================
# 9) تحليل الازدحام التلقائي (يضيف بلاغات AI)
# ============================


@app.get("/detect-traffic")
def detect_traffic():
    """
    الزر 🔥 تحليل الازدحام:
    - ينشئ نقاط عشوائية حول المدينة
    - يمررها للنموذج
    - يسجل البلاغات في قاعدة البيانات كمصدر AI
    """
    base_lat, base_lng = 24.47, 39.61
    now = datetime.now()
    day = now.weekday()
    hour = now.hour

    conn = sqlite3.connect("incidents.db")
    c = conn.cursor()

    for _ in range(10):
        lat = base_lat + random.uniform(-0.03, 0.03)
        lng = base_lng + random.uniform(-0.03, 0.03)
        traffic_num = random.choice([0, 1, 2])
        traffic_label = ["منخفض", "متوسط", "مرتفع"][traffic_num]

        X = [[lat, lng, day, hour, traffic_num]]
        pred = int(model.predict(X)[0])
        risk = RISK_MAP.get(pred, "غير محدد")
        rec = make_recommendation(risk)

        c.execute(
            """
            INSERT INTO incidents
            (time, lat, lng, day, hour, traffic,
             predicted_risk, observed_risk, incident_type,
             recommendation, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                now.strftime("%Y-%m-%d %H:%M:%S"),
                lat,
                lng,
                day,
                hour,
                traffic_label,
                risk,
                risk,  # observed نفس المتوقَّع في الحالات التلقائية
                "تحليل تلقائي للازدحام",
                rec,
                "AI",
            ),
        )

    conn.commit()
    conn.close()

    return {"msg": "تم تحليل الازدحام وإضافة بلاغات متوقعة جديدة."}


# ============================
# 10) تمركز الدوريات
# ============================


@app.get("/patrol-forecast")
def patrol_forecast():
    """
    يعيد أفضل 3 مواقع مرشحة لتمركز الدوريات
    بناءً على البلاغات ذات الخطورة العالية.
    """
    conn = sqlite3.connect("incidents.db")
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT lat, lng FROM incidents WHERE predicted_risk='مرتفع'"
    ).fetchall()
    conn.close()

    if not rows:
        # مواقع افتراضية
        return [
            {"lat": 24.47, "lng": 39.61},
            {"lat": 24.48, "lng": 39.60},
            {"lat": 24.46, "lng": 39.62},
        ]

    # نختار 3 نقاط مميزة (أو أقل إن لم تتوفر)
    unique = []
    seen = set()
    for r in rows:
        key = (round(r["lat"], 4), round(r["lng"], 4))
        if key in seen:
            continue
        seen.add(key)
        unique.append({"lat": key[0], "lng": key[1]})
        if len(unique) >= 3:
            break

    return unique


# ============================
# 11) تصدير PDF (نسخة بسيطة مؤقتًا)
# ============================


@app.get("/export-pdf")
def export_pdf():
    # حالياً نرجع HTML بسيط؛ لاحقاً ممكن نستخدم مكتبة لتوليد PDF حقيقي
    html = """
    <html lang="ar" dir="rtl">
    <head><meta charset="utf-8"><title>تقرير AmanAI</title></head>
    <body>
    <h2>تقرير AmanAI</h2>
    <p>سيتم لاحقاً إضافة تقرير PDF تفصيلي هنا.</p>
    </body>
    </html>
    """
    return html
