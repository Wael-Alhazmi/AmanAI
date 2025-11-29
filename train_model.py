import pandas as pd
import numpy as np
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# محاولة استيراد LightGBM و XGBoost
try:
    from xgboost import XGBClassifier
    xgboost_available = True
except:
    xgboost_available = False

try:
    import lightgbm as lgb
    lightgbm_available = True
except:
    lightgbm_available = False


# =====================================================
# 1) تحميل البيانات
# =====================================================

df = pd.read_csv("external_traffic.csv")

print("\n📥 تم تحميل external_traffic.csv")
print("عدد السجلات:", len(df))


# =====================================================
# 2) تجهيز البيانات
# =====================================================

features = ["lat", "lng", "day", "hour", "traffic_num"]
target = "risk_label"

X = df[features]
y = df[target]

# تقسيم البيانات 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n📊 تقسيم البيانات:")
print("Training:", len(X_train), " | Test:", len(X_test))


# =====================================================
# 3) قائمة النماذج المرشحة
# =====================================================

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=300, max_depth=12, random_state=42
    ),

    "GradientBoosting": GradientBoostingClassifier(),

    "LogisticRegression": LogisticRegression(max_iter=200),

    "MLP_NeuralNet": MLPClassifier(hidden_layer_sizes=(64, 32),
                                   max_iter=400, random_state=42)
}

# إضافة XGBoost إذا متوفر
if xgboost_available:
    models["XGBoost"] = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss"
    )

# إضافة LightGBM إذا متوفر
if lightgbm_available:
    models["LightGBM"] = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1
    )


# =====================================================
# 4) تدريب كل نموذج واختيار الأفضل
# =====================================================

best_model = None
best_f1 = -1
metrics_dict = {}

print("\n🔍 بدء تدريب النماذج...\n")

for name, model in models.items():
    print(f"🚀 تدريب النموذج: {name}")

    try:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")

        print(f"  → Accuracy = {acc:.4f}")
        print(f"  → Macro F1 = {f1:.4f}\n")

        metrics_dict[name] = {
            "accuracy": acc,
            "f1_macro": f1,
            "classification_report": classification_report(
                y_test, preds, output_dict=False
            )
        }

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = name

    except Exception as e:
        print(f"❌ النموذج {name} فشل: {e}\n")


print("\n========================================")
print(f"🏆 أفضل نموذج: {best_model_name}")
print(f"🔢 Macro F1: {best_f1:.4f}")
print("========================================\n")


# =====================================================
# 5) حفظ أفضل نموذج
# =====================================================

with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("💾 تم حفظ النموذج في model.pkl")


# =====================================================
# 6) حفظ المقاييس في ملف JSON
# =====================================================

with open("metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics_dict, f, indent=4, ensure_ascii=False)

print("📊 تم حفظ مؤشرات الأداء في metrics.json\n")


print("🎉 التدريب اكتمل بنجاح! النموذج جاهز للعمل في مشروع AmanAI.")
