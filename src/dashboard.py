# dashboard.py
# ─────────────────────────────────────────────────────────────────────────────
# Streamlit dashboard for the clinical lab predictor.
# Run with: streamlit run src/dashboard.py
#
# Five pages:
#   1. Predict — enter lab values, get a prediction + SHAP explanation
#   2. Performance — test set metrics and feature importance
#   3. Fairness — bias audit across age groups
#   4. Governance — model card, audit log, version registry
#   5. Monitoring — live prediction stats and drift indicators
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(
    page_title="Clinical Lab Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load everything once ──────────────────────────────────────────────────────

@st.cache_resource
def load_model_and_scaler():
    try:
        model  = joblib.load("models/rf_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        return None, None


@st.cache_resource
def load_shap_explainer(model):
    try:
        import shap
        return shap.TreeExplainer(model)
    except Exception:
        return None


@st.cache_data
def load_json_file(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def prep_features(vals: dict) -> pd.DataFrame:
    df = pd.DataFrame([vals])
    df["GlucoseInsulinRatio"] = df["Glucose"] / (df["Insulin"] + 1)
    df["BMICategory"] = int(pd.cut([vals["BMI"]], bins=[0,18.5,25,30,100], labels=[0,1,2,3])[0])
    df["AgeGroup"]    = int(pd.cut([vals["Age"]], bins=[0,30,45,60,120],   labels=[0,1,2,3])[0])
    return df


# ── Sidebar navigation ────────────────────────────────────────────────────────

model, scaler = load_model_and_scaler()
explainer     = load_shap_explainer(model) if model else None

st.sidebar.title("🧬 Clinical Lab Predictor")
st.sidebar.caption("Research prototype — not for clinical use")
page = st.sidebar.radio("", ["🔬 Predict", "📊 Performance", "⚖️ Fairness", "📋 Governance", "📈 Monitoring"])

if model is None:
    st.error("Model files not found. Run the training pipeline first: `python src/data_curation.py` → ... → `python src/governance.py`")
    st.stop()

# ── PAGE: Predict ─────────────────────────────────────────────────────────────
if page == "🔬 Predict":
    st.title("Diabetes Risk Prediction")
    st.info("Enter patient lab values. All fields are required. Normal ranges shown for reference.")

    col1, col2 = st.columns(2)

    with col1:
        glucose = st.slider("Glucose (mg/dL)",  44, 400, 120, help="Normal fasting: 70–100 mg/dL")
        insulin = st.slider("Insulin (μU/mL)",   0, 900,  80, help="Normal fasting: 2–25 μU/mL")
        bmi     = st.slider("BMI (kg/m²)",      10.0, 70.0, 28.0, step=0.1)
        bp      = st.slider("Diastolic BP (mm Hg)", 20, 200, 70, help="Normal: 60–80 mm Hg")

    with col2:
        age         = st.slider("Age (years)",        18, 100, 35)
        pregnancies = st.slider("Pregnancies",         0,  20,  1)
        skin        = st.slider("Skin Thickness (mm)", 0, 100, 25)
        dpf         = st.slider("Diabetes Pedigree",  0.0, 3.0, 0.5, step=0.01)

    st.markdown("---")
    if st.button("Run Prediction", type="primary", use_container_width=True):
        vals = {
            "Pregnancies": pregnancies, "Glucose": glucose,
            "BloodPressure": bp, "SkinThickness": skin,
            "Insulin": insulin, "BMI": bmi,
            "DiabetesPedigreeFunction": dpf, "Age": age,
        }
        X        = prep_features(vals)
        X_scaled = scaler.transform(X)
        prob     = float(model.predict_proba(X_scaled)[0][1])
        pred     = int(prob >= 0.5)

        # result banner
        r1, r2, r3 = st.columns(3)
        risk = "HIGH" if prob > 0.6 else "MODERATE" if prob > 0.3 else "LOW"
        if pred == 1:
            r1.error(f"⚠️ POSITIVE\n\nProbability: {prob:.1%}")
        else:
            r1.success(f"✅ NEGATIVE\n\nProbability: {prob:.1%}")
        r2.metric("Risk Level", risk)
        r3.metric("Confidence", f"{max(prob, 1 - prob):.1%}")

        # SHAP explanation
        if explainer is not None:
            st.subheader("What drove this prediction?")
            shap_vals = explainer.shap_values(X_scaled)
            vals_arr  = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
            feat_names = X.columns.tolist()

            shap_df = pd.DataFrame({"Feature": feat_names, "Value": vals_arr}) \
                        .sort_values("Value", key=abs, ascending=True)

            fig, ax = plt.subplots(figsize=(9, 5))
            colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in shap_df["Value"]]
            ax.barh(shap_df["Feature"], shap_df["Value"], color=colors, edgecolor="none")
            ax.axvline(0, color="#333", linewidth=0.8)
            ax.set_xlabel("SHAP value (positive = increases risk)")
            ax.set_title("Feature contributions to this prediction")
            red_p  = mpatches.Patch(color="#e74c3c", label="Increases risk")
            green_p = mpatches.Patch(color="#2ecc71", label="Decreases risk")
            ax.legend(handles=[red_p, green_p], loc="lower right")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("SHAP explanations not available (install the `shap` package to enable them).")

        st.caption("⚠️ Research prototype. Any clinical interpretation requires physician review.")

# ── PAGE: Performance ─────────────────────────────────────────────────────────
elif page == "📊 Performance":
    st.title("Model Performance")
    report = load_json_file("docs/eval_report.json")
    meta   = load_json_file("models/rf_metadata.json")

    if not report:
        st.warning("Run `python src/evaluate.py` first.")
        st.stop()

    m = report.get("overall_metrics", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUC-ROC",      m.get("auc_roc"))
    c2.metric("Accuracy",     m.get("accuracy"))
    c3.metric("Sensitivity",  m.get("sensitivity"), help="Rate of correctly identifying diabetic patients")
    c4.metric("FNR",          m.get("false_negative_rate"), help="Rate of missed diagnoses — lower is better")

    st.subheader("Confusion Matrix")
    cm = m.get("confusion_matrix", [[0,0],[0,0]])
    cm_df = pd.DataFrame(cm, index=["Actual: No", "Actual: Yes"], columns=["Pred: No", "Pred: Yes"])
    st.dataframe(cm_df, use_container_width=True)

    if "feature_importance" in meta:
        st.subheader("Feature Importance (Random Forest)")
        fi   = meta["feature_importance"]
        fi_df = pd.DataFrame(fi.items(), columns=["Feature", "Importance"]).sort_values("Importance")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(fi_df["Feature"], fi_df["Importance"], color="#3498db", edgecolor="none")
        ax.set_xlabel("Mean decrease in impurity")
        plt.tight_layout()
        st.pyplot(fig)

# ── PAGE: Fairness ────────────────────────────────────────────────────────────
elif page == "⚖️ Fairness":
    st.title("Bias & Fairness Audit")
    st.write("Performance broken down by patient age group. A large AUC gap between groups suggests the model may be less reliable for some patients.")

    report = load_json_file("docs/eval_report.json")
    if not report:
        st.warning("Run `python src/evaluate.py` first.")
        st.stop()

    fairness = report.get("fairness_audit", {})
    summary  = fairness.pop("_summary", {})

    if summary:
        flag = summary.get("auc_flag", "")
        if flag == "REVIEW":
            st.warning(f"⚠️ AUC gap = {summary['auc_gap']} — review recommended")
        else:
            st.success(f"✅ AUC gap = {summary['auc_gap']} — within acceptable range")

    if fairness:
        df = pd.DataFrame(fairness).T
        st.dataframe(df, use_container_width=True)

# ── PAGE: Governance ──────────────────────────────────────────────────────────
elif page == "📋 Governance":
    st.title("Model Governance")
    tab1, tab2, tab3 = st.tabs(["Model Card", "Audit Log", "Version Registry"])

    with tab1:
        if os.path.exists("docs/model_card.md"):
            st.markdown(open("docs/model_card.md").read())
        else:
            st.info("Run `python src/governance.py` to generate the model card.")

    with tab2:
        audit = load_json_file("docs/audit_log.json")
        if audit:
            for entry in reversed(audit):
                with st.expander(f"{entry['timestamp'][:19]}  |  {entry['event']}"):
                    st.json(entry)
        else:
            st.info("No audit log yet.")

    with tab3:
        registry = load_json_file("docs/model_registry.json")
        if registry:
            for v in reversed(registry):
                with st.expander(f"v{v['version']}  —  {v['created_at'][:10]}"):
                    st.json(v)
        else:
            st.info("No versions registered yet.")

# ── PAGE: Monitoring ──────────────────────────────────────────────────────────
elif page == "📈 Monitoring":
    st.title("Prediction Monitoring")
    log_path = "monitoring/predictions.jsonl"

    if not os.path.exists(log_path):
        st.info("No predictions logged yet. Make some on the Predict page first.")
        st.stop()

    rows  = [json.loads(l) for l in open(log_path)]
    probs = [r["probability"] for r in rows]
    preds = [r["prediction"]  for r in rows]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Predictions", len(rows))
    c2.metric("Positive Rate",     f"{sum(preds)/len(preds):.1%}")
    c3.metric("Mean Probability",  f"{sum(probs)/len(probs):.3f}")

    st.subheader("Probability Distribution")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.hist(probs, bins=25, color="#3498db", edgecolor="white", alpha=0.85)
    ax.axvline(0.5, color="red", linestyle="--", label="Decision threshold (0.5)")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Recent Predictions (last 20)")
    recent = pd.DataFrame([{
        "Time":        r["ts"][:19],
        "Prediction":  "Positive" if r["prediction"] else "Negative",
        "Probability": f"{r['probability']:.3f}",
    } for r in rows[-20:]])
    st.dataframe(recent, use_container_width=True)
