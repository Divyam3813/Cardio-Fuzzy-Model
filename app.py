import json
import numpy as np
import streamlit as st

# ============================================================
# PAGE CONFIG & VIBRANT STYLING
# ============================================================

st.set_page_config(
    page_title="Dr. Cardio — Clinical Diagnostic Suite",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Vibrant custom CSS with gradients and glowing cards
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 100%);
    }
    .stMetric {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(236, 72, 153, 0.2);
        padding: 15px;
        border-radius: 14px;
        box-shadow: 0 8px 20px -4px rgba(236, 72, 153, 0.15);
    }
    .assistant-card {
        background: linear-gradient(135deg, rgba(236, 72, 153, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%);
        border: 1px solid rgba(236, 72, 153, 0.3);
        padding: 20px;
        border-radius: 16px;
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .stButton>button {
        background: linear-gradient(135deg, #ec4899 0%, #8b5cf6 100%);
        color: white;
        border-radius: 10px;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1rem;
        box-shadow: 0 4px 15px rgba(236, 72, 153, 0.4);
    }
    .stButton>button:hover {
        opacity: 0.9;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL FILES
# ============================================================

@st.cache_data
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

try:
    fis = load_json("heart_anfis_python_config.json")
    config = load_json("heart_anfis_config.json")
except FileNotFoundError as e:
    st.error(f"Required model file not found: {e}")
    st.stop()
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.stop()

# ============================================================
# MODEL CONFIGURATION
# ============================================================

FEATURES = config["features"]
MEAN = np.asarray(config["normalization"]["mean"], dtype=float)
STD = np.asarray(config["normalization"]["std"], dtype=float)
CLIP_MIN = float(config["normalization"]["clipMin"])
CLIP_MAX = float(config["normalization"]["clipMax"])
THRESHOLD = float(config["decision"]["threshold"])

INPUTS = fis["inputs"]
OUTPUTS = fis["outputs"]
RULES = fis["rules"]

AND_METHOD = fis.get("andMethod", "prod").lower()
OR_METHOD = fis.get("orMethod", "probor").lower()

# Gemini model used by the chatbot (change it here only)
GEMINI_MODEL = "gemini-3.8-flash"

# ============================================================
# GAUSSIAN MEMBERSHIP FUNCTION (100% ORIGINAL CALCULATION)
# ============================================================

def gaussmf(x, params):
    sigma = float(params[0])
    center = float(params[1])
    if sigma <= 0:
        return 0.0
    value = np.exp(-((x - center) ** 2) / (2.0 * sigma ** 2))
    return float(value)

def membership_value(x, mf):
    mf_type = mf["type"].lower()
    params = mf["parameters"]
    if mf_type == "gaussmf":
        return gaussmf(x, params)
    raise ValueError(f"Unsupported input membership function: {mf_type}")

def apply_and(values):
    values = np.asarray(values, dtype=float)
    if AND_METHOD == "prod":
        return float(np.prod(values))
    if AND_METHOD == "min":
        return float(np.min(values))
    raise ValueError(f"Unsupported AND method: {AND_METHOD}")

def apply_or(values):
    values = np.asarray(values, dtype=float)
    if OR_METHOD == "probor":
        return float(1.0 - np.prod(1.0 - values))
    if OR_METHOD == "max":
        return float(np.max(values))
    raise ValueError(f"Unsupported OR method: {OR_METHOD}")

def normalize_input(values):
    values = np.asarray(values, dtype=float)
    normalized = (values - MEAN) / STD
    normalized = np.clip(normalized, CLIP_MIN, CLIP_MAX)
    return normalized

def evaluate_anfis(raw_values):
    x = normalize_input(raw_values)
    rule_strengths = []
    rule_outputs = []

    for rule in RULES:
        antecedents = rule["antecedent"]
        membership_values = []

        for i, mf_index in enumerate(antecedents):
            if mf_index == 0:
                membership_values.append(1.0)
                continue

            mf_index = int(mf_index) - 1
            mf_list = INPUTS[i]["membershipFunctions"]

            if mf_index < 0 or mf_index >= len(mf_list):
                raise ValueError(f"Invalid MF index {mf_index + 1} for input {i + 1}")

            mf = mf_list[mf_index]
            value = membership_value(x[i], mf)
            membership_values.append(value)

        connection = int(rule.get("connection", 1))
        if connection == 1:
            firing = apply_and(membership_values)
        elif connection == 2:
            firing = apply_or(membership_values)
        else:
            raise ValueError(f"Unsupported rule connection: {connection}")

        weight = float(rule.get("weight", 1.0))
        firing *= weight

        consequent_index = int(rule["consequent"]) - 1
        output_mfs = OUTPUTS["membershipFunctions"]
        consequent_mf = output_mfs[consequent_index]
        consequent_type = consequent_mf["type"].lower()
        parameters = np.asarray(consequent_mf["parameters"], dtype=float)

        if consequent_type == "linear":
            if len(parameters) != len(x) + 1:
                raise ValueError("Linear consequent parameter count does not match number of inputs.")
            rule_output = float(np.dot(parameters[:-1], x) + parameters[-1])
        elif consequent_type == "constant":
            rule_output = float(parameters[0])
        else:
            raise ValueError(f"Unsupported Sugeno consequent type: {consequent_type}")

        rule_strengths.append(firing)
        rule_outputs.append(rule_output)

    rule_strengths = np.asarray(rule_strengths, dtype=float)
    rule_outputs = np.asarray(rule_outputs, dtype=float)
    denominator = np.sum(rule_strengths)

    if denominator <= 1e-12:
        prediction = float(np.mean(rule_outputs))
    else:
        prediction = float(np.sum(rule_strengths * rule_outputs) / denominator)

    prediction = float(np.clip(prediction, 0.0, 1.0))
    return prediction, x, rule_strengths

# ============================================================
# SIDEBAR CONTROLS & MODEL METRICS
# ============================================================

with st.sidebar:
    st.markdown("### 🩺 Dr. Cardio's Control Panel")
    if st.button("🗑️ Clear Chat History & Memory", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_summary = "No previous context."
        st.success("Memory cleared successfully!")
        st.rerun()

    st.divider()
    st.markdown("### 📊 ANFIS Validation")
    test_metrics = config["performance"]["test"]
    st.metric("Test Accuracy", f"{test_metrics['accuracy'] * 100:.2f}%")
    st.metric("ROC-AUC Score", f"{test_metrics['rocAUC'] * 100:.2f}%")
    st.metric("Sensitivity", f"{test_metrics['recall'] * 100:.2f}%")
    st.metric("Specificity", f"{test_metrics['specificity'] * 100:.2f}%")

    st.divider()
    st.caption("Powered by Fuzzy Logic")

# ============================================================
# HEADER WITH ASSISTANT INTRODUCTION
# ============================================================

st.markdown("""
    <div class="assistant-card">
        <h2 style='margin-bottom: 5px; color: #f472b6;'>❤️ Meet Dr. Cardio 🩺</h2>
        <p style='color: #cbd5e1; font-size: 1.05rem; margin-bottom: 0;'>
            Your intelligent clinical partner! Adjust the patient parameters below to watch the <b>Live Risk Gauge</b> update instantly in real-time. Ask Dr. Cardio anything in the chat below for wellness and lifestyle advice.
        </p>
    </div>
""", unsafe_allow_html=True)

# ============================================================
# SIMPLIFIED TYPE GUIDE EXPANDER
# ============================================================

with st.expander("📖 Beginner-Friendly Parameter & Type Guide", expanded=False):
    st.markdown("""
    * **Chest Pain Type (`cp`):** Categorizes chest discomfort:
      * `0` (Typical Angina): Classic heart-related chest pressure triggered by exertion.
      * `1` (Atypical Angina): Mild or unusual chest discomfort.
      * `2` (Non-anginal Pain): Chest pain unrelated to heart blockages.
      * `3` (Asymptomatic): No chest pain experienced at all.
    * **Maximum Heart Rate (`thalach`):** Peak heart rate achieved during physical exertion or stress testing (50–250 bpm).
    * **ST Depression (`oldpeak`):** Electrocardiographic measurement of ST-segment depression induced by exercise relative to rest (0.0–10.0).
    * **Exercise Induced Angina (`exang`):** Chest pain triggered explicitly by physical exertion (`0`: No, `1`: Yes).
    * **Major Vessels Blocked (`ca`):** Number of major coronary vessels (0–3) colored or obstructed under fluoroscopy.
    * **Nuclear Stress Scan (`thal`):** Blood flow assessment defect category:
      * `0` (Normal): Healthy blood flow.
      * `1` (Fixed Defect): Permanent damage from past events.
      * `2` (Reversible Defect): Blood flow drops during stress (sign of ischemia).
      * `3` (Other): Miscellaneous flow patterns.
    """)

# ============================================================
# INTERACTIVE SLIDER INPUTS (INSTANT EVALUATION - NO BUTTON NEEDED)
# ============================================================

st.markdown("### 🎚️ Adjust Patient Parameters (Live Evaluation)")

col1, col2, col3 = st.columns(3)

with col1:
    cp = st.selectbox(
        "Chest Pain Type (cp)",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "0: Typical Angina",
            1: "1: Atypical Angina",
            2: "2: Non-anginal Pain",
            3: "3: Asymptomatic"
        }[x]
    )

    thalach = st.slider(
        "Maximum Heart Rate (thalach)",
        min_value=50.0,
        max_value=250.0,
        value=150.0,
        step=1.0
    )

with col2:
    oldpeak = st.slider(
        "ST Depression (oldpeak)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    exang = st.selectbox(
        "Exercise Induced Angina (exang)",
        options=[0, 1],
        format_func=lambda x: {0: "0: No", 1: "1: Yes"}[x]
    )

with col3:
    ca = st.selectbox(
        "Major Vessels Blocked (ca)",
        options=[0, 1, 2, 3, 4]
    )

    thal = st.selectbox(
        "Nuclear Stress Scan (thal)",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "0: Normal",
            1: "1: Fixed Defect",
            2: "2: Reversible Defect",
            3: "3: Other"
        }[x]
    )

# ============================================================
# INSTANT ANFIS CALCULATION & RISK GAUGE DISPLAY
# ============================================================

try:
    raw_values = np.array([cp, thalach, oldpeak, exang, ca, thal], dtype=float)
    prediction, normalized, rule_strengths = evaluate_anfis(raw_values)
    risk_percentage = prediction * 100.0

    if prediction >= THRESHOLD:
        classification = "Higher Risk"
        badge_color = "#ef4444"
    else:
        classification = "Lower Risk"
        badge_color = "#10b981"

    st.markdown("---")
    st.markdown("### ⚡ Live Diagnostic Summary & Risk Gauge")

    res_c1, res_c2 = st.columns([1.2, 1])

    with res_c1:
        st.metric("Live ANFIS Risk Score", f"{risk_percentage:.2f}%")
        st.markdown(f"**Decision Threshold:** {THRESHOLD * 100:.1f}% Risk")
        st.markdown(f"**Diagnostic Triage:** <span style='color: {badge_color}; font-weight: bold;'>{classification}</span>", unsafe_allow_html=True)
        
        if prediction >= THRESHOLD:
            st.warning("⚠️ **Clinical Notice:** The evaluated profile score surpasses the classification threshold, indicating elevated risk indicators.")
        else:
            st.success("✅ **Clinical Notice:** The evaluated profile score is below the decision threshold (Standard Monitoring).")

    with res_c2:
        st.markdown("#### 🎯 Risk Gauge Meter")
        st.progress(int(round(risk_percentage)))
        st.caption(f"Current Risk Level: {risk_percentage:.1f}% out of 100%")

    # Save into session state for Dr. Cardio chat context
    st.session_state["last_risk"] = f"{risk_percentage:.2f}"
    st.session_state["last_classification"] = classification

except Exception as e:
    st.error(f"Prediction error: {e}")

# ============================================================
# DR. CARDIO CHATBOT (COLOURFUL & INTERACTIVE)
# ============================================================

st.markdown("---")
st.markdown("### 💬 Chat with Dr. Cardio 🩺")
st.markdown("Ask Dr. Cardio for heart-healthy tips, workout adjustments, or dietary suggestions tailored to your live risk score.")

# Securely load API key from Streamlit secrets or fallback
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    try:
        from config_keys import GEMINI_API_KEY
    except ImportError:
        GEMINI_API_KEY = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_summary" not in st.session_state:
    st.session_state.chat_summary = "No prior interaction."

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask Dr. Cardio about diet, exercise, or heart health..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not GEMINI_API_KEY or GEMINI_API_KEY == "your_actual_gemini_api_key_here":
            st.error("Gemini API key not found. Please add `GEMINI_API_KEY` in your Streamlit secrets.")
        else:
            try:
                from google import genai
                from google.genai import types
                client = genai.Client(api_key=GEMINI_API_KEY)

                risk_context = st.session_state.get("last_risk", "not yet evaluated")
                class_context = st.session_state.get("last_classification", "Unknown")

                system_instruction = (
                    f"You are Dr. Cardio, a friendly, empathetic, and knowledgeable cardiovascular health and wellness expert. "
                    f"The user's recent model evaluation yielded a live risk score of {risk_context}% ({class_context}). "
                    f"Provide actionable, encouraging lifestyle advice regarding diet, exercise, sleep, and stress management. "
                    f"Keep every reply concise: under 150 words, a few short points at most. "
                    f"End with one short reminder that this is for educational purposes only, it can make mistakes, and they should consult a real doctor for proper diagnosis."
                )

                # Only the last 8 messages are sent, so input size stays small
                contents = [f"{m['role']}: {m['content']}" for m in st.session_state.messages[-8:]]

                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        max_output_tokens=1000,
                    ),
                )

                ai_response = response.text
                st.markdown(ai_response)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})

            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    st.error("Dr. Cardio has hit the API usage limit for now. Please try again later.")
                else:
                    st.error(f"Error communicating with Gemini API: {e}")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94a3b8; font-size: 0.9rem;'>"
    "For educational purpose only, it can make mistakes. For proper diagnosis consult a doctor."
    "</div>",
    unsafe_allow_html=True
)
