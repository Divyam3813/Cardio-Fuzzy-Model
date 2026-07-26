# 🫀 CardioFuzzy AI — Clinical ANFIS Diagnostic System

## 🚀 Live Demo

🔗 [Launch CardioFuzzy AI](https://cardio-fuzzy.streamlit.app/) 

## 📖 Project Overview

**CardioFuzzy AI** is an advanced, explainable clinical decision-support platform engineered for cardiovascular disease risk assessment. Traditional deep learning models in healthcare often suffer from the "black-box" dilemma—making high-stakes predictions without revealing *why* a diagnosis was reached.

This platform bridges that gap by implementing an **Adaptive Neuro-Fuzzy Inference System (ANFIS)**. By hybridizing the linguistic transparency of fuzzy logic with the automated optimization of neural networks, CardioFuzzy AI provides high diagnostic accuracy alongside full visual and rule-based auditability.

---

## 🏗️ Technical Architecture: Why ANFIS?

ANFIS merges fuzzy inference systems (FIS) with neural network learning algorithms.

1. **Fuzzy Layer (Membership Functions):** Numerical clinical inputs (such as maximum heart rate or ST depression) are mapped onto linguistic fuzzy sets (e.g., *Low*, *Moderate*, *High*) using continuous membership functions.
2. **Rule Base Layer:** Evaluates a set of linguistic `IF-THEN` rules derived from clinical heuristics and trained parameter tuning.
3. **Neural Optimization:** Backpropagation and least-squares estimation methods optimize the premise and consequent parameters of the fuzzy system to minimize training error.

---

## 🗂️ Repository File Structure

```text
📁 cardiofuzzy-app/
│── 📄 app.py                  # Core Streamlit dashboard UI, layout engine, and live risk calculator
│── 📄 othermodels.py          # Comparative benchmarking suite (ANN, Random Forest, Logistic Regression)
│── 📄 heart_anfis.fis         # Exported MATLAB Fuzzy Inference System (FIS) configuration structure
│── 📄 heart.csv               # Processed UCI Heart Disease clinical validation dataset
│── 📄 requirements.txt        # Python package dependencies manifest
└── 📄 README.md               # Complete technical documentation (this file)
```

---

## 📊 Performance Benchmarks & Validation

Evaluated on unseen clinical test splits, the ANFIS architecture outperforms rigid black-box models, striking an optimal balance between precision and safety-critical medical recall:

| Diagnostic Architecture       | Accuracy (%) | Sensitivity / Recall (%) | Interpretability Index                  |
|--------------------------------|:------------:|:-------------------------:|-------------------------------------------|
| **ANFIS (Proposed Model)**    | 86.67%       | 90.00%                    | 🟢 100% White-Box (Extractable Rules)      |
| Random Forest                  | 78.69%       | 84.85%                    | 🟡 30% (Ensemble Feature Importance)       |
| Logistic Regression             | 77.05%       | 75.76%                    | 🟢 90% (Linear Weights)                    |
| Deep Neural Net (ANN)          | 72.10%       | 81.82%                    | 🔴 0% (Opaque Hidden Layers)               |

---

## 🎛️ Key Dashboard Features

### Interactive Clinical Parameter Triage
Real-time input controls for key diagnostic indicators:

- **Chest Pain Type (`cp`):** Typical angina, atypical angina, non-anginal pain, or asymptomatic.
- **Max Heart Rate (`thalach`):** Continuous physiological tracking.
- **ST Depression (`oldpeak`):** Exercise-induced ECG changes.
- **Exercise Angina (`exang`):** Binary clinical indicator.
- **Major Vessels (`ca`):** Fluoroscopy vessel blockages (0–4).
- **Nuclear Stress Scan (`thal`):** Normal, fixed defect, or reversible defect.

### Visual Analytics Suite
- Dynamic risk gauge indicator with strict clinical triage cutoffs.
- Multi-axis feature contribution radar profile comparing patient vitals against baseline thresholds.
- **Fuzzy Rule Auditor:** Expandable rule inspection panel allowing cardiologists to audit specific rules triggered during evaluation.

---

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
