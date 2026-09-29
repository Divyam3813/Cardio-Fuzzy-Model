# 🫀 CardioFuzzy AI — ANFIS-Based Heart Disease Risk Assessment System

<p align="center">

**An Explainable Neuro-Fuzzy AI System for Continuous Cardiovascular Risk Assessment**

🔗 **Live Demo:** [CardioFuzzy AI](https://cardio-fuzzy.streamlit.app/)

</p>

---

## 📌 Overview

**CardioFuzzy AI** is an interactive **Adaptive Neuro-Fuzzy Inference System (ANFIS)** application designed to demonstrate explainable heart disease risk assessment using clinical features.

Unlike conventional machine-learning classifiers that primarily return a discrete class such as *Disease / No Disease*, CardioFuzzy AI uses a **Sugeno-type fuzzy inference system** to generate a **continuous risk score from 0% to 100%**.

The system combines:

* 🧠 **ANFIS** for neuro-fuzzy prediction
* 🔬 **Fuzzy logic** for interpretable rule-based reasoning
* 📊 **Continuous 0–100% risk scoring**
* 📈 **Interactive visual analytics**
* 🧾 **Fuzzy rule inspection**
* 🤖 **Google Gemini-powered health assistant**
* 🌐 **Multilingual conversational support**
* 🖥️ **Streamlit interactive interface**

> ⚠️ **Important:** CardioFuzzy AI is an educational/research prototype and is **not a medical device, diagnostic tool, or substitute for professional medical advice**. Its risk score should not be used to make clinical decisions.

---

# 🚀 Live Demo

### 🔗 [Launch CardioFuzzy AI](https://cardio-fuzzy.streamlit.app/)

The application provides an interactive interface where users can enter cardiovascular parameters and observe the resulting ANFIS risk score.

---

# 🧠 Why ANFIS?

**Adaptive Neuro-Fuzzy Inference System (ANFIS)** combines the learning capability of neural networks with the interpretability of fuzzy logic.

Traditional machine-learning models can produce highly accurate predictions while providing limited insight into the reasoning process.

ANFIS instead represents the prediction process through fuzzy rules such as:

```text
IF chest pain is high
AND maximum heart rate is low
AND ST depression is high
THEN cardiovascular risk is high
```

The parameters of the fuzzy system are optimized during training, allowing the model to learn nonlinear relationships from the dataset.

---

# 🏗️ System Architecture

The CardioFuzzy AI pipeline can be summarized as:

```text
                    ┌──────────────────────┐
                    │   Clinical Inputs    │
                    │                      │
                    │ cp                   │
                    │ thalach              │
                    │ oldpeak              │
                    │ exang                │
                    │ ca                   │
                    │ thal                 │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Data Preprocessing   │
                    │ & Normalization      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Membership Functions │
                    │   Fuzzification      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Fuzzy Rule Base    │
                    │      IF → THEN       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   ANFIS Inference    │
                    │  Sugeno FIS Model    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Continuous Risk Score │
                    │       0–100%         │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
             Risk Visualization     Rule Explanation
```

---

# 🔬 ANFIS Pipeline

The system consists of several major stages.

## 1. Input Layer

The application accepts six cardiovascular features:

| Feature   | Description                                |
| --------- | ------------------------------------------ |
| `cp`      | Chest pain type                            |
| `thalach` | Maximum heart rate achieved                |
| `oldpeak` | ST depression induced by exercise          |
| `exang`   | Exercise-induced angina                    |
| `ca`      | Number of major vessels                    |
| `thal`    | Thalassemia / nuclear stress-test category |

---

## 2. Preprocessing

The training pipeline performs preprocessing before ANFIS training.

The model uses training-set statistics for normalization so that the preprocessing parameters are derived from the training data rather than from the complete dataset.

This helps reduce the possibility of **data leakage** during model development.

---

## 3. Fuzzification

Numerical inputs are transformed into fuzzy membership values.

For example:

```text
Heart Rate
      │
      ├── Low
      ├── Medium
      └── High
```

Instead of treating a value as belonging exclusively to one category, fuzzy logic allows partial membership across multiple linguistic regions.

This makes the system suitable for representing gradual transitions between clinical feature ranges.

---

## 4. Fuzzy Rule Evaluation

The fuzzy inference system evaluates learned rules based on the membership values of the input features.

Conceptually:

```text
IF Feature A is High
AND Feature B is Low
AND Feature C is Moderate
THEN Risk is High
```

Multiple rules can be activated simultaneously.

---

## 5. ANFIS Optimization

During training, ANFIS adjusts its parameters to reduce prediction error.

The system combines:

* Fuzzy inference
* Neural-network-style parameter optimization
* Membership-function tuning
* Consequent parameter estimation

The resulting FIS is exported from MATLAB and used by the Streamlit application.

---

# 📊 Continuous Risk Score

One of the main goals of CardioFuzzy AI is to produce a **continuous risk score between 0% and 100%** rather than simply returning a binary class.

For example:

```text
Risk Score
───────────────
  0%   ──────────────── 100%
       ↑
   Model Output
```

Possible outputs may therefore look like:

```text
23.47%
51.82%
76.35%
91.04%
```

The displayed percentage represents the **model's continuous output**, not a clinically validated probability of developing heart disease.

> **Important distinction:** A 75% model output should not automatically be interpreted as "there is a 75% medical probability of heart disease." The model output is a risk score generated by the trained ANFIS system.

---

# 🎛️ Interactive Dashboard

The Streamlit interface provides interactive controls for the model inputs.

### 🩺 Clinical Parameters

### Chest Pain Type — `cp`

Represents the type of chest pain reported by the individual.

### Maximum Heart Rate — `thalach`

Represents the maximum heart rate achieved during testing.

### ST Depression — `oldpeak`

Represents exercise-induced ST-segment depression.

### Exercise Angina — `exang`

Binary indicator representing exercise-induced angina.

### Major Vessels — `ca`

Represents the number of major vessels identified through fluoroscopy.

### Thalassemia — `thal`

Represents the thalassemia-related test category used by the dataset.

---

# 📈 Visualization

CardioFuzzy AI provides visual feedback to make the model output easier to understand.

Depending on the application configuration, the dashboard can display:

* 🎯 Continuous risk gauge
* 📊 Risk percentage
* 🕸️ Feature/risk profile visualization
* 🧩 Fuzzy rule information
* 📋 Input summary
* 💬 AI-generated explanations

These visualizations are intended to improve **model interpretability**, not to replace clinical interpretation.

---

# 🔍 Fuzzy Rule Auditor

A key feature of CardioFuzzy AI is the ability to inspect the fuzzy inference process.

The rule auditor is intended to provide visibility into the fuzzy rules that contribute to the model output.

This helps demonstrate the fundamental advantage of fuzzy systems:

> **The prediction process can be represented using human-readable rules rather than being treated entirely as a black box.**

---

# 🤖 Gemini AI Health Assistant

CardioFuzzy AI also includes an optional conversational assistant powered by the **Google Gemini API**.

### Features

* 🌐 Multilingual conversations
* 💬 Natural-language interaction
* 🧠 Explanations of cardiovascular concepts
* 📄 Assistance with understanding model outputs
* 🩺 General educational information about cardiovascular risk factors
* 🔄 Interactive follow-up questions

The chatbot is designed as an **educational assistant** and should not be interpreted as a physician or clinical diagnostic system.

### 🔐 API Key Security

The Gemini API key should **never be hard-coded into the source code or committed to GitHub**.

For Streamlit deployment, configure the API key using **Streamlit Secrets**.

Example:

```toml
GEMINI_API_KEY = "your-api-key"
```

Do not expose the actual API key in:

* `app.py`
* GitHub repositories
* README files
* screenshots
* public notebooks
* client-side code

---

# 🗂️ Project Structure

```text
cardiofuzzy-app/
│
├── 📄 app.py
│   └── Streamlit application, UI, ANFIS inference
│       and Gemini assistant integration
│
├── 📄 othermodels.py
│   └── Comparative machine-learning models
│       such as ANN, Random Forest and Logistic Regression
│
├── 📄 heart_anfis.fis
│   └── Exported MATLAB ANFIS/FIS model
│
├── 📄 heart_anfis_config.json
│   └── Model configuration and preprocessing information
│
├── 📄 heart.csv
│   └── Dataset used for experimentation/evaluation
│
├── 📄 requirements.txt
│   └── Python dependencies
│
└── 📄 README.md
    └── Project documentation
```

> Keep the repository structure synchronized with the files actually committed to GitHub. If a file is not included in the repository, remove it from this section.

---

# 🧪 Model Development

The ANFIS model was developed using **MATLAB Fuzzy Logic Toolbox** and subsequently integrated into the Python/Streamlit application.

The general workflow is:

```text
Dataset
   ↓
Data Cleaning
   ↓
Feature Selection
   ↓
Train / Validation / Test Split
   ↓
Training-Only Normalization
   ↓
Initial FIS Generation
   ↓
ANFIS Training
   ↓
Validation
   ↓
Model Selection
   ↓
Untouched Test Evaluation
   ↓
Export FIS
   ↓
Streamlit Integration
```

This separation is particularly important because the test set should remain isolated until the final evaluation.

---

# 🧮 Model Input Features

The current ANFIS model uses the following feature ordering:

```python
[
    "cp",
    "thalach",
    "oldpeak",
    "exang",
    "ca",
    "thal"
]
```

⚠️ **Feature ordering is critical.**

The order used during Streamlit inference must exactly match the order used during model training and FIS generation.

Changing the order can produce incorrect predictions even if the individual values are valid.

---

# 🔄 MATLAB → Streamlit Integration

The project uses MATLAB for ANFIS model development and Python/Streamlit for deployment.

```text
              MATLAB
                 │
                 ▼
        ┌─────────────────┐
        │ ANFIS Training  │
        └────────┬────────┘
                 │
                 ▼
          heart_anfis.fis
                 │
                 ▼
        ┌─────────────────┐
        │ Python/Streamlit│
        │     Backend     │
        └────────┬────────┘
                 │
                 ▼
        Interactive Web App
```

This architecture allows the model to be trained using MATLAB's fuzzy-logic ecosystem while providing a modern web interface through Streamlit.

---

# 🛠️ Technologies Used

| Technology             | Purpose                           |
| ---------------------- | --------------------------------- |
| 🧠 ANFIS               | Neuro-fuzzy risk modeling         |
| 🔬 MATLAB              | FIS generation and ANFIS training |
| 🐍 Python              | Application and inference layer   |
| 🎈 Streamlit           | Interactive web application       |
| 🤖 Google Gemini API   | Conversational AI assistant       |
| 📊 NumPy               | Numerical computation             |
| 🐼 Pandas              | Data processing                   |
| 📈 Plotly / Matplotlib | Visualization                     |

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

The application should then be available at:

```text
http://localhost:8501
```

---

# 🔐 Environment Configuration

If Gemini functionality is enabled, configure your API key securely.

For local development, use an environment variable or Streamlit secrets rather than placing the key directly inside the Python source.

Example:

```text
GEMINI_API_KEY=your_api_key_here
```

For Streamlit Cloud, configure the secret through the application's **Secrets** settings.

---

# 📦 Deployment

The application can be deployed using **Streamlit Community Cloud**.

General deployment process:

```text
GitHub Repository
       ↓
Streamlit Community Cloud
       ↓
Install requirements.txt
       ↓
Load heart_anfis.fis
       ↓
Launch app.py
       ↓
Public Web Application
```

Live application:

🔗 https://cardio-fuzzy.streamlit.app/

---

# 📋 Requirements

Typical dependencies include:

```text
streamlit
numpy
pandas
plotly
scikit-learn
google-generativeai
```

The exact dependency list should always be taken from the project's `requirements.txt`.

---

# 🎯 Project Objectives

The project was developed to demonstrate how neuro-fuzzy systems can be applied to healthcare-oriented machine-learning problems while maintaining a greater degree of interpretability.

### Main objectives

* ✅ Develop an ANFIS-based cardiovascular risk model
* ✅ Produce a continuous 0–100 risk score
* ✅ Preserve an interpretable fuzzy rule structure
* ✅ Separate training, validation and test evaluation
* ✅ Integrate the model into a web application
* ✅ Provide interactive visualizations
* ✅ Add multilingual AI-assisted explanations
* ✅ Demonstrate MATLAB-to-Python model deployment

---

# 📚 Dataset

The project uses a heart-disease dataset containing clinical and diagnostic attributes commonly used in machine-learning research.

The dataset is used for **research and educational experimentation**.

Dataset-derived model performance should not be interpreted as evidence of clinical effectiveness because real-world clinical deployment requires substantially broader validation, including appropriate patient populations, external datasets, calibration analysis, prospective evaluation and regulatory review.

---

# ⚠️ Medical Disclaimer

**CardioFuzzy AI is an educational and research project.**

It is **not intended to:**

* diagnose heart disease;
* predict an individual's actual medical outcome;
* replace a physician or qualified healthcare professional;
* provide emergency medical advice;
* determine treatment;
* recommend medication;
* make clinical decisions.

The ANFIS output is a **machine-learning model score**, not a medically validated probability.

If you have symptoms or concerns about your cardiovascular health, consult an appropriately qualified healthcare professional.

In an emergency, contact your local emergency medical service.

---

# 🔮 Future Improvements

Potential future development includes:

* [ ] External-dataset validation
* [ ] Calibration analysis
* [ ] Confidence/uncertainty estimation
* [ ] Additional explainability techniques
* [ ] More extensive fuzzy-rule visualization
* [ ] Model comparison dashboard
* [ ] Improved accessibility
* [ ] Automated model monitoring
* [ ] Larger and more diverse datasets
* [ ] Prospective evaluation
* [ ] Secure production-grade API architecture

---

# 👨‍💻 Author

**Divyam Jhawar**

Electronics & Instrumentation Engineering
Nirma University

---

# 📄 License

This project is licensed under the **MIT License**.

See the [MIT License](https://opensource.org/licenses/MIT) for details.

---

# ⭐ Acknowledgements

This project combines concepts from:

* Adaptive Neuro-Fuzzy Inference Systems
* Fuzzy Logic
* Machine Learning
* Explainable AI
* Healthcare AI
* Natural Language Processing
* Generative AI
* Interactive Data Visualization

Built as an academic/research-oriented AI project exploring the intersection of **fuzzy systems, machine learning, and interactive healthcare applications**.

---

<p align="center">

### 🫀 CardioFuzzy AI

**Neuro-Fuzzy Intelligence • Explainable AI • Interactive Risk Assessment**

</p>
