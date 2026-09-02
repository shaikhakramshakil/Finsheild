# Finsheild - ML-FIRST DEVELOPMENT PLAN

We are building Finsheild, a research-oriented digital payment fraud intelligence platform.

The current objective is ONLY to build and validate the machine-learning and LLM intelligence layer.

## IMPORTANT

Do NOT build:

- React
- frontend
- dashboard
- UI
- FastAPI
- deployment
- authentication
- production infrastructure

Those will be implemented later during the hackathon.

The current priority is obtaining working, evaluated ML models and a fine-tuned fraud-investigation LLM.

---

# DEVELOPMENT ORDER

Implement the project in exactly this order:

## Phase 1: Dataset

Find/configure a suitable publicly available fraud dataset.

Do not fabricate the dataset.

Create:

- dataset loader
- preprocessing
- exploratory analysis
- train/validation/test split

Prevent data leakage.

Document:

- dataset source
- features
- target variable
- class distribution
- preprocessing
- split strategy

Do not claim results before running the experiments.

---

# Phase 2: Baseline

Train an interpretable baseline classifier.

Prefer:

Logistic Regression

Evaluate:

- precision
- recall
- F1
- ROC-AUC where appropriate
- PR-AUC where appropriate
- confusion matrix

Save the model and metrics.

---

# Phase 3: XGBoost

Train XGBoost as the primary supervised fraud classifier.

Perform reasonable preprocessing and hyperparameter tuning.

Do not blindly optimize for accuracy because fraud datasets can be highly imbalanced.

Evaluate using appropriate metrics.

Save:

models/xgboost/

and:

evaluation/reports/

---

# Phase 4: Synthetic Digital Payment Environment

The public dataset may not contain all relationships required for our final architecture.

Create a clearly labelled synthetic transaction environment containing:

- users
- accounts
- devices
- merchants
- transactions
- locations

Generate realistic relationships.

Include synthetic suspicious scenarios:

- account takeover
- unusual amount
- unusual transaction time
- transaction velocity
- new device
- unusual location
- device sharing
- mule-account-like behavior
- unusual merchant behavior

Clearly distinguish synthetic data from public data.

---

# Phase 5: Feature Engineering

Create reusable feature modules.

Transaction features:

- amount
- transaction type
- merchant category
- hour
- day

Behavioral features:

- historical average amount
- amount deviation
- transaction frequency
- usual transaction hour
- usual merchant categories
- known/new device

Velocity features:

- transactions in recent time windows
- recent transaction amount
- transaction bursts

Location features:

- previous location
- current location
- approximate distance
- unusual location indicator

Device features:

- known/new device
- device-account relationships
- number of accounts using a device

Every feature must be generated without leaking future information.

---

# Phase 6: Behavioral Profiling

Create user behavioral profiles using historical transactions.

For each user with enough history, estimate:

- normal transaction amount
- normal frequency
- typical hours
- common merchants/categories
- known devices
- normal geographic behavior

For each new transaction calculate deviations from that profile.

These deviations become fraud signals.

---

# Phase 7: Anomaly Detection

Implement Isolation Forest initially.

The anomaly model must detect unusual behavior.

Important:

Anomaly does NOT automatically mean fraud.

Expose anomaly score separately.

Evaluate whether anomaly signals improve the overall fraud detection system.

---

# Phase 8: Rule Engine

Implement transparent fraud rules.

Examples:

- excessive transaction velocity
- new device + high-value transaction
- unusual location
- unusual amount
- multiple accounts sharing device
- abnormal transaction timing

Every rule should return:

rule\_id\
rule\_name\
severity\
description

Rules must be configurable.

---

# Phase 9: Graph Intelligence

Build a transaction relationship graph using NetworkX.

Nodes:

- users
- accounts
- devices
- merchants

Edges:

- owns
- uses
- transacts\_with
- shares\_device

Calculate useful graph features:

- node degree
- shared-device count
- connected suspicious accounts
- suspicious neighbor count
- transaction connectivity

Do NOT implement a GNN yet.

First establish whether graph-derived features provide useful information.

---

# Phase 10: Risk Fusion

Combine:

1. XGBoost fraud score
2. anomaly score
3. behavioral signals
4. graph signals
5. rule signals

Create a transparent risk-fusion engine.

Output:

risk\_score\
risk\_level\
decision\
evidence

Risk levels:

GREEN\
YELLOW\
RED

Possible decisions:

APPROVE\
STEP\_UP\
BLOCK\
INVESTIGATE

Thresholds must be configurable and must not be presented as real banking standards.

---

# Phase 11: Explainability

Use SHAP where appropriate for the supervised model.

For each suspicious transaction produce evidence such as:

- amount significantly above historical behavior
- new device
- high velocity
- unusual transaction time
- unusual location
- suspicious graph relationship

The evidence must come from actual calculated features.

Do not allow explanations to invent evidence.

---

# Phase 12: Generate LLM Training Dataset

Now that the fraud pipeline works, generate the LLM training dataset from the actual pipeline outputs.

The LLM is NOT the fraud classifier.

Its job is to act as a:

**Fraud Investigation Copilot**

Input:

structured fraud evidence.

Example:

{\
"transaction\_amount": 50000,\
"historical\_average": 4200,\
"new\_device": true,\
"location\_distance\_km": 400,\
"recent\_transaction\_count": 8,\
"xgboost\_score": 0.91,\
"anomaly\_score": 0.83,\
"triggered\_rules": [\
"NEW\_DEVICE\_HIGH\_VALUE",\
"HIGH\_VELOCITY"\
],\
"graph\_signals": {\
"shared\_device\_accounts": 4\
}\
}

Expected output:

{\
"risk\_level": "HIGH",\
"fraud\_type": "ACCOUNT\_TAKEOVER",\
"summary": "...",\
"evidence": [],\
"recommended\_action": "..."\
}

Generate diverse examples covering:

- legitimate transactions
- account takeover
- unusual spending
- velocity fraud
- device compromise
- unusual location
- mule-account behavior
- merchant anomalies
- coordinated account/device behavior

The training data must be derived from actual calculated features and model outputs.

---

# Phase 13: Base LLM Evaluation

Select an appropriate currently available small open-weight instruct model suitable for cloud fine-tuning.

Prefer a model in the approximately 2B-4B range initially.

Do not download the model permanently to the user's computer.

The model should be downloaded into the cloud/Colab environment.

Before fine-tuning:

Evaluate the base model on the held-out fraud-investigation test dataset.

Record measurable results.

---

# Phase 14: QLoRA Fine-tuning

Fine-tune the existing open-weight model using QLoRA/LoRA.

Use:

- PyTorch
- Transformers
- PEFT
- TRL where appropriate
- bitsandbytes where supported
- Hugging Face datasets

The base model should remain frozen.

Train adapter weights.

Training must run in Google Colab or another cloud GPU environment.

Automatically detect CUDA.

Do not assume a particular GPU.

All GPU-dependent parameters must be configurable.

Save:

- LoRA adapter
- tokenizer
- training configuration
- training logs

---

# Phase 15: Fine-tuned LLM Evaluation

Evaluate the fine-tuned model on the exact same held-out test set used for base-model evaluation.

Compare:

BASE MODEL\
vs\
FINE-TUNED MODEL

Measure where meaningful:

- fraud scenario classification
- risk classification
- F1
- JSON validity
- evidence grounding
- hallucination/error rate
- consistency

Generate a comparison report.

Never invent values.

---

# Phase 16: Model Export

At the end of the ML phase, produce:

models/\
├── baseline/\
├── xgboost/\
├── anomaly/\
├── risk\_fusion/\
└── llm/\
└── adapter/

Also produce:

evaluation/\
├── reports/\
├── figures/\
└── metrics.json

The final artifacts must be usable later by the backend/UI.

---

# PHASE CONTROL

Work on ONE phase at a time.

After completing a phase:

1. Run tests.
2. Run the relevant experiment.
3. Save outputs.
4. Update documentation.
5. Report exactly what happened.
6. Report actual metrics if the experiment ran.
7. Clearly identify anything that remains incomplete.
8. STOP.

Do not automatically proceed to the next phase.

---

# AI AGENT RULES

You are allowed to:

- create files
- edit files
- create Python scripts
- install dependencies
- execute experiments
- debug errors
- write tests
- generate datasets
- train models
- evaluate models

You are NOT allowed to:

- fabricate results
- fabricate datasets
- claim a model was trained when it wasn't
- hide errors
- silently skip failed experiments
- introduce unnecessary technologies
- build the frontend during this stage

When GPU training is required, prepare the code for Google Colab.

The user's local computer does not have a GPU.

---

# FIRST COMMAND

Start with Phase 1 only.

First inspect the repository.

Then implement the dataset pipeline.

Do not train the LLM.

Do not build the UI.

Do not proceed beyond Phase 1 without explicit instruction.
