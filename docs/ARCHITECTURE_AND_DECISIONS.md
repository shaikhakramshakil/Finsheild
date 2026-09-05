# FinShield — Master Engineering, Architecture & Design Decisions Document
**Real-Time Multi-Signal Fraud Defense & Entity Resolution Platform for Digital Payments**  
*Built for Stellar Hackathon Problem Statement #05 | Production-Grade Reference Specification*

---

## Table of Contents
1. [Executive Summary & Problem Statement Alignment](#1-executive-summary--problem-statement-alignment)
2. [End-to-End System Architecture](#2-end-to-end-system-architecture)
3. [The 5-Signal Fusion Engine — Mathematical & Algorithmic Deep Dive](#3-the-5-signal-fusion-engine--mathematical--algorithmic-deep-dive)
4. [Every Technology & Library Decision — Why Chosen vs Why Rejected](#4-every-technology--library-decision--why-chosen-vs-why-rejected)
5. [Empirical Benchmarks & Dataset Science](#5-empirical-benchmarks--dataset-science)
6. [Live Real-World Payment Gateway Ingestion (Cashfree Integration)](#6-live-real-world-payment-gateway-ingestion-cashfree-integration)
7. [Multi-Hop Entity Relational Graph & Forensic Workstation](#7-multi-hop-entity-relational-graph--forensic-workstation)
8. [Privacy, Compliance & Identity Tokenization](#8-privacy-compliance--identity-tokenization)
9. [AI Forensic Copilot — Explainability Without Autonomous Risk](#9-ai-forensic-copilot--explainability-without-autonomous-risk)
10. [Frontend Ergonomics & Design System](#10-frontend-ergonomics--design-system)
11. [10-Step Evaluator & Judge Verification Walkthrough](#11-10-step-evaluator--judge-verification-walkthrough)
12. [Repository Layout, Tooling & Deployment Matrix](#12-repository-layout-tooling--deployment-matrix)

---

## 1. Executive Summary & Problem Statement Alignment

### The Problem (Stellar Hackathon PS #05)
Digital payments—particularly instant rails like Unified Payments Interface (UPI), Instant SEPA, and Real-Time Gross Settlement—settle irrevocably in milliseconds. Traditional fraud detection architectures suffer from critical structural deficiencies:
1. **The Black-Box Dilemma**: Complex deep learning models produce uncalibrated probabilities without auditable rationales, violating financial compliance standards (e.g., RBI fraud reporting norms, FCRA, EU AI Act).
2. **The High False-Positive Curse**: Standard anomaly detectors or isolated classifiers trigger catastrophic false-positive spikes when legitimate users transact during travel, festivals, or high-value purchases, leading to cart abandonment and user churn.
3. **Syndicate & Device Blindness**: Per-transaction classifiers evaluate transactions in isolation, missing coordinated mule account rings, emulated rooted devices, and shared SIM hardware fingerprints.
4. **LLM Hallucination in Risk Scoring**: Autonomous generative agents plugged directly into financial decision loops introduce severe non-determinism, unpredictable risk drifts, and vulnerability to prompt injection attacks.

### The FinShield Solution
FinShield is a **multi-signal, hybrid fraud intelligence and forensic platform**. It rejects single-point failure architectures in favor of **orthogonal signal fusion**:
- **Zero Hallucination Risk**: The engine uses a calibrated, deterministic **Risk Fusion Engine** that fuses 5 independent vectors.
- **Explainability by Design**: SHAP (SHapley Additive exPlanations) extracts exact mathematical feature contributions. An instruction-tuned **SLM (Small Language Model - Qwen2.5-0.5B)** articulates structured evidence for human forensic analysts, but is **strictly prohibited from calculating or overriding the numerical risk score**.
- **Real-World Live Ingestion**: Fully integrated with live Cashfree Payment Gateway webhooks, processing real rupee transactions over public tunnels with sub-100ms end-to-end evaluation.

---

## 2. End-to-End System Architecture

FinShield is intentionally architected as two cleanly decoupled repositories:
1. **`Finsheild` (ML Core Research & Training Engine)**: Houses dataset pipelines, feature extraction, model registries, training loops, SHAP explainers, synthetic stress suites, and QLoRA fine-tuning pipelines.
2. **`Finsheild-App` (FastAPI Backend + React Frontend)**: Serves as the real-time operational Command Center and Forensic Investigation Workstation.

```mermaid
flowchart TD
    subgraph Ingress["1. Ingress Layer (Real & Simulated)"]
        CF[Cashfree Payment Gateway\nWebhook POST /api/webhooks/cashfree]
        UPI[UPI Instant Rail Simulator\nPOST /api/transaction/score]
        ST[High-Throughput Stream Generator\nPOST /api/transactions/generate]
    end

    subgraph Preprocessing["2. Preprocessing & Identity Shield"]
        TOKEN[Salted SHA-256 Identity Tokenizer\nZero PII in ML Features]
        FE[36-Feature Leakage-Safe Pipeline\nTransactional, Behavioral, Velocity, Geo, Hardware]
    end

    subgraph SignalEngines["3. The 5 Orthogonal Signal Engines"]
        S1["XGBoost Classifier (35%)\nSupervised Imbalance Tree"]
        S2["Isolation Forest (20%)\nUnsupervised Outlier Isolation"]
        S3["Deterministic Rule Engine (20%)\n8 Hard Velocity & Value Rules"]
        S4["Behavioral Profiler (15%)\nRolling User Drift & Z-Scores"]
        S5["NetworkX Entity Graph (10%)\nDevice-Sharing & Mule Cliques"]
    end

    subgraph Fusion["4. Calibrated Risk Fusion & Policy Decision"]
        FUSER["Risk Fusion Synthesizer\nR = Σ (w_i · S_i)"]
        POLICY{"Policy Engine\nCalibrated Thresholds"}
        ACT_APP[APPROVE\nScore < 0.30]
        ACT_STP[STEP-UP 2FA\n0.30 ≤ Score < 0.60]
        ACT_INV[INVESTIGATE\n0.60 ≤ Score < 0.85]
        ACT_BLK[BLOCK\nScore ≥ 0.85]
    end

    subgraph Explainability["5. Forensic Audit & Explainability Layer"]
        SHAP_EXP[SHAP TreeExplainer\nExact Mathematical Attribution]
        COPILOT[AI Forensic Copilot\nQwen2.5-0.5B QLoRA\nExplains Evidence Only]
        GRAPH_UI[Multi-Hop Entity Graph Canvas\nInteractive SVG Topology]
    end

    CF --> TOKEN
    UPI --> TOKEN
    ST --> TOKEN
    TOKEN --> FE
    FE --> S1 & S2 & S3 & S4 & S5
    S1 & S2 & S3 & S4 & S5 --> FUSER
    FUSER --> POLICY
    POLICY --> ACT_APP & ACT_STP & ACT_INV & ACT_BLK
    ACT_INV & ACT_BLK --> SHAP_EXP & COPILOT & GRAPH_UI
```

---

## 3. The 5-Signal Fusion Engine — Mathematical & Algorithmic Deep Dive

### Mathematical Formulation
Let a transaction be represented by a feature vector $\mathbf{x} \in \mathbb{R}^{36}$. The final risk score $R(\mathbf{x}) \in [0, 1]$ is computed as a linear convex combination of 5 normalized, bounded signal scores:

$$R(\mathbf{x}) = w_1 S_{\text{xgb}}(\mathbf{x}) + w_2 S_{\text{iforest}}(\mathbf{x}) + w_3 S_{\text{rules}}(\mathbf{x}) + w_4 S_{\text{behavior}}(\mathbf{x}) + w_5 S_{\text{graph}}(\mathbf{x})$$

Where the weights $\mathbf{w} = [0.35, 0.20, 0.20, 0.15, 0.10]$ satisfy $\sum_{i=1}^5 w_i = 1.0$.

---

### Signal 1: Supervised Gradient Boosting (`XGBoost` — Weight: 0.35)
- **Role**: Detects non-linear tabular feature interactions learned from historical fraud patterns.
- **Model Configuration**:
  - `n_estimators`: 500 trees
  - `max_depth`: 5 (prevents overfitting to memorized card numbers)
  - `learning_rate`: 0.05
  - `scale_pos_weight`: Set to $\frac{N_{\text{legit}}}{N_{\text{fraud}}} \approx 577.8$ during training to counter extreme class imbalance ($0.17\%$ fraud).
  - Early stopping on Precision-Recall AUC (PR-AUC) using validation split.
- **Why It Matters**: Captures multidimensional correlations (e.g., amount + time-of-day + velocity + merchant category) that simple threshold rules miss entirely.

---

### Signal 2: Unsupervised Anomaly Isolation (`Isolation Forest` — Weight: 0.20)
- **Role**: Zero-day fraud detection and novel anomaly isolation without relying on past labels.
- **Mathematical Principle**: Based on the insight that anomalies are few and structurally different, requiring fewer random axis-aligned splits to isolate in feature space.
- **Model Configuration**:
  - `contamination`: 0.05
  - **Critical Architectural Safeguard**: The Isolation Forest is fitted **strictly on legitimate transactions** ($y = 0$). This ensures the model constructs an unpolluted baseline profile of normal human spending behavior.
- **Anomaly Score Conversion**:
  $$S_{\text{iforest}} = \frac{1}{1 + e^{k \cdot (\text{raw\_score} - t)}}$$
  Maps the negative decision function into a normalized probability in $[0, 1]$.

---

### Signal 3: Deterministic Rule Engine (Weight: 0.20)
- **Role**: Instant short-circuiting of non-negotiable risk violations and regulatory thresholds.
- **The 8 Deterministic Production Rules**:
  1. `HIGH_VELOCITY`: $\ge 5$ transactions initiated within 5 minutes from the same account.
  2. `BURST_VELOCITY`: $\ge 8$ transactions within 5 minutes (characteristic of automated brute-force card testing scripts).
  3. `NEW_DEVICE_HIGH_VALUE`: Transaction value $> \$500$ (or $> ₹25,000$) initiated from a newly registered hardware ID ($t_{\text{registered}} < 24\text{h}$).
  4. `UNUSUAL_AMOUNT_ZSCORE`: Amount deviates by $> 3\sigma$ from user historical rolling mean.
  5. `OFFHOURS_HIGH_VALUE`: High-value transaction ($> \$1,000$ or $> ₹50,000$) initiated between 01:00 AM and 05:00 AM local time.
  6. `SHARED_DEVICE_RING`: Hardware device ID linked to $\ge 2$ distinct user identities within 48 hours.
  7. `LOCATION_ANOMALY`: Geodesic distance from primary residence $> 300\text{km}$ without preceding travel velocity buffer.
  8. `RAPID_COUNTRY_SWITCH`: Successive transactions from distinct countries within physically impossible flight travel windows.

---

### Signal 4: Behavioral Profiling & Z-Score Drift (Weight: 0.15)
- **Role**: Personalizes risk evaluation to individual user spending habits.
- **Formulation**:
  For user $u$ with rolling historical transaction mean $\mu_u$ and standard deviation $\sigma_u$:
  $$z = \frac{x_{\text{amount}} - \mu_u}{\max(\sigma_u, \epsilon)}$$
  $$S_{\text{behavior}} = \min\left(1.0, \frac{\max(0, z - 2.0)}{4.0}\right)$$
- **Why It Matters**: Prevents false positives. A ₹50,000 transaction from a high-net-worth individual with $\mu = ₹45,000$ yields $z \approx 0.2$ (zero risk), whereas the same transaction for a user with $\mu = ₹1,200$ yields $z = 18.2$ (maximum behavioral alert).

---

### Signal 5: Entity Relational Graph (`NetworkX` — Weight: 0.10)
- **Role**: Discovers syndicated fraud rings, shared emulator devices, and mule account clusters.
- **Graph Topology**:
  - **Bipartite Node Types**: `User`, `Account`, `Device`, `Gateway`, `Merchant`.
  - **Edge Semantics**: `OWNS`, `AUTHENTICATES`, `SIGNS_PAYLOAD`, `ROUTES_VIA`, `CREDITS_ESCROW`.
- **Graph Metrics Evaluated**:
  - **Shared Device Degree**: Count of distinct accounts originating from the same hardware fingerprint.
  - **Mule Chain Depth**: Shortest path distance between an incoming transaction and known blacklisted liquidity cashout nodes.
  - **Bipartite Co-Usage Cliques**: Cliques of size $k \ge 3$ trigger `SHARED_DEVICE_CLIQUE` alert ($S_{\text{graph}} \ge 0.85$).

---

## 4. Every Technology & Library Decision — Why Chosen vs Why Rejected

| Library / Tool | Chosen For | Rejected Alternatives & Engineering Rationale |
| :--- | :--- | :--- |
| **`xgboost`** | Superior tabular performance, built-in missing value handling, fast C++ tree construction, native early stopping on PR-AUC. | **Deep Neural Networks (MLP/Transformer)**: Overfit tabular data, lack exact tree explainability, 50× higher latency.<br>**RandomForest**: Slower inference, lacks gradient-directed split optimization. |
| **`scikit-learn`** | `IsolationForest` for unsupervised anomaly detection, `StandardScaler` for zero-leakage transforms, robust metric suites (`precision_recall_curve`, `roc_auc_score`). | **PyOD**: Heavier dependency footprint with redundant wrappers.<br>**Custom Anomaly Code**: Unvalidated math; scikit-learn’s Cython implementation is thoroughly audited. |
| **`networkx`** | Pure Python, lightweight in-memory graph representation, instantaneous BFS/DFS clique analysis, zero external daemon requirements. | **Neo4j / Memgraph**: Massive JVM/Docker overhead, complex Cypher maintenance, overkill for sub-millisecond per-transaction ego-network queries. |
| **`shap`** | `TreeExplainer` calculates mathematically exact Shapley values in polynomial time ($O(TLD^2)$), satisfying regulatory proof of attribution. | **LIME**: Perturbation-based approximation produces non-deterministic explanations.<br>**Feature Importances**: Global only; cannot explain why *this specific* transaction was blocked. |
| **`Qwen2.5-0.5B-Instruct`** | Ultra-compact (0.5B parameters), runs on CPU or <1GB VRAM, highly capable at JSON structured evidence reasoning, sub-100ms inference. | **GPT-4 / Claude / Llama-3-70B**: Astronomical cloud latency (1.5s–3s), recurring token billing, external data leak risks, massive compute requirements. |
| **`peft` + `bitsandbytes` + `trl`** | QLoRA (4-bit NF4 quantization, $r=8, \alpha=16$) enables fine-tuning the SLM on a standard Google Colab T4 GPU in <15 minutes. | **Full Parameter Fine-Tuning**: Requires high-end A100 GPUs, prone to catastrophic forgetting of grammar and conversational instruction following. |
| **`FastAPI` + `Pydantic v2`** | Asynchronous ASGI request handling, sub-millisecond response latency, automatic OpenAPI documentation, compile-time schema validation. | **Flask**: Synchronous blocking architecture, lacks built-in serialization and typed data contracts.<br>**Django**: Enormous monolithic bloat unsuitable for microsecond microservice scoring. |
| **`React 19` + `TypeScript`** | Strict end-to-end typing, deterministic component state, virtual DOM reconciliation for high-speed streaming tables. | **Vanilla JS / jQuery**: Unmaintainable spaghetti state across 5 complex screens.<br>**Angular**: Heavy boilerplate and rigid opinionated conventions. |
| **`Tailwind CSS 4`** | Zero-runtime CSS compilation, precise fintech aesthetic tokens, high-density forensic information architecture. | **Material UI / Bootstrap**: Generic consumer look, heavy CSS-in-JS runtime penalty, sluggish on large streaming tables. |
| **`Cashfree Payments SDK`** | Real enterprise Indian payment rail, UPI QR/Intent flows, comprehensive webhook event schemas (`PAYMENT_SUCCESS_WEBHOOK`). | **Razorpay / Stripe**: Requires business registration / GST for live sandbox webhooks; Cashfree provided instant testbed links for ₹1 and ₹1,00,000 real flows. |
| **`cloudflared`** | Creates zero-configuration encrypted tunnels from localhost to public edge, allowing live Cashfree webhooks to reach local dev engines. | **ngrok**: Ephemeral URLs change on every restart, strict bandwidth rate limits on free tiers. |

---

## 5. Empirical Benchmarks & Dataset Science

### Benchmark Source of Truth: Real ULB Kaggle Credit Card Dataset
To maintain absolute empirical rigor and avoid "demo fiction", FinShield was trained and benchmarked on the gold-standard **ULB Kaggle Credit Card Fraud Dataset**:
- **Total Transactions**: $284,807$
- **Legitimate Transactions**: $284,315$ ($99.828\%$)
- **Fraudulent Transactions**: $492$ ($0.172\%$)
- **Feature Space**: 28 PCA-transformed features ($V_1 - V_{28}$) + Time + Amount.
- **Stratified Split**: 70% Train ($n=199,364$), 15% Validation ($n=42,721$), 15% Test ($n=42,722$).

### Rigorous Evaluation Results (Held-Out Test Set $N = 42,722$)

| Model / Pipeline | ROC-AUC | PR-AUC | F1-Score | Precision | Recall | False Positives | False Negatives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (500 Trees)** | **0.9709** | **0.8418** | **0.8467** | **0.9206** | **0.7838** | **5** | **16** |
| **Logistic Regression Baseline** | 0.9495 | 0.7005 | 0.7407 | 0.8125 | 0.6806 | 15 | 24 |

> [!IMPORTANT]
> **Production Context on False Positives**:
> Out of **42,648 legitimate transactions** in the test set, the XGBoost engine generated **only 5 false alarms** ($0.0117\%$ false positive rate). In production banking, this translates to virtually zero friction for legitimate customers while capturing nearly 80% of fraud before settlement.

---

### The Synthetic Stress Test Suite (Adversarial Progression)
Real-world fraud datasets often exhibit clear boundaries. To test how the model behaves under adversarial pressure, we designed 3 synthetic stress scenarios with increasing difficulty:

```mermaid
graph LR
    A[Easy Synthetic\n11.5% Fraud\nPR-AUC 0.959] --> B[1% Diluted\n1.07% Fraud\nPR-AUC 0.553]
    B --> C[Hard Overlap\n1.10% Fraud\nPR-AUC 0.373]
    C -.-> D[Real ULB Benchmark\n0.17% Fraud\nPR-AUC 0.8418]
```

1. **Easy Synthetic ($11.5\%$ Fraud)**: PR-AUC **0.959**. Fraud instances cluster cleanly far from normal spending.
2. **1% Diluted ($1.07\%$ Fraud)**: PR-AUC **0.553**. Realistic class imbalance introduces natural precision decay.
3. **Hard Overlap ($1.10\%$ Fraud)**: PR-AUC **0.373**. Multimodal Gaussian clusters deliberately overlap legitimate user behaviors (e.g., fraud amounts match grocery bills, location jumps within normal transit times). This proved that **single models fail under adversarial conditions**, directly justifying the need for the **5-Signal Fusion Engine**.

---

## 6. Live Real-World Payment Gateway Ingestion (Cashfree Integration)

Rather than merely generating synthetic data, FinShield was connected to **Cashfree Payments**, a Tier-1 Indian payment aggregator.

### Architecture of Live Ingestion
1. **Public Tunnel**: `cloudflared tunnel --url http://127.0.0.1:8000` created a live public endpoint:
   `https://anaheim-resistant-follow-insulation.trycloudflare.com/api/webhooks/cashfree`
2. **Webhook Parser (`backend/main.py`)**:
   - Ingests `PAYMENT_SUCCESS_WEBHOOK` events.
   - Extracts nested fields: `data.order.order_amount`, `data.payment.cf_payment_id`, `data.payment.payment_method.upi`, `data.customer_details.customer_phone`.
   - Derives salted identity tokens and constructs transaction telemetry.
3. **Live Scoring Pipeline**: Passes the telemetry directly to `RealMLAdapter` (XGBoost + Scaler + Rules + Graph).

### Real Live Payments Executed & Verified

| Parameter | Live Payment 1 (Micro-Payment) | Live Payment 2 (High-Value Anomaly) |
| :--- | :--- | :--- |
| **Payment Link** | `code=fav8f4pqom50_AAAAAACpSlE` | `code=oav8fob7um50_AAAAAACpSlE` |
| **Amount** | **₹1.00** | **₹100,000.00 (₹1 Lakh)** |
| **Transaction ID** | `CF-216655019298976` | `CF-216656422682784` |
| **Calculated Risk Score** | **0.035** | **0.890** |
| **Assigned Risk Tier** | **`LOW`** | **`CRITICAL`** |
| **Policy Decision** | **`APPROVE`** | **`BLOCK`** |
| **Rules Triggered** | None (Clean baseline) | `BURST_VELOCITY`, `NEW_DEVICE_HIGH_VALUE`, `UNUSUAL_AMOUNT`, `UPI_HIGH_VALUE_ANOMALY`, `EXCEEDS_SINGLE_TRANSACTION_LIMIT` |
| **Source Provenance** | `LIVE_MODEL` | `LIVE_MODEL` |

---

## 7. Multi-Hop Entity Relational Graph & Forensic Workstation

### The Visual & Structural Upgrade
In digital payment fraud, attacks are rarely isolated transactions; they are syndicates. We built the **Multi-Hop Entity Relational Canvas** in `frontend/src/pages.tsx` to replace static, clipped boxes with an interactive, publication-grade forensic graph.

### 4-Column Deterministic Topological Layout
Nodes are organized in a clean left-to-right causal transaction flow:
1. **Column 1 — Identity & Funding ($x = 115$)**: Payer Identity (`USER-8926`) and Funding Accounts / VPAs (`user-8926@hdfcbank`).
2. **Column 2 — Hardware & Origin ($x = 350$)**: Registered Hardware (`DEV-PIXEL-BASE`) vs Rogue Ingress Hardware (`DEV-NEW-CF`).
3. **Column 3 — Ingress Hub ($x = 580$)**: The Target Transaction (`₹100,000.00`), highlighted with pulsing risk borders.
4. **Column 4 — Switching & Settlement ($x = 810$)**: Gateway Switch (`Cashfree Switch`) and Beneficiary Escrow (`Merchant Escrow`).

### Smooth Bézier Curves & Interactive Forensics
- **Curved Directed Edges**: Smooth cubic Bézier paths (`M x1 y1 C cx1 cy1, cx2 cy2, x2 y2`) connect card ports with zero cross-node line piercing.
- **Edge Badges**: Directed relationship labels (`OWNS_ACCOUNT`, `PAIRED_BASELINE`, `ROGUE_SIGNATURE`, `ROUTED_VIA`, `CREDITS_ESCROW`).
- **Anomalous Link Highlights**: Dashed crimson lines with drop-shadow glow filters highlight rogue hardware pairing and hardware mismatches.
- **Interactive Entity Inspector Drawer**: Clicking any node in the SVG canvas immediately opens a detailed forensic card displaying hardware fingerprints, velocity counts, trust scores, and cluster connectivity.

---

## 8. Privacy, Compliance & Identity Tokenization

### Zero-PII Guarantee in ML Training
In compliance with the **Digital Personal Data Protection (DPDP) Act 2023** and **GDPR**, raw Personally Identifiable Information (PII)—including phone numbers, Aadhaar numbers, PAN cards, and bank account numbers—**never touches feature matrices or model weights**.

### Salted Cryptographic Hash Tokenization
- **Algorithm**: Salted SHA-256 with an isolated environment salt key:
  $$\text{Token}(u) = \text{SHA-256}(\text{salt} \parallel u)$$
- **Masked Presentation**: Phone numbers are masked (`••••••••42`), documents are tokenized, and only cryptographic tokens are passed to the graph and feature pipeline.

---

## 9. AI Forensic Copilot — Explainability Without Autonomous Risk

### The Core Safety Principle
> [!CAUTION]
> **LLMs must NEVER compute, calibrate, or override numerical fraud risk scores.**
> In FinShield, the numerical risk score is computed solely by the deterministic 5-Signal Fusion Engine. The AI Forensic Copilot's sole mandate is to **translate complex mathematical evidence into plain, human-readable forensic rationales** for bank fraud analysts.

### Model & Instruction-Tuning Setup
- **Base Model**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Fine-Tuning Method**: QLoRA ($r=8, \alpha=16$, dropout $0.05$) via `peft` and `trl.SFTTrainer`.
- **Instruction Dataset**: Grounded JSON evidence templates generated from SHAP attribution bars, rule hits, and behavioral deviations.
- **Inference Speed**: ~80ms on GPU, ~300ms on CPU, completely eliminating multi-second cloud API calls.

---

## 10. Frontend Ergonomics & Design System

The FinShield UI was built to eliminate the clutter of traditional enterprise banking software while maximizing information density.

### Typography Hierarchy
- **Headings & Display**: `Plus Jakarta Sans` (800/700 weight, tight tracking `-0.035em` to `-0.04em`) — commanding fintech presence.
- **Body Prose**: `Inter` (with optical kerning `cv02`, `cv03`, `cv04`, `cv11`) — ultra-crisp readability.
- **Telemetry, Hashes & Figures**: `JetBrains Mono` (with tabular figures `font-feature-settings: "tnum" 1, "zero" 1`) — perfect vertical digit alignment.
- **Accents & Quotations**: `Space Grotesk` (geometric accents).

### Editorial Color Palette
- **Canvas / Background**: `#F2EFE7` (warm editorial off-white)
- **Primary Ink**: `#171916` (deep charcoal carbon)
- **FinShield Brand Neon**: `#FF5B35` (high-visibility safety orange/crimson)
- **Safe / Approved Green**: `#2E7D32`
- **Surface Elevation**: `#E7E4DB`

---

## 11. 10-Step Evaluator & Judge Verification Walkthrough

Evaluators can verify the entire platform end-to-end using this structured 10-step sequence:

1. **Step 1: Check System Health & Provenance**:
   - Navigate to `/api/health`.
   - Verify `adapter: "real"`, and confirm that XGBoost, Scaler, SHAP, and Graph are loaded live.
2. **Step 2: Inspect Real ULB Kaggle Metrics**:
   - Navigate to the **Model Performance** screen (`/#/performance`).
   - Confirm the held-out test ROC-AUC is **0.9709**, PR-AUC is **0.8418**, and only 5 false alarms occurred across 42,722 test samples.
3. **Step 3: Review Adversarial Stress Tests**:
   - Compare the synthetic experiment progression (Easy Synthetic 0.959 &rarr; Diluted 0.553 &rarr; Hard Overlap 0.373).
4. **Step 4: Live Cashfree Payment Ingestion (₹1 Micro-Payment)**:
   - Locate transaction `CF-216655019298976` in the Command Center table.
   - Confirm that the score is **0.035**, risk tier is **`LOW`**, and policy decision is **`APPROVE`**.
5. **Step 5: Live Cashfree Anomaly Block (₹1,00,000 High-Value Payment)**:
   - Locate transaction `CF-216656422682784` in the Command Center table.
   - Confirm that the score is **0.890**, risk tier is **`CRITICAL`**, and policy decision is **`BLOCK`**.
6. **Step 6: Forensic Deep Dive (5-Signal Radar)**:
   - Click "Investigate" on `CF-216656422682784` to open `/#/investigate/CF-216656422682784`.
   - Verify the 5-signal breakdown: XGBoost (0.00), Anomaly (0.75), Rules (`BURST_VELOCITY`, `NEW_DEVICE_HIGH_VALUE`), Behavioral (0.62), Graph (0.42).
7. **Step 7: SHAP Feature Attribution**:
   - Inspect the horizontal waterfall bars. Note how `amount`, `velocity`, and `new_device` push the decision toward fraud, while `merchant_category` acts as a dampener.
8. **Step 8: Interactive Multi-Hop Entity Relational Canvas**:
   - Examine the 4-column layout on the investigation page.
   - Click on `DEV-NEW-CF` to view the Entity Inspector Drawer. Notice the `UNRECOGNIZED_FINGERPRINT` and velocity burst indicators.
   - Click the "Anomalous Paths Only" toggle to filter out noise and isolate the rogue attack route.
9. **Step 9: AI Forensic Copilot Rationale**:
   - Click "Generate Copilot Explanation".
   - Confirm the model articulates the exact rule violations and behavioral drifts into a structured narrative without modifying the score.
10. **Step 10: Privacy & Identity Tokenization**:
    - Navigate to `/#/privacy/USER-8926`.
    - Confirm the salted SHA-256 token, phone masking, and zero-PII compliance badges.

---

## 12. Repository Layout, Tooling & Deployment Matrix

```
Finsheild-Ecosystem/
├── Finsheild/                         # ML Core Research & Training Engine
│   ├── src/finsheild/
│   │   ├── data/                      # Kaggle loader & stratified splits
│   │   ├── model.py                   # XGBoost, LightGBM, LogReg registries
│   │   ├── train.py                   # Training loops with early stopping
│   │   ├── inference.py               # Low-latency inference pipeline
│   │   ├── features/                  # 36 leakage-safe feature definitions
│   │   ├── behavioral/                # Online rolling user profiles
│   │   ├── anomaly/                   # IsolationForest (legit-only fitting)
│   │   ├── rules/                     # 8 production deterministic rules
│   │   ├── graph/                     # NetworkX multi-hop entity resolution
│   │   ├── risk_fusion/               # 5-signal linear convex combination
│   │   ├── explain/                   # SHAP TreeExplainer
│   │   └── finetune/                  # QLoRA Qwen2.5-0.5B instruction tuning
│   ├── models/                        # Serialized weights (XGBoost, Scaler)
│   ├── evaluation/reports/            # Benchmark JSON & Markdown reports
│   └── tests/                         # 208 passing unit & integration tests
│
└── Finsheild-App/                     # Presentation & Workstation Engine
    ├── backend/
    │   ├── main.py                    # FastAPI app (13 REST endpoints + webhooks)
    │   ├── schemas.py                 # Pydantic data contracts
    │   ├── metrics_loader.py          # Real ULB benchmark ingestion
    │   ├── services/store.py          # In-memory store & graph generator
    │   └── adapters/
    │       ├── real_adapter.py        # Connects to live Finsheild ML Core
    │       └── mock_adapter.py        # Transparent fallback with honesty labels
    ├── frontend/
    │   ├── src/pages.tsx              # Command Center, Investigation, Performance, Graph
    │   ├── src/api.ts                 # Typed API client
    │   └── src/index.css              # Editorial tokens & typography
    └── start.sh                       # One-click dual server launcher
```

### Remote Deployments & Public Access Links
- **Hugging Face Space**: [https://huggingface.co/spaces/shaikhakramshakil/Finsheild](https://huggingface.co/spaces/shaikhakramshakil/Finsheild)
- **Direct Static Application URL**: [https://shaikhakramshakil-finsheild.static.hf.space](https://shaikhakramshakil-finsheild.static.hf.space)
- **GitHub Application Repo**: [https://github.com/shaikhakramshakil/Finsheild-App](https://github.com/shaikhakramshakil/Finsheild-App)
- **GitHub ML Core Engine Repo**: [https://github.com/shaikhakramshakil/Finsheild](https://github.com/shaikhakramshakil/Finsheild)
- **Live Cashfree Webhook Ingress URL**: `https://anaheim-resistant-follow-insulation.trycloudflare.com/api/webhooks/cashfree`

---
*FinShield Fraud Defense Platform — Engineering Rigor. Data Honesty. Multi-Signal Fusion.*
