<div align="center">

# Marek Cackowski
### Junior Machine Learning Engineer | Deep Learning & Multimodal Systems

*Designing adaptive ML pipelines and cognitive funnels that route routine cases to fast models—reserving heavy Deep Learning compute for high-risk and ambiguous edge cases.*

</div>

---

### Tech Stack

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit_Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/SQL-003B57?style=flat-square&logo=mysql&logoColor=white" />
  <img src="https://img.shields.io/badge/PostgreSQL-316192?style=flat-square&logo=postgresql&logoColor=white" />
  <img src="https://img.shields.io/badge/MongoDB-47A248?style=flat-square&logo=mongodb&logoColor=white" />
  <img src="https://img.shields.io/badge/Redis-DC382D?style=flat-square&logo=Redis&logoColor=white" />
  <img src="https://img.shields.io/badge/MinIO-C72026?style=flat-square&logo=minio&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white" />
</p>

---

### Engineering Approach

- **Compute-Aware Architectures:** Building multi-stage cascades and uncertainty routers instead of running expensive neural networks blindly on every sample.
- **Multimodal & Hybrid Modeling:** Combining tabular ensembles, spatial 2D/3D computer vision, graph networks, and character-level transformers.
- **Production-First Execution:** Decoupling lightweight APIs from heavy GPU inference via asynchronous workers, real-time WebSockets, and containerized storage.

---

### Flagship Projects

#### 1. [TBI Clinical Decision Support System](https://github.com/MarekCackowski/TBI)
> **Multimodal ICU triage architecture predicting physiological deterioration through a 5-stage cognitive funnel optimized for ~95% Negative Predictive Value.**

```text
[EHR Telemetry] ──▶ Stage 1: Tabular NAS Ensemble (Fast Triage)
                         ├── (Clear-Cut / Certain — Skip Compute) ──────────────────────┐
                         │ (Borderline / Anomaly)                                       │
                         ▼                                                              │
                    Stage 2: Isolation Forest & GMM ──▶ Stage 3: Specialist Pool        │
                                                             ├── (Resolved) ────────────┤
                                                             │ (Critical Last Resort)   │
                                                             ▼                          │
                                                        Stage 4: 3D CT Vision           │
                                                             │                          │
                                                             ▼                          │
 [GNN Drug Graphs + Bayesian Trajectories] ──▶ Stage 5: EBM Meta-Judge ◀────────────────┘
```

* **Core Impact:** Minimizes false negatives in critical care while avoiding unnecessary 3D volumetric inference on clear-cut patients.

<details>
<summary><b>Deep Dive: Cognitive Funnel & Model Architecture</b></summary>

* **Stage 1 — Initial Rapid Screening (Dynamic Tabular NAS):** All patients are first evaluated by a stacking ensemble dynamically optimized by Optuna. It processes EHR telemetry to resolve clear-cut cases instantly:
  * **XGBoost:** Provides precise residual regularization to prevent overfitting on noisy medical telemetry.
  * **LightGBM:** Utilizes leaf-wise growth optimized to capture non-linear interactions across high-dimensional features.
  * **CatBoost:** Employs symmetrically built trees to heavily penalize over-segmentation on severe minority mortality classes.
  * **TabNet:** Applies sequential attention mechanisms to mimic human feature selection and reasoning.
  * **BernoulliNB (Red Flag Expert):** Acts as a high-sensitivity binary filter for critical clinical thresholds.
* **Stage 2 — Intermediate Phenotype Routing:** Standard decision trees split feature space along strict right angles, failing on complex borderline patients. An **Isolation Forest** isolates out-of-distribution anomalies, while a **Gaussian Mixture Model** performs soft clustering to route ambiguous cases to specialized mathematical paradigms.
* **Stage 3 — Targeted Specialist Models:** High-risk, ambiguous cases are delegated to a grid-searched pool of experts:
  * **HistGradientExpert:** Optimized for residual patterns and complex decision boundaries on highly dense feature subsets.
  * **NonLinear SVM:** Projects borderline cases into higher-dimensional spaces using a balanced kernel to find smooth geometric separation where standard trees break down.
  * **NonLinear MLP:** Models continuous, dense interactions of metabolic collapse for patients in profound neurogenic shock.
  * **ClinicalProfileExpert (LogReg + KNN):** A soft voting ensemble combining calibrated linear risk weights with the physiological distance to historical records.
* **Stage 4 — On-Demand Spatial Vision:** Reserved for the most complex cases, deploying deep learning backbones to extract structural pathologies directly from raw CT scans:
  * **ResNet50 + CRNN:** Multilabel classification of hemorrhages and tracking the 3D spatial dynamics of mass lesions across slices.
  * **DenseNet 3D + ViT:** Captures global spatial context across volumetric slices to instantly detect calvarial and skull base fractures.
  * **UNet ➔ EfficientNet:** Precision extraction of the corpus callosum to detect microscopic diffuse axonal injuries.
* **Stage 5 — Holistic Judge & Longitudinal Monitoring:** Aggregates visual and clinical logits into a final Meta-Judge that actively optimizes for maximum Negative Predictive Value (**~95% NPV**). Continuously supported by **Graph Neural Networks** modeling pharmacological interactions and a **Bayesian Trajectory Monitor** tracking physiological risk decay over time.

</details>

---

#### 2. [Handwritten Text Recognition](https://github.com/MarekCackowski/Handwritten-Text-Recognition)
> **Asynchronous HTR pipeline designed to process challenging, degraded handwriting through a multi-stage cognitive cascade simulating human visual and cognitive reading processes.**

```text
[Word Image] ──▶ Stage 1: ResNet-CRNN (Word-Level CTC Baseline)
                      ├── (Confident Word — Skip CapsNet) ──────────────────────────────┐
                      │ (High Variance / Ambiguous)                                     │
                      ▼                                                                 │
                 Stage 2: Capsule Networks + TTA (Uncertain Char Crops)                 │
                      │ (Refined Chars + <unc> Tags)                                    │
                      ▼                                                                 │
                 Stage 3: ByT5 + Semantic Veto (Text Extracted from Previous Stages) ◀──┘
                      │
                      ▼
                 [Final Verified Transcript]
```

- **Core Impact:** Bridges word-level optical recognition, targeted geometric verification on uncertain character crops, and full-sentence language modeling inside a decoupled microservice architecture.

<details>
<summary><b>Deep Dive: Vision-Language Cascade & Infrastructure</b></summary>

- **Stage 1 — Word-Level Optical Recognition (`ResNet-CRNN`):** Processes whole word images through a CTC baseline while an `UncertaintyGatingAgent` and Monte Carlo Dropout estimate predictive variance—allowing confident words to bypass heavy geometric routing.
- **Stage 2 — Uncertain Character Crop Verification (`Capsule Networks + TTA`):** Routes only high-variance, ambiguous character crops to **Capsule Networks** with Test-Time Augmentation for fine-grained geometric verification and `<unc>` span tagging.
- **Stage 3 — Full-Sentence Semantic Correction (`ByT5 + Semantic Veto`):** Consolidates all word outputs into full handwritten sentences for context-aware linguistic correction by a fine-tuned, character-level **ByT5 Transformer**, guarded by a semantic veto against hallucinations.
- **Scalable Infrastructure:** Decouples the API from heavy GPU inference. A lightweight **FastAPI** gateway delegates prediction tasks to **Celery** workers using **Redis**, which also handles real-time WebSocket status streams. Document storage and generated PDFs are managed by **MinIO**, while user feedback and authentication are persisted securely in **MongoDB**.
- **Synthetic Generation & Polish Support:** Utilizes custom synthetic data generation pipelines to master complex Polish diacritics and simulate severe ink degradation.

</details>
