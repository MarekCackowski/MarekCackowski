# Marek Cackowski
**Junior Machine Learning Engineer | Data Scientist**

I specialize in designing, optimizing, and deploying Deep Learning architectures and classical ML models. I build smart pipelines that route typical cases to fast models, avoiding heavy computation where unnecessary.

### Tech Stack
<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit_Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/SQL-003B57?style=flat&logo=mysql&logoColor=white" />
  <img src="https://img.shields.io/badge/Redis-DC382D?style=flat&logo=Redis&logoColor=white" />
  <img src="https://img.shields.io/badge/MongoDB-4EA94B?style=flat&logo=mongodb&logoColor=white" />
  <img src="https://img.shields.io/badge/MinIO-C72026?style=flat&logo=minio&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white" />
</p>

* **[TBI Clinical Decision Support System](https://github.com/MarekCackowski/TBI)**  
  A multimodal triage architecture for ICU patients predicting physiological deterioration through a computationally optimized cognitive funnel:

  * **Initial Rapid Screening (Dynamic Tabular NAS):** All patients are first evaluated by a stacking ensemble dynamically optimized by Optuna. It processes EHR telemetry to resolve clear cut cases instantly:
    * **XGBoost:** Provides precise residual regularization to prevent overfitting on noisy medical telemetry.
    * **LightGBM:** Utilizes leaf wise growth optimized to capture non linear interactions across high dimensional features.
    * **CatBoost:** Employs symmetrically built trees to heavily penalize over segmentation on severe minority mortality classes.
    * **TabNet:** Applies sequential attention mechanisms to mimic human feature selection and reasoning.
    * **BernoulliNB (Red Flag Expert):** Acts as a high sensitivity binary filter for critical clinical thresholds.

  * **Intermediate Phenotype Routing:** Standard decision trees split feature space along strict right angles, failing on complex borderline patients. An **Isolation Forest** isolates out of distribution anomalies, while a **Gaussian Mixture Model** performs soft clustering to route ambiguous cases to specialized mathematical paradigms.

  * **Targeted Specialist Models:** High risk, ambiguous cases are delegated to a grid searched pool of experts:
    * **HistGradientExpert:** Optimized for residual patterns and complex decision boundaries on highly dense feature subsets.
    * **NonLinear SVM:** Projects borderline cases into higher dimensional spaces using a balanced kernel to find smooth geometric separation where standard trees break down.
    * **NonLinear MLP:** Models continuous, dense interactions of metabolic collapse for patients in profound neurogenic shock.
    * **ClinicalProfileExpert (LogReg + KNN):** A soft voting ensemble combining calibrated linear risk weights with the physiological distance to historical records.

  * **On Demand Spatial Vision:** Reserved for the most complex cases, deploying deep learning backbones to extract structural pathologies directly from raw CT scans:
    * **ResNet50 + CRNN:** Multilabel classification of hemorrhages and tracking the 3D spatial dynamics of mass lesions across slices.
    * **DenseNet 3D + ViT:** Captures global spatial context across volumetric slices to instantly detect calvarial and skull base fractures.
    * **UNet → EfficientNet:** Precision extraction of the corpus callosum to detect microscopic diffuse axonal injuries.

  * **Holistic Judge & Longitudinal Monitoring:** Aggregates visual and clinical logits into a final Meta Judge that actively optimizes for maximum Negative Predictive Value (~95% NPV). Continuously supported by **Graph Neural Networks** modeling pharmacological interactions and a **Bayesian Trajectory Monitor** tracking physiological risk decay over time.

* **[Handwritten Text Recognition](https://github.com/MarekCackowski/Handwritten-Text-Recognition)**  
  An asynchronous HTR pipeline designed to process challenging, degraded handwriting through a multi stage cognitive cascade simulating human visual and cognitive reading processes.
  
  * **Dynamic Uncertainty Routing:** The system employs an `UncertaintyGatingAgent` and Monte Carlo Dropout to estimate predictive variance. Confident character predictions fast track through a **ResNet+CRNN** baseline. Highly ambiguous characters are dynamically routed to **Capsule Networks** for rigorous geometric verification using TTA.
  * **Semantic Correction:** A fine tuned, character level **ByT5 Transformer** performs the final linguistic correction, analyzing the raw vision output alongside broader sentence context to fix OCR errors.
  * **Scalable Infrastructure:** The system architecture decouples the API from heavy GPU inference. A lightweight **FastAPI** gateway delegates prediction tasks to **Celery** workers using **Redis**, which also handles real time WebSocket status streams. Document storage and generated PDFs are managed by **MinIO**, while user feedback and authentication are persisted securely in **MongoDB**.
  * **Synthetic Generation & Polish Support:** Utilizes custom synthetic data generation pipelines to master complex Polish diacritics and simulate severe ink degradation.
