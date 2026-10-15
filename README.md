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

### Featured Projects
* **[TBI Clinical Decision Support System](https://github.com/MarekCackowski/TBI)**  
  An advanced multimodal triage architecture for ICU patients designed to predict physiological deterioration. The system operates through a multi stage pipeline:
  * **Vision Module:** Extracts spatial anomalies and clinical indicators from CT/MRI brain scans.
  * **Tabular & Time Series:** Analyzes EHR trajectories using an asymmetric tree ensemble and **GMM Phenotype Routers**.
  * **Drug Graphs:** Models complex pharmacological interactions and patient state changes using Graph Neural Networks.
  * **Multimodal Fusion:** Aggregates visual, clinical, and graph embeddings for a holistic, high precision prognosis.

* **[Handwritten Text Recognition](https://github.com/MarekCackowski/Handwritten-Text-Recognition)**  
  An asynchronous HTR pipeline designed to process challenging, degraded handwriting through a multi stage cognitive cascade simulating human visual and cognitive reading processes.
  * **Dynamic Uncertainty Routing:** The system employs an `UncertaintyGatingAgent` and Monte Carlo Dropout to estimate predictive variance. Confident character predictions fast track through a **ResNetCRNN** baseline. Highly ambiguous characters are dynamically routed to **Capsule Networks** for rigorous geometric verification using TTA. 
  * **Semantic Correction:** A fine tuned, character level **ByT5 Transformer** performs the final linguistic correction, analyzing the raw vision output alongside broader sentence context to fix OCR errors.
  * **Scalable Infrastructure (Backend):** The system architecture decouples the API from heavy GPU inference. A lightweight **FastAPI** gateway delegates prediction tasks to **Celery** workers using **Redis**, which also handles real time WebSocket status streams. Document storage and generated PDFs are managed by **MinIO**, while user feedback and authentication are persisted securely in **MongoDB**.
  * **Synthetic Generation & Polish Support:** Utilizes custom synthetic data generation pipelines to master complex Polish diacritics and simulate severe ink degradation.
---
