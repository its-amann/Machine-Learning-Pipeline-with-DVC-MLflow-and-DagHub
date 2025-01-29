# Machine Learning Pipeline with DVC, MLflow and DagHub 🚀

[![DVC](https://img.shields.io/badge/DVC-Data%20Version%20Control-945DD6?style=for-the-badge&logo=dvc)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Machine%20Learning%20Lifecycle-0194E2?style=for-the-badge&logo=mlflow)](https://mlflow.org/)
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![DagsHub](https://img.shields.io/badge/Tracking%20with-DagsHub-E67701?style=for-the-badge)](https://dagshub.com/)

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Pipeline Architecture](#️-pipeline-architecture)
- [DagHub Integration](#-daghub-integration)
- [Data Management](#-data-management)
- [Model Training](#-model-training)
- [Model Evaluation](#-model-evaluation)
- [Setup Instructions](#-setup-instructions)
- [Usage Guide](#-usage-guide)


## 🎯 Project Overview

This project implements a robust machine learning pipeline using DVC (Data Version Control) for data and pipeline management, MLflow for experiment tracking, with all experiments and artifacts stored and monitored on DagHub. The pipeline includes data preprocessing, model training with hyperparameter tuning, and model evaluation stages.

> **Dataset**: The project uses the Diabetes prediction dataset, which contains various health metrics to predict diabetes occurrence. The dataset is version controlled using DVC to ensure reproducibility.

<details>
<summary>Project Structure 📁</summary>

```
dvc_data_pipeline/
├── data/
│   ├── raw/        # Original diabetes dataset
│   └── preprocess/ # Preprocessed data
├── models/         # Trained model artifacts
├── src/
│   ├── preprocess.py  # Data preprocessing
│   ├── train.py       # Model training with MLflow tracking
│   └── evaluate.py    # Model evaluation
├── _asserts/       # Documentation assets
├── dvc.yaml        # DVC pipeline configuration
├── params.yaml     # Model parameters
└── requirements.txt
```
</details>

## 🏗️ Pipeline Architecture

The ML pipeline is orchestrated using DVC, which manages both the data versioning and the model training process. All experiments are automatically tracked in DagHub through MLflow integration.

![Data Pipeline](_asserts/data%20pipeline%20photo%20created%20by%20dagshub.png)

### Pipeline Stages:

1. **Preprocessing** 🔄
   - Input: Raw diabetes dataset from `data/raw/data.csv`
   - Process: Data cleaning and preparation
   - Output: Processed data in `data/preprocess/data.csv`

2. **Training** 🎯
   - Input: Processed data
   - Process: RandomForest model training with GridSearchCV
   - Parameters Tuned:
     - n_estimators: [100, 200]
     - max_depth: [5, 10, None]
     - min_samples_split: [2, 5, 10]
     - min_samples_leaf: [1, 2, 4]
   - Output: Trained model in `models/model.pkl`
   - MLflow Tracking: All hyperparameters, metrics, and artifacts are logged to DagHub

3. **Evaluation** 📊
   - Input: Trained model and test data
   - Process: Model performance evaluation
   - Output: Performance metrics tracked in MLflow on DagHub

### DVC Pipeline Configuration

![DVC YAML](_asserts/dvc%20yaml%20file.png)

## <a name="-daghub-integration"></a>🌐 DagHub Integration

This project leverages DagHub for:
- Experiment tracking through MLflow
- Model registry and versioning
- Centralized storage of metrics and artifacts
- Collaborative experimentation
![image](https://github.com/user-attachments/assets/4fae734f-6a89-4fbd-b462-c59b96174773)

The MLflow tracking server is configured to use DagHub's servers:
```python
os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/its-amann/dvc_data_pipeline.git"
os.environ["MLFLOW_TRACKING_USERNAME"] = "username"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "your-token"
```

All experiments, including:
- Hyperparameters
- Metrics (accuracy, confusion matrix)
- Model artifacts
- Classification reports
Are automatically logged to DagHub and can be viewed in the DagHub dashboard.

## <a name="-data-management"></a>📊 Data Management

The data is version controlled using DVC, ensuring reproducibility and traceability.

![Data Overview](_asserts/viewing%20the%20data.png)

### Parameters Configuration

Model and pipeline parameters are managed through `params.yaml`:

![Parameters](_asserts/params%20yaml%20file%20photo.png)

## <a name="-model-training"></a>🤖 Model Training

The training process integrates advanced features:
- Data splitting (80/20 train/test)
- Hyperparameter tuning using GridSearchCV
- RandomForest classifier training
- Automatic metric logging to DagHub via MLflow
- Model registry integration for version control

### Model Artifacts

![Model Artifacts](_asserts/model%20artifacts%20photo.png)

### Training Dashboard on DagHub

![Training Dashboard](_asserts/dashboard%20of%20the%20first%20experiment%20giving%20overview%20about%20the%20experiment.png)

## <a name="-model-evaluation"></a>📈 Model Evaluation

### Performance Metrics

The model's performance metrics are automatically logged to DagHub:

![Classification Report](_asserts/model%20classification%20report.png)

### Confusion Matrix

![Confusion Matrix](_asserts/model%20confusion%20matrix.png)

### MLflow Tracking on DagHub

All experiments are tracked and can be compared on DagHub:

![Model Parameters and Accuracy](_asserts/seeing%20the%20parameters%20and%20the%20accuracy%20of%20the%20model.png)

## <a name="-setup-instructions"></a>🚀 Setup Instructions

1. Clone the repository:
```bash
git clone https://dagshub.com/username/dvc_data_pipeline.git
cd dvc_data_pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure DagHub Integration:
   - Create an account on [DagHub](https://dagshub.com)
   - Create a new repository
   - Get your authentication token from Settings → Applications
   - Configure environment variables:
```bash
export MLFLOW_TRACKING_URI="https://dagshub.com/your-username/dvc_data_pipeline.mlflow"
export MLFLOW_TRACKING_USERNAME="your-username"
export MLFLOW_TRACKING_PASSWORD="your-token"
```

4. Set up DVC:
```bash
dvc remote add origin https://dagshub.com/your-username/dvc_data_pipeline.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user your-username
dvc remote modify origin --local password your-token
```

## <a name="-usage-guide"></a>📖 Usage Guide

1. Pull the data using DVC:
```bash
dvc pull
```

2. Run the complete pipeline:
```bash
dvc repro
```

3. View results:
   - Local: `dvc metrics show`
   - DagHub: Visit your repository on DagHub to view:
     - Experiment tracking
     - Model metrics
     - Artifacts
     - Training history

### 🔄 Pipeline Execution

To run individual stages:
```bash
# Preprocessing
dvc run -n preprocess python src/preprocess.py

# Training
dvc run -n train python src/train.py

# Evaluation
dvc run -n evaluate python src/evaluate.py
```

### 📊 Viewing Experiments

1. On DagHub:
   - Go to your repository on DagHub
   - Navigate to the "Experiments" tab
   - View all runs, compare metrics, and download artifacts

2. Local MLflow UI (if needed):
```bash
mlflow ui --backend-store-uri https://dagshub.com/your-username/dvc_data_pipeline.mlflow
```

---

<p align="center">
  Made with ❤️ using DVC, MLflow and DagHub
</p>
