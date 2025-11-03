# Animal Classification Model

<div align="center">
  <img src="https://img.shields.io/github/stars/mlops-hub/classifier-model.svg?style=for-the-badge" alt="GitHub Stars" />
  <img src="https://img.shields.io/github/forks/mlops-hub/classifier-model.svg?style=for-the-badge" alt="GitHub Forks" />
  <img src="https://img.shields.io/github/contributors/mlops-hub/classifier-model.svg?style=for-the-badge" alt="Contributors" />
  <img src="https://img.shields.io/github/last-commit/mlops-hub/classifier-model/main.svg?style=for-the-badge" alt="Last Commit" />
  <img src="https://img.shields.io/badge/python-3.12.x-blue?style=for-the-badge" alt="Python Version" />
</div>

<hr />

## Table of Contents

- [Overview](#overview)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Libraries & Tools](#libraries-and-tools)
- [Model](#model)
- [Setup & Installation](#setup--installation)
    - [Clone repo](#step-1-clone-the-repository)
    - [Create Virtual Environment](#step-2-create-virtual-environment)
    - [Install Dependencies](#step-3-install-dependencies)
    - [Training the Model Pipeline](#step-4-run-the-model-workflow-step-by-step)
        - [Notebook Ingestion](#1-run-notebookingestion)
        - [Run Data Pipeline](#2-run-data-pipeline)
        - [Run Feast Server](#3-run-feast)
        - [Run MLflow](#4-run-mlflow)
        - [Run Model Pipeline](#5-run-model-pipeline)
    - [Deploy model in KServe local](#step-5-deploy-model-in-kserve-locally)
    - [Testing/Prediction](#step-6-testingprediction)
    - [Execute Frontend](#step-7-run-frontend)
- [MLOPs Monitoring](#mlops-monitoring)
- [Contribution](#contribution)
- [References](#references)


## Overview

This project is an **"Animal Classification System"** built using machine learning. It predicts the class of an animal (e.g., Mammal, Bird, Fish, etc.) based on its features. The model is trained on the UCI Zoo dataset and related class information


## Datasets

[UCI Zoo Dataset](https://archive.ics.uci.edu/dataset/111/zoo)

[Kaggle Zoo Animal Classification](https://www.kaggle.com/datasets/uciml/zoo-animal-classification/data)

From kaggle's zoo dataset, I have added some missing values to practice EDA and cleaning data.

The main dataset used is [zoo.csv](./datasets/raw/zoo.csv) which contains features such as:


| Feature      | Description           |
|--------------|-----------------------|
| hair         | Has hair              |
| feathers     | Has feathers          |
| eggs         | Lays eggs             |
| milk         | Produces milk         |
| airborne     | Can fly/airborne      |
| aquatic      | Lives in water        |
| predator     | Is a predator         |
| toothed      | Has teeth             |
| backbone     | Has backbone          |
| breathes     | Breathes air          |
| venomous     | Is venomous           |
| fins         | Has fins              |
| legs         | Number of legs        |
| tail         | Has tail              |
| domestic     | Is domesticated       |
| catsize      | Cat-sized             |
| class_type   | Class type (numeric)  |
| class_name   | Class name (label)    |


## Project Structure

```bash
mlflow-training-pipeline
    |__ requirements.txt                   # install dependency pacakges
    |__ feature_store/
        |__ feature_names.pkl              # save feature_names for testing|__ feature_store/
        |__ features.py                    # setup registry in feast
        |__ data/                          # save preprocessd.parquet file
    |__ flask-app/
        |__ app.py
        |__ static/*
        |__ templates/*
        |__ .env
    |__ ml/
        |__ data/
            |__ raw/*                      # original datasets              
            |__ /*                         # datasets created when you run code            
        |__ logs/                          # logs for hyperparamter tuning values
        |__ notebooks/                     # data ingestion, eda, experiment with different models, etc..
        |__ src/
            |__ data_piepline/*.py         # data_pipeline folder
            |__ model_pipeline/*.py        # model_pipeline folder
        |__ kserve/                        # kserve setup
            |__ pediction.py           
        |__ tests/                         # to test model
    |__ monitoring/                        # monitoring scripts
        |__ scripts/*
        |__ db_logs/*                      
        |__ inference_logger.db
        |__ live_data.db
        |__ evidently_monitor/
            |__ index.py
            |__ *.py

```


## Libraries and Tools

- **Machine Learning**: scikit-learn
- **Type of Machine Learning**: Supervised ML
- **Visual Charts**: matplotlib, seaborn
- **data validation**: pandera
- **Save model**: joblib
- **DEployment**: kserve
- **Metrics**: prometheus-client


## Model

- **ML Algorithm**: Logistic Regression 
- **Evaluation**: Accuracy, classification report, confusion matrix
- **Output**: Predicted animal class
- **Metrics**: Prometheus, Evidently AI (data drift, prediction drift, data quality, classifcation summary)


## Setup & Installation

### Step-1: Clone the repository

```bash
git clone https://github.com/mlops-hub/classifier-model.git
cd mlflow-training-pipeline
```

### Step-2: Create Virtual Environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Mac and Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step-3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step-4: Run the Model Workflow Step-by-Step

#### 1. Run notebook/ingestion

```bash
cd mlflow-training-pipeline
cd ml
cd notebooks/ingestion
```

- Click 'run all' in tab
- It saves merged dataset at [`data/ingestion/*.csv`](./ml/data/ingestion//merged_df.csv). This is dataset used for ml pipelines.


#### 2. Run Data Pipeline

```bash
cd mlflow-training-pipeline
python -m ml.src.data_pipeline.index
```

#### 3. Run Feast
To store preprocessed-dataset. 
On new terminal, go to [`feature_repo/`](./feature_repo/feature_store.yaml) folder.

```bash
cd mlflow-training-pipeline
cd feature_repo
```

**Export environment variables**

```bash
export POSTGRES_USER=<postgres-name>
export POSTGRES_PASSWORD=<postgres-password>
export POSTGRES_HOST=<postgres-host>
export POSTGRES_PORT=<postgres-port>
export POSTGRES_DB=<postgres-db-name>
export REDIS_HOST=<redis-host>
export REDIS_PORT=<redis-port>
export REDIS_PASSWORD=<redis-password>
```

**Execute the code**

```bash
feast apply

# feast materialize <start-date> <end-date>
feast materialize 2025-06-01T00:00:00 2025-10-10T23:59:59
```

**(Optional) To run feast locally**

```bash
python main.py
```

#### 4. Run MLflow

If Mlflow is running in cloud. Skip this step.

```bash
cd mlfow-training-pipeline
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

#### 5. Run model-pipeline

Model pipeline gets data from feast, use feast credentials.

```bash
export POSTGRES_USER=<postgres-name>
export POSTGRES_PASSWORD=<postgres-password>
export POSTGRES_HOST=<postgres-host>
export POSTGRES_PORT=<postgres-port>
export POSTGRES_DB=<postgres-db-name>
export REDIS_HOST=<redis-host>
export REDIS_PORT=<redis-port>
export REDIS_PASSWORD=<redis-password>
```

If mlflow is running in cloud (aws/azure), use aws/azure credentials.

```bash
# bash
export AZURE_STORAGE_ACCOUNT_NAME=<account-name>
export AZURE_STORAGE_ACCESS_KEY=<access-key>

# powershell
$env:AZURE_STORAGE_ACCOUNT_NAME=<account-name>
$env:AZURE_STORAGE_ACCESS_KEY=<access-key>
```

**Execute the code**

```bash
python -m ml.src.model_pipeline.index
```

#### step-5: Deploy Model in KServe locally

```bash
cd ml/kserve
python prediction.py
```

#### Step-6: Testing/Prediction

Run [`inference_test.py`](./ml/tests/inference_test.py) to make predictions. If 'animal' is not found, you will be prompted to enter animal features, and the model will predict the class.


```bash
cd mlflow-training-pipeline
python ml.tests.inference_test
```

#### Step-7: Run Frontend
A separate virtual environment is created for [frontend](./frontend).

```bash
cd frontend

# for windows
python -m venv venv
venv/Scripts/activate

# for macOS and linux
python3 -m venv venv
source venv/Scripts/activate
```

**Execute the code**

```bash
python app.py
```

## MLOps Monitoring

[Monitoring](./monitoring/) helps track the performance of your ML model, monitor important metrics, and keep an eye on the underlying system infrastructure.

### Running the `scripts/` Folder

When you run the data pipeline and model pipeline, the reference data and live data are automatically stored in a SQLite3 database.

However, the `class_name` column in the live_data table may contain `NULL` values.

This happens when certain animals are not found in the Feast feature store, which is used to save the reference dataset.

In such cases, predictions are made using the newly extracted features.

To address this, you need to update and run the **[feedback-loop](./monitoring/scripts/feedback_loop.py)** script to populate the missing `class_name` values in the database.

```bash
cd monitoring/scripts

python -m feedback_loop.py
```

### Running metrics in Evidently

Setup Evidently cloud by signning or logging in [Evidently](https://docs.evidentlyai.com/introduction).

Generate Access Token and save the configuration in `.env` file.

```txt
EVIDENTLY_URL=https://app.evidently.cloud
EVIDENTLY_TOKEN=<access-token>
```

Go to [evidently_monitor/](./monitoring/evidently_monitor/) and run the pipeline

```bash
cd monitoring

python -m evidently_monitor.index
```

This will save the reports automatically in `reports/` fodler based on date and time.

You can also see the metrics in Evidently cloud via your account.


## Contribution
Please read our [Contributing Guidelines](./CONTRIBUTION.md) before submitting pull requests.


## License
This project is under [MIT Licence](./LICENCE) support.
