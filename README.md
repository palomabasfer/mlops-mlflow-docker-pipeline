# Production MLOps Pipeline: MLflow, Docker & GitHub Actions CI/CD

![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ed.svg)
![CI/CD](https://img.shields.io/badge/GitHub%20Actions-Automated%20Pipeline-green.svg)
![Drift Monitoring](https://img.shields.io/badge/Monitoring-KS--Test%20Drift-orange.svg)
![Build](https://img.shields.io/badge/Build-Passing-brightgreen.svg)

## 📌 Problem Overview
Deploying enterprise Machine Learning models into production requires continuous integration, experiment reproducibility, model versioning, and real-time data drift monitoring. This repository implements an end-to-end **MLOps Framework** integrating **MLflow tracking**, **Docker containerization**, **Kolmogorov-Smirnov statistical drift detection**, and automated **GitHub Actions CI/CD**.

---

## 🛠️ Key Technologies
- **MLOps Experiment Tracking**: MLflow tracking server logging hyperparameters, RMSE/R2 metrics, and model artifacts.
- **Data Drift Detection**: Automated KS-statistical distribution monitoring to detect concept drift in production features.
- **Containerization**: Optimized `Dockerfile` & multi-container `docker-compose` orchestration.
- **CI/CD Automation**: GitHub Actions workflow validating code quality, executing unit tests, and building containers automatically on every push.

---

## 📐 Architecture Diagram

```text
+-----------------------+     +-------------------------------+     +---------------------------+
| Automated Data Stream | --> | MLflow Experiment Tracker     | --> | Gradient Boosting Model   |
| (Scaler & Generator)  |     | (Parameters & Artifacts)      |     | (RMSE & R2 Evaluation)    |
+-----------------------+     +-------------------------------+     +---------------------------+
                                                                                  |
                                                                                  v
+-----------------------+     +-------------------------------+     +---------------------------+
| GitHub Actions CI/CD  | <-- | Docker Container Build        | <-- | Statistical Drift Monitor |
| (Automated Testing)   |     | (docker-compose deployment)   |     | (KS-Test Feature Drift)   |
+-----------------------+     +-------------------------------+     +---------------------------+
```

---

## 🚀 Quickstart Guide

### 1. Run with Docker Compose
```bash
docker-compose up --build
```

### 2. Local Setup & Tests
```bash
git clone https://github.com/palomabasfer/mlops-mlflow-docker-pipeline.git
cd mlops-mlflow-docker-pipeline
pip install -r requirements.txt
pip install -e .
pytest
```

---

## 👤 Author
**Paloma Bas Fernández** — Double Degree in Mathematics & Computer Engineering (University of Seville)  
[GitHub Profile](https://github.com/palomabasfer)
