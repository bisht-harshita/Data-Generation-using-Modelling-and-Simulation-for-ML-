# Data Generation and ML Analysis using SimPy

## Overview
This project demonstrates the generation of a synthetic dataset using Modelling and Simulation (SimPy) and compares various Machine Learning models trained on this data.

## Methodology

### 1. Simulation (SimPy)
We modeled a Call Center / Queueing System:

Process: Customers arrive and queue for service agents.

Inputs (Parameters):
- Num_Agents: Number of service agents available (2-20).
- Arrival_Interval: Average time between customer arrivals (1-10 mins).
- Service_Time: Average time for an agent to serve a customer (5-30 mins).

Output (Target):
- Average_Wait_Time: The mean waiting time for customers in the queue per simulation run.

Data Generation:
- We ran 1000 independent simulations.
- Each simulation had randomly sampled input parameters.
- This created a diverse dataset representing various traffic and resource load scenarios.

### 2. Machine Learning Pipeline

Preprocessing:
- Dataset: simulation_data.csv
- Split: 80% Train, 20% Test
- Scaling: Standardization (StandardScaler)

Models Evaluated:

Linear Models:
- Linear Regression
- Ridge
- Lasso

Trees & Ensembles:
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- AdaBoost

Other:
- Support Vector Regressor (SVR)
- K-Nearest Neighbors (KNN)

Evaluation Metric:
- R2 Score (Coefficient of Determination)
- RMSE

## Results

Top Performing Models:

| Rank | Model             | R2 Score | RMSE  |
|------|------------------|----------|-------|
| 1    | Random Forest    | ~0.93    | 12.95 |
| 2    | XGBoost          | ~0.93    | 13.08 |
| 3    | Gradient Boosting| ~0.92    | 13.63 |

Analysis:

- Random Forest and XGBoost performed best, capturing the non-linear relationship between resource availability (agents) and wait times (queue dynamics).
- Linear models performed poorly (~0.45 R2), confirming that the system behavior (queueing theory constraints) is highly non-linear.

## Files

- solution.ipynb: Jupyter Notebook with complete code.
- data_generation.py: Python script for simulation.
- ml_analysis.py: Python script for model training.
- simulation_data.csv: The generated synthetic dataset.
- model_results.csv: Detailed metrics of all trained models.
