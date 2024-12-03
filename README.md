# Traffic Occupancy Rate Patterns Analysis

## Overview

This repository contains the code and resources for the project titled **"Analyzing Traffic Occupancy Rate Patterns on San Francisco Freeways."** The primary objective of this project is to analyze occupancy data collected from various sensors on San Francisco freeways to understand traffic flow patterns throughout the week.

## Motivation

Understanding traffic patterns is crucial for effective space management and congestion mitigation. By analyzing occupancy rates, we can identify peak traffic times and trends, which can inform infrastructure planning and improve traffic flow.

## Objectives

The main objectives of this project are:

-   Analyze daily and hourly occupancy patterns.
-   Identify average occupancy rates for each day of the week.
-   Highlight peak traffic times during weekdays.
-   Explore hourly trends to pinpoint rush hours.
-   Visualize the results for actionable insights.

## Technologies Used

This project utilizes several technologies and libraries:

-   **NumPy**: For numerical computations.
-   **Pandas**: For data manipulation and analysis.
-   **Matplotlib**: For creating visualizations.
-   **Seaborn**: For enhanced data visualizations.
-   **Scikit-learn**: For implementing machine learning models.
-   **Statsmodels**: For statistical modeling and time-series analysis.

## Data Description

The dataset used in this project consists of 15 months of daily occupancy rate data obtained from the California Department of Transportation PEMS website. The data includes:

-   Measurements taken every 10 minutes from various sensors.
-   Time series data representing each day with dimensions corresponding to sensors and time intervals.

The dataset has been preprocessed to remove anomalies and public holidays, resulting in a total of 440 time series for analysis.

## System Design

### Algorithms Selected

1. **Random Forest Classifier**: An ensemble learning method that reduces overfitting while increasing accuracy by aggregating multiple decision trees.
2. **Gradient Boosting Classifier**: A sequential model that corrects errors made by previous models, enhancing predictive performance.
3. **Voting Classifier**: Combines predictions from Random Forest and Gradient Boosting using a weighted soft voting approach.

### System Architecture

The architecture includes:

1. **Input Layer**: Accepts preprocessed datasets.
2. **Preprocessing Layer**: Extract features and standardizes features using `StandardScaler`.
3. **Model Layer**: Implements Random Forest, Gradient Boosting, and Voting Classifier.
4. **Evaluation Layer**: Computes accuracy and generates classification reports.

## Installation

To set up the project locally, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/veeravivekt/CMPE255-traffic-occupancy-patterns.git
    ```

2. Navigate into the project directory:

    ```bash
    cd CMPE255-traffic-occupancy-patterns
    ```

3. Initial project setup (includes downloading dataset, setting up virtual environment and installing dependencies):

    ```bash
    ./setup.sh
    ```

## Usage

To run the analysis, execute the following command in your terminal:

```bash
python run.py
```

This will preprocess the data, train the models, and generate visualizations of traffic patterns.

## Results

The model achieved an accuracy of `92.49%` in predicting traffic patterns across different days of the week. Key metrics include:

| Day       | Precision | Recall | F1-Score |
| --------- | --------- | ------ | -------- |
| Monday    | 1.00      | 1.00   | 1.00     |
| Tuesday   | 0.92      | 0.92   | 0.92     |
| Wednesday | 0.77      | 0.77   | 0.77     |
| Thursday  | 0.70      | 0.70   | 0.70     |
| Friday    | 0.84      | 0.73   | 0.78     |
| Saturday  | 0.89      | 0.93   | 0.91     |
| Sunday    | 0.91      | 1.00   | 0.95     |

## Contributors

This project was developed by:

-   Divya Varshini Suravaram (017506580)
-   Soumith Reddy Podduturi (016706612)
-   Veera Vivek Telagani (017503915)

## Presentation and Report

-   [Presentation](Presentation.pdf)
-   [Report](Report.pdf)
