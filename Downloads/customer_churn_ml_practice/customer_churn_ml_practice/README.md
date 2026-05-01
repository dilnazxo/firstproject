# Customer Churn ML Practice Project

## Project Title
Predictive Analytics System for Customer Churn Risk Assessment

## Description
This project is a machine learning prototype developed for industrial practice.  
The goal is to predict whether a customer is likely to churn based on structured customer data.

The project includes:
- data loading and inspection;
- data cleaning;
- exploratory data analysis;
- preprocessing pipeline;
- model training;
- model evaluation;
- feature importance analysis;
- report-ready figures.

## Technologies
- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- seaborn
- Jupyter Notebook
- GitHub
- LaTeX / Overleaf

## Folder Structure

```text
customer-churn-ml-practice/
│
├── data/
│   └── customer_churn.csv
│
├── notebooks/
│   └── churn_prediction.ipynb
│
├── src/
│   └── train_model.py
│
├── figures/
│   ├── ml_pipeline.png
│   ├── feature_importance.png
│   ├── churn_distribution.png
│   ├── correlation_heatmap.png
│   ├── notebook_screenshot_placeholder.png
│   └── github_repository_placeholder.png
│
├── report_overleaf/
│   └── main.tex
│
├── presentation_outline/
│   └── presentation_script.md
│
├── requirements.txt
└── README.md
```

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the training script:

```bash
python src/train_model.py
```

3. Or open the notebook:

```bash
jupyter notebook notebooks/churn_prediction.ipynb
```

## Dataset
For the final version, use a public customer churn dataset such as:
- Telco Customer Churn dataset from IBM / Kaggle.

Place the CSV file inside the `data/` folder and name it:

```text
customer_churn.csv
```

The script includes fallback synthetic data generation, so it can run even before the real dataset is added.

## Report
The LaTeX report template is located in:

```text
report_overleaf/main.tex
```

Upload this file and the `figures/` folder to Overleaf.

## Evidence to Add
For final submission, add:
- screenshot of Jupyter Notebook output;
- screenshot of GitHub repository;
- generated model evaluation table;
- feature importance chart;
- GitHub repository link;
- workplace supervisor information.
