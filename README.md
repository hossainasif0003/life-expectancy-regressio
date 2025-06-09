# life-expectancy-regressio
Regression analysis and visualization on Life Expectancy dataset using Python (Linear, Multiple, Polynomial Regression with scikit-learn and seaborn).

# Life Expectancy Regression Analysis

**Author:** Asif Hossain  
**Date:** 2025-06-09

---

## Project Description

This project analyzes a Life Expectancy dataset using multiple regression techniques to model and predict life expectancy based on socioeconomic factors.

The main steps in this analysis include:

- Data loading and inspection
- Handling missing data with mean or median imputation based on skewness
- Data cleaning and duplicate removal
- Feature engineering and correlation analysis
- Visualization with bar charts, scatter plots, histograms, and boxplots
- Regression modeling using:
  - Simple Linear Regression
  - Multiple Linear Regression
  - Polynomial Regression (1D and multi-feature)
  - Polynomial Regression with pipeline including scaling
- Model evaluation using R² Score and Mean Squared Error (MSE)
- Comparison of model performances

---

## Dataset

The dataset file (`Life Expectancy Data.csv`) contains data such as:

- Country
- GDP
- Population
- Income composition of resources
- Life expectancy
- Total expenditure
- Schooling
- And other relevant variables

> **Note:** Update the dataset filepath in the script to match your local file location.

---

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
