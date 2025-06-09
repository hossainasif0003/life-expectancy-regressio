"""
Author: Asif Hossain
Date: 2025-06-09
Project: Life Expectancy Regression Analysis
Description: This script performs various regression analyses on the Life Expectancy dataset,
including data cleaning, visualization, and model comparison.
"""


from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# Load dataset
filepath = r"C:\Users\Asif Hossain\Desktop\Dataset\Life Expectancy Data.csv"
df = pd.read_csv(filepath)

# Inspect dataset
print(df.head())
print(df.info())

# Categorical and numerical columns
df_categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
df_numerical = df.select_dtypes(include='number').columns.tolist()

# Handle missing values
missing_cols = df.columns[df.isnull().any()]
df_skewed = df[missing_cols].skew()
for col in missing_cols:
    skew_val = df_skewed[col]
    if -0.5 < skew_val < 0.5:
        df[col].fillna(df[col].mean(), inplace=True)
        print(f"\nFilled missing values in '{col}' with MEAN (skew={skew_val:.2f})\n")
    else:
        df[col].fillna(df[col].median(), inplace=True)
        print(f"\nFilled missing values in '{col}' with MEDIAN (skew={skew_val:.2f})\n")

# Remove duplicates
df.drop_duplicates(inplace=True)

# Clean column names
df.columns = df.columns.str.strip()

# Feature Engineering
avg_life_exp_by_income = df.groupby('Income composition of resources')['Life expectancy'].mean()
avg_gdp_by_country = df.groupby('Country')['GDP'].mean()
avg_GDP_perExpenditure = df.groupby('Country')['Total expenditure'].mean()
key_indicators = ['GDP', 'Life expectancy', 'Total expenditure']
print("\nCorrelation Matrix:\n")
print(df[key_indicators].corr())

# Bar chart: Life Expectancy by Income Composition
avg_life_exp_by_income.plot(kind='bar')
plt.xlabel('Income composition of resources')
plt.ylabel('Average Life Expectancy')
plt.title('Average Life Expectancy by Income Composition of Resources')
plt.show()

# GDP per capita
df["GDP_per_Capita"] = df["GDP"] / df["Population"]

# Scatter plot (log scale)
plt.figure()
plt.style.use('dark_background')
X_log = np.log(df["GDP_per_Capita"])
Y = df["Life expectancy"]
sns.scatterplot(x=X_log, y=Y, marker="X", color='red')
plt.title('Log GDP per Capita vs Life Expectancy')
plt.show()

# Histogram
plt.figure()
plt.style.use("dark_background")
sns.histplot(df["Life expectancy"])
plt.title('Distribution of Life Expectancy')
plt.xlabel('Life Expectancy')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Boxplot by Income Group (discretized income composition)
df['IncomeGroup'] = pd.cut(df['Income composition of resources'],
                           bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                           labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
plt.figure()
sns.boxplot(data=df, x='IncomeGroup', y="Life expectancy", palette='colorblind')
plt.grid(True)
plt.show()

# ------------------------
# Simple Linear Regression
# ------------------------
lm = LinearRegression()
X1 = np.log(df[["GDP_per_Capita"]])
lm.fit(X1, Y)
y_predict1 = lm.predict(X1)
intercept1 = lm.intercept_
coef1 = lm.coef_
r2_1 = r2_score(Y, y_predict1)
mse_1 = mean_squared_error(Y, y_predict1)

print("\nSimple Linear Regression:\n")
print("Intercept:", intercept1)
print("Coefficient:", coef1)
print("R² Score:", r2_1)
print("Mean Squared Error:", mse_1)

plt.figure()
plt.scatter(X1, Y, color='blue')
plt.plot(X1, y_predict1, color='red')
plt.title("Linear Regression: Log GDP per Capita vs Life Expectancy")
plt.xlabel("Log GDP per Capita")
plt.ylabel("Life Expectancy")
plt.grid(True)
plt.show()

# ----------------------------
# Multiple Linear Regression
# ----------------------------
lm2 = LinearRegression()
X2 = df[['GDP_per_Capita', 'Total expenditure', 'Schooling', 'Population']]
lm2.fit(X2, Y)
y_predict2 = lm2.predict(X2)
r2_2 = lm2.score(X2, Y)
mse_2 = mean_squared_error(Y, y_predict2)

print("\nMultiple Linear Regression:\n")
print("Intercept:", lm2.intercept_)
print("Coefficients:", lm2.coef_)
print("R² Score:", r2_2)
print("Mean Squared Error:", mse_2)

# -----------------------
# Polynomial Regression 1D
# -----------------------
poly = PolynomialFeatures(degree=2)
X_poly1d = poly.fit_transform(X1)
lm3 = LinearRegression()
lm3.fit(X_poly1d, Y)
y_predict3 = lm3.predict(X_poly1d)
r2_3 = r2_score(Y, y_predict3)
mse_3 = mean_squared_error(Y, y_predict3)

print("\nPolynomial Regression (1D):\n")
print("R² Score:", r2_3)
print("Mean Squared Error:", mse_3)

plt.figure()
plt.scatter(X1, Y)
plt.plot(X1, y_predict3, color="red")
plt.title("Polynomial Regression (1D)")
plt.xlabel('Log GDP per Capita')
plt.ylabel('Life Expectancy')
plt.grid(True)
plt.show()

# ----------------------------
# Polynomial Regression (Multi)
# ----------------------------
poly2 = PolynomialFeatures(degree=2)
X_poly2d = poly2.fit_transform(X2)
lm4 = LinearRegression()
lm4.fit(X_poly2d, Y)
y_predict4 = lm4.predict(X_poly2d)
r2_4 = r2_score(Y, y_predict4)
mse_4 = mean_squared_error(Y, y_predict4)

print("\nPolynomial Regression (Multiple Features):\n")
print("R² Score:", r2_4)
print("Mean Squared Error:", mse_4)

# ----------------------------
# Polynomial Regression (Pipeline)
# ----------------------------
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('model', LinearRegression())
])
pipeline.fit(X2, Y)
y_predict5 = pipeline.predict(X2)
r2_5 = r2_score(Y, y_predict5)
mse_5 = mean_squared_error(Y, y_predict5)

print("\nPolynomial Regression with Pipeline:\n")
print("R² Score:", r2_5)
print("Mean Squared Error:", mse_5)

# -------------------
# Model Comparison
# -------------------
print("\n--- Model Comparison ---\n")
print(f"Simple Linear Regression        → R²: {r2_1:.4f}, MSE: {mse_1:.4f}\n")
print(f"Multiple Linear Regression      → R²: {r2_2:.4f}, MSE: {mse_2:.4f}\n")
print(f"Polynomial Regression (1D)      → R²: {r2_3:.4f}, MSE: {mse_3:.4f}\n")
print(f"Polynomial Regression (Multi)   → R²: {r2_4:.4f}, MSE: {mse_4:.4f}\n")
print(f"Polynomial Regression (Pipeline)→ R²: {r2_5:.4f}, MSE: {mse_5:.4f}\n")

# -------------------
# Optional: Residual Plot
# -------------------
residuals = Y - y_predict2
plt.figure()
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution (Multiple Linear Regression)")
plt.xlabel("Residuals")
plt.grid(True)
plt.show()

# -------------------
# Optional: DataFrame
# -------------------
comparison_df = pd.DataFrame({
    'Model': [
        'Simple Linear Regression',
        'Multiple Linear Regression',
        'Polynomial Regression (1D)',
        'Polynomial Regression (Multi)',
        'Polynomial Regression (Pipeline)'
    ],
    'R2 Score': [r2_1, r2_2, r2_3, r2_4, r2_5],
    'MSE': [mse_1, mse_2, mse_3, mse_4, mse_5]
})
print(comparison_df)
