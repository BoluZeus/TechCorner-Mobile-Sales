# TechCorner Mobile Sales Analysis

## Table of Content

- [Project Overview](#project-overview)
- [Data sources](#data-sources)
- [Tool Used](#tool-used)
- [Technology Used](#technology-used)
- [Data Cleaning and Preparation](#data-cleaning-and-preparation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Machine Learning Models](#machine-learning-models)
- [Data Analysis](#data-analysis)

### Project Overview

This project analyzes 10 months of mobile sales data from a retail shop in Bangladesh to uncover customer behavior, sales trends, and marketing effectiveness. The dataset includes information on customer demographics, buying patterns, Facebook marketing reach, and returning customers.

### Data sources

The primary dataset used for this analysis is the "TechCorner_Sales_update.csv", it contains detailed information about each sale made by TechCorner over the last 10 months.

### Tool Used

- Jupyter notebook - Cleaning, Data Analysis, Creating reports

### Technology Used 
- Python
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Scikit-Learn


### Data Cleaning and Preparation

Before analysis, the dataset was cleaned and preprocessed to ensure accuracy:
1. Data Loading and Inspection.
2. Handled missing values – Removed or imputed missing data in customer details and sales records.
3. Corrected inconsistencies – Standardized categorical variables (e.g., location names, gender labels).
4. Converted data types – Ensured numerical columns (e.g., prices, ages) were in the correct format.
5. Created new features – Extracted month from date.

These steps ensured the dataset was clean, structured and ready for analysis.

### Exploratory Data Analysis

- Sales Trends – Monthly sales fluctuations and location-based demand.
- Customer Behavior – Age, gender, and buying habits of mobile customers.
- Marketing Impact – Evaluated Facebook promotions vs. organic sales.
- Top-Selling Phones – Identified the most popular mobile brands & models.
- Pricing Trends – Analyzed price distribution and customer spending patterns.


### Machine Learning Models

- Returning Customer Prediction – Built a model to predict if a customer is likely to return.
- Customer Segmentation – Applied K-Means clustering to group customers by purchasing behavior.

### Data Analysis(Numerical Breakdown and Visualization)

1. Count of sales per month
   
```python

monthly_sales = df.groupby("Month").size()
print(monthly_sales)
```
```
df["Month"] = df["Date"].dt.to_period("M")

plt.figure(figsize=(10,5))
df.groupby("Month").size().plot(kind="line", marker="o", color="blue")

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Number of Sales")
plt.grid(True)
plt.show()
```
![GitHub image 1](https://github.com/user-attachments/assets/1dfc0b66-ac91-416b-997e-bc0d6b514eb8)

  




