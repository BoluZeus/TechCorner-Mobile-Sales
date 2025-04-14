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


### Data Analysis

#### Numerical Breakdown and Visualization

#### 1. Count of sales per month
   
```python

monthly_sales = df.groupby("Month").size()
print(monthly_sales)
```
![github image 1b](https://github.com/user-attachments/assets/0ec8516c-60b4-43b3-a88c-c5baa34a2c6f)


```python
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

Findings:
- Sales started at 153 in May 2024 and grew steadily over the months.
- By September 2024, sales reached 916, showing an upward trend.

Insight: The shop experienced consistent growth in sales, suggesting effective marketing or increased demand over time.

Observations:
- Sales increased steadily from May to September 2024.
- There were no major drops, indicating a consistent demand for mobile phones.



#### 2. Sales Distribution by Customer Location

```python
location_sales = df["Cus. Location"].value_counts()
print(location_sales)
```

![github image 2](https://github.com/user-attachments/assets/218108c9-80b7-4a95-b6e9-5e740e85b184)


```python
plt.figure(figsize=(6,5))
sns.countplot(x=df["Cus. Location"], palette="coolwarm", order=df["Cus. Location"].value_counts().index)

plt.title("Sales by Customer Location")
plt.xlabel("Customer Location")
plt.ylabel("Number of Sales")
plt.xticks(rotation=45)
plt.show()
```

![github image 2b](https://github.com/user-attachments/assets/0d000c5d-341b-443b-80f2-83a5f055602a)

Findings:
- Most customers came from Outside Rangamati (3000 sales).
- Rangamati Sadar (2972 sales) and Inside Rangamati (2899 sales) had almost equal sales.

Insight: The shop has strong reach outside the local region, meaning demand exists beyond Rangamati. Expanding delivery options could further boost sales.

#### 3. Age Distribution of Customers 

```python
age_stats = df["Age"].describe()
print(age_stats)
```
![github image 3](https://github.com/user-attachments/assets/1e6ba9de-24c0-4f0f-8e0e-3cc8c4042678)


```python
plt.figure(figsize=(8,5))
sns.histplot(df["Age"], bins=20, kde=True, color="skyblue")

plt.title("Age Distribution of Customers")
plt.xlabel("Age")
plt.ylabel("Count")
plt.grid(True)
plt.show()
```

![github image 3b](https://github.com/user-attachments/assets/99888ee6-1c9d-4f62-a38c-c863b6c8f697)

Findings:
- The average age of buyers is 34 years.
- Most customers are between 26 and 42 years old (middle 50% range).
- The youngest buyer is 18, and the oldest is 50.

Insight: Marketing efforts should focus on the 26-42 age group, as they represent the largest customer base.


