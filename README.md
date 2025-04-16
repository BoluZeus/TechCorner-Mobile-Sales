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

#### 4. Gender Distribution of Buyers

```python
# Count of sales by gender(Numerical Breakdown)
gender_sales = df["Gender"].value_counts()
print(gender_sales)
```

![github image 4a](https://github.com/user-attachments/assets/2be63110-97a6-4a32-b1c4-11861fddecb7)

```python
plt.figure(figsize=(6,6))
df["Gender"].value_counts().plot.pie(autopct="%1.1f%%", colors=["lightblue", "lightcoral"])
plt.title("Gender Distribution of Buyers")
plt.ylabel("")  # Hide y-label
plt.show()
```

![github image 4b](https://github.com/user-attachments/assets/85178c94-1bdf-4cc2-9ecb-9dabdea0589f)


Findings:
- Females (F): 4453 buyers (50.2%)
- Males (M): 4418 buyers (49.8%)

Insight: Sales are almost equally split between genders.
Marketing strategies should cater to both male and female buyers equally.

#### 5. Top 10 Most Sold Mobile Names

```python
# Top 10 most sold mobile models(Numerical Breakdown)
top_10_mobiles = df["Mobile Name"].value_counts().nlargest(10)
print(top_10_mobiles)
```

![github image 5a](https://github.com/user-attachments/assets/9fe8baae-ff13-4642-9786-93e4c33fd1b5)

```python
plt.figure(figsize=(10,5))

top_mobiles = df["Mobile Name"].value_counts().nlargest(10)
sns.barplot(x=top_mobiles.index, y=top_mobiles.values, palette="coolwarm")

plt.title("Top 10 Most Sold Mobile Names")
plt.xlabel("Mobile Name")
plt.ylabel("Number of Units Sold")
plt.xticks(rotation=45)
plt.show()
```

![github image 5b](https://github.com/user-attachments/assets/2aadd22d-e279-4990-8d20-3d691a59a42d)

Findings:
- Moto G85 5G (8/128GB) is the best-selling model (560 units sold).
- Samsung, Xiaomi, Google Pixel, and iPhone models also dominate the top 10.

Insight:
- Budget 5G models like Moto G85 5G & Galaxy M35 5G are very popular.
- Premium models (Pixel 8 Pro, iPhone 16 Pro) also have strong demand.
- The shop should keep stock of these models and focus promotions on these brands.

#### 6. Sales Price Distribution

```python
# Statistics on selling price(Numerical Breakdown)
price_stats = df["Sell Price"].describe()
print(price_stats)
```

![github image 6a](https://github.com/user-attachments/assets/30228d4d-8bbf-4015-8927-82612164fed8)

```python
plt.figure(figsize=(8,5))
sns.boxplot(x=df["Sell Price"], color="lightcoral")

plt.title("Sales Price Distribution")
plt.xlabel("Sell Price (BDT)")
plt.show()
```

![github image 6b](https://github.com/user-attachments/assets/e6fbbd2f-6d6c-44b5-98bd-fcc2c70024dd)

Findings:
- Average phone price is 25,068 BDT.
- Most phones are priced between 17,466 BDT and 25,777 BDT (middle 50% range).
- The cheapest phone is 12,702 BDT, and the most expensive is 200,465 BDT.

Insight:
- The store mainly sells mid-range smartphones (17K - 25K BDT).
- There are high-end sales (200K BDT), suggesting demand for premium models.
- Pricing strategy should focus on both budget and premium segments.

#### 7. Facebook Marketing Effectiveness

```python
# Count of customers who came from facebook(Numerical Breakdown)
facebook_effectiveness = df["Does he/she Come from Facebook Page?"].value_counts()
print(facebook_effectiveness)
```

![github image 7a](https://github.com/user-attachments/assets/5ff74e2d-9ef8-423f-8152-3aed3499775e)


```python
plt.figure(figsize=(4,4))
sns.countplot(x=df["Does he/she Come from Facebook Page?"], palette="muted")

plt.title("Facebook Marketing Effectiveness")
plt.xlabel("Came from Facebook?")
plt.ylabel("Count")
plt.show()
```

![githu image 7b](https://github.com/user-attachments/assets/5d003d97-1e73-4d17-bc6f-6655194e8ed8)


#### 8. New vs Returning Customers


```python
# count New vs Returning Customers(Numerical Breakdown)
returning_customers = df["Did he/she buy any mobile before?"].value_counts()
print(returning_customers)
```

![github image 8a](https://github.com/user-attachments/assets/0fd1a972-a5b0-479d-8718-9c0d143a16b5)


```pyhton
plt.figure(figsize=(4,4))
sns.countplot(x=df["Did he/she buy any mobile before?"], palette="pastel")

plt.title("New vs. Returning Customers")
plt.xlabel("Returning Customer?")
plt.ylabel("Count")
plt.show()
```

![github image 8b](https://github.com/user-attachments/assets/a69795b9-a97c-4022-be8b-83ab3a8e28a7)


#### 9. Count of Facebook Followers

```python
# Distribution of customers following the facebook page (Numerical Breakdown)
facebook_page = df["Does he/she Followed Our Page?"].value_counts()
print(facebook_page)
```

![github image 9a](https://github.com/user-attachments/assets/e2fefd05-47ec-48ff-b1ef-e4bc8f71b67e)

```python
# Count of Facebook followers
fb_follow_counts = df["Does he/she Followed Our Page?"].value_counts()

# Plot
plt.figure(figsize=(4, 4))
sns.barplot(x=fb_follow_counts.index, y=fb_follow_counts.values, palette=['#3B9C9C', '#F08080'])
plt.title("Distribution of Customers Following the Facebook Page")
plt.xlabel("Follows Facebook Page")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()
```

![github image 9b](https://github.com/user-attachments/assets/259116fe-f48d-4fad-b12c-b2a410ac73f2)




