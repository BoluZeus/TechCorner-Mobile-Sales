# 📱 TechCorner Mobile Sales Analysis

> A comprehensive analysis of **10 months of mobile sales data** from TechCorner, a retail mobile phone shop in Bangladesh — uncovering customer behaviour, sales trends, revenue patterns, and marketing effectiveness.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Tools Used](#-tools-used)
- [Data Cleaning & Preparation](#-data-cleaning--preparation)
- [Analysis Sections](#-analysis-sections)
- [Key Findings](#-key-findings)
- [Recommendations](#-recommendations)
- [How to Run](#-how-to-run)

---

## 🔍 Project Overview

TechCorner is a mobile phone retail shop based in Rangamati, Bangladesh. This project analyses their sales records across **May 2024 – March 2025**, covering:

- Monthly sales and revenue trends
- Customer demographics (age, gender, location)
- Top-selling models and brands
- Facebook marketing effectiveness
- New vs returning customer behaviour
- Cross-variable insights (e.g. do Facebook customers spend more?)

The goal is to surface **actionable business insights** that can guide stock decisions, marketing strategy, and customer retention efforts.

---

## 📁 Dataset

| Property | Details |
|----------|---------|
| **File** | `TechCorner_Sales_update.csv` |
| **Rows** | 8,871 transactions |
| **Period** | May 2024 – March 2025 (10 months) |
| **Source** | TechCorner retail shop, Rangamati, Bangladesh |

**Columns:**

| Original Column | Cleaned Name | Description |
|----------------|-------------|-------------|
| `Cus.ID` | `customer_id` | Unique customer identifier |
| `Date` | `Date` | Date of sale |
| `Cus. Location` | `location` | Customer's area (Inside/Outside/Sadar) |
| `Age` | `Age` | Customer age |
| `Gender` | `Gender` | Customer gender |
| `Mobile Name` | `mobile_name` | Phone model purchased |
| `Sell Price` | `sell_price` | Sale price in BDT |
| `Does he/she Come from Facebook Page?` | `facebook_referral` | Came via Facebook? |
| `Does he/she Followed Our Page?` | `follows_page` | Follows Facebook page? |
| `Did he/she buy any mobile before?` | `returning_customer` | Returning customer? |
| `Did he/she hear of our shop before?` | `heard_of_shop` | Awareness before visit? |

---

## 🛠 Tools Used

| Tool | Purpose |
|------|---------|
| **Python** | Core analysis language |
| **Pandas** | Data loading, cleaning, manipulation |
| **NumPy** | Numerical operations |
| **Matplotlib** | Visualisations |
| **Seaborn** | Statistical charts |
| **Jupyter Notebook** | Interactive analysis environment |

---

## 🧹 Data Cleaning & Preparation

Steps taken before analysis:

1. **Loaded and inspected** the dataset for shape, types, and nulls
2. **Converted data types** — `Sell Price` and `Age` to numeric, `Date` to datetime
3. **Dropped nulls** — removed rows with missing values
4. **Removed empty column** — dropped `Unnamed: 11` (trailing blank column)
5. **Renamed columns** — replaced long, messy column names with clean snake_case equivalents
6. **Feature engineering:**
   - Extracted `month` from `Date` for time-series grouping
   - Extracted `brand` from the first word of `mobile_name`
   - Created `age_group` bins: `18–25`, `26–35`, `36–45`, `46–50`

---

## 📊 Analysis Sections

### 1. Monthly Sales Trend
Sales grew steadily from **153 transactions in May 2024** to a peak of **943 in October 2024**, before a slight taper in early 2025 — a typical post-holiday pattern.

### 2. Sales by Customer Location
Demand is nearly equal across all three zones (Outside Rangamati: 3,000 | Sadar: 2,972 | Inside: 2,899), confirming strong broad geographic reach.

### 3. Age Distribution
The average buyer is **34 years old**. The core customer base falls between **26–42**. Age ranges from 18 to 50.

### 4. Gender Distribution
Sales are split almost perfectly: **50.2% Female / 49.8% Male** — marketing should target both genders equally.

### 5. Top 10 Best-Selling Models
Budget 5G models dominate volume:
- 🥇 Moto G85 5G 8/128 — 560 units
- 🥈 Galaxy S24 Ultra 12/256 — 541 units
- 🥉 Note 11S 6/128 — 538 units

Premium models (Pixel 8 Pro, iPhone 16 Pro) also feature in the top 10.

### 6. Price Distribution
- **Average sell price:** BDT 25,068
- **Middle 50%:** BDT 17,466 – 25,777
- **Range:** BDT 12,702 – 200,465

The long right tail confirms meaningful premium phone demand.

### 7. Revenue Analysis
| Metric | Value |
|--------|-------|
| **Total Revenue** | BDT 222,381,657 |
| **Total Transactions** | 8,871 |
| **Avg Revenue per Sale** | BDT 25,068 |
| **Estimated Profit (12% margin)** | BDT ~26,685,799 |

**Samsung leads all brands in total revenue** (BDT 42.2M), despite sharing the volume top-10 with budget brands. This confirms that premium Samsung stock drives disproportionate revenue.

> *Note: Profit is estimated using a 12% retail margin assumption, typical for mobile retail in Bangladesh. No cost-price column is present in the dataset.*

### 8. Facebook Marketing Effectiveness
- **34.8%** of customers came via the Facebook page
- **Facebook customers spend slightly more** on average (BDT 25,206 vs BDT 24,995)
- Facebook is a proven, quality acquisition channel — not just a volume driver

### 9. New vs Returning Customers
- **75.3%** first-time buyers → strong acquisition
- **24.7%** returning customers → retention opportunity

### 10. Cross-Analyses

**Do Facebook customers spend more?**
Yes — BDT 25,206 vs BDT 24,995 average. Facebook attracts quality buyers.

**Which age group spends the most?**
The **46–50** age group has the highest average spend, suggesting older customers prefer higher-end models.

**Do returning customers prefer different brands?**
Returning customer rates are broadly consistent across all major brands, suggesting loyalty is to the shop rather than a specific brand.

---

## 💡 Key Findings

| # | Finding | Implication |
|---|---------|-------------|
| 1 | Total revenue: **BDT 222M** across 8,871 transactions | Strong business performance over 10 months |
| 2 | Sales grew steadily from 153 → 943/month | Marketing and demand momentum is building |
| 3 | Samsung leads revenue despite sharing volume with budget brands | Premium Samsung stock drives outsized revenue |
| 4 | Budget 5G models dominate volume | Midrange 5G is the core customer demand |
| 5 | 34.8% of customers come from Facebook | Facebook is a proven acquisition channel |
| 6 | Facebook customers spend slightly more | FB marketing attracts quality buyers |
| 7 | 75.3% are first-time customers | Strong acquisition, but retention needs attention |
| 8 | Older customers (46–50) spend the most | Premium promotions should target older demographics |
| 9 | Demand is equal across all 3 zones | Delivery expansion could unlock untapped demand |

---

## ✅ Recommendations

1. **Stock strategy** — Maintain strong inventory of budget 5G models (BDT 17K–26K) as the core volume driver. Always have the top 10 models available.

2. **Premium push** — Introduce targeted promotions for high-end Samsung, iPhone, and Pixel models aimed at the 36–50 age group, who spend the most per transaction.

3. **Facebook marketing** — Continue and increase Facebook investment. The channel delivers 34.8% of customers who spend above average.

4. **Loyalty programme** — With 75.3% first-time buyers, a simple repeat-purchase incentive (e.g., discount on next phone) could significantly improve retention rates.

5. **Delivery expansion** — Outside Rangamati customers represent the largest single location group. A delivery or courier partnership would capture more of this demand without a physical presence.

6. **Data improvement** — Adding a purchase/cost price column to future sales records would enable true profit margin analysis rather than estimates.

---

## ▶️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/BoluZeus/TechCorner-Mobile-Sales.git
cd TechCorner-Mobile-Sales

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn jupyter

# 3. Launch the notebook
jupyter notebook TechCorner_Sales_Analysis.ipynb
```

Make sure `TechCorner_Sales_update.csv` is in the same directory as the notebook.

---

## 👤 Author

**BoluZeus**  
Data Analyst | [GitHub Profile](https://github.com/BoluZeus)

---

*This project was built as part of a data analytics portfolio to demonstrate end-to-end EDA skills using Python.*
