# Week 3: Customer Segmentation using K-Means Clustering

## Description

This project performs customer segmentation using the Online Shoppers Purchasing Intention Dataset. The goal is to identify different groups of customers based on their browsing behavior and shopping activity.

The project includes data preprocessing, feature encoding, data scaling, dimensionality reduction using PCA, cluster analysis using K-Means Clustering, and customer persona generation.

## Dataset

- Dataset: Online Shoppers Purchasing Intention Dataset
- Source: UCI Machine Learning Repository
- Records: 12,330
- Features Used: 17

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- VS Code

## Project Workflow

### 1. Data Loading
- Loaded the Online Shoppers dataset.
- Explored dataset structure and dimensions.

### 2. Data Preprocessing
- Checked missing values.
- Removed the Revenue column.
- Encoded categorical features:
  - Month
  - VisitorType
  - Weekend

### 3. Feature Scaling
- Applied StandardScaler to normalize all features.

### 4. PCA (Principal Component Analysis)
- Reduced 17 features into 2 principal components.
- Retained approximately 30.67% of total variance.

### 5. Elbow Method
- Calculated WCSS values for K = 1 to 10.
- Identified the optimal number of clusters.

### 6. Silhouette Score
- Evaluated clustering quality.
- Best score achieved at K = 3.

### 7. K-Means Clustering
- Applied K-Means with 3 clusters.
- Segmented customers into distinct groups.

### 8. Customer Personas
Generated three customer segments:

#### Cluster 0 – Regular Visitors
- Medium browsing activity
- Average engagement
- Largest customer group

#### Cluster 1 – Highly Engaged Customers
- High administrative and product-related activity
- Spend more time on the website
- Most valuable customer segment

#### Cluster 2 – Low Engagement Customers
- Very low activity
- Visit fewer pages
- Least engaged segment

## Results

### PCA Results
- Components: 2
- Total Variance Retained: 30.67%

### Best Number of Clusters
- K = 3

### Silhouette Score
- 0.5907

### Cluster Distribution
- Cluster 0: 9,626 customers
- Cluster 1: 1,550 customers
- Cluster 2: 1,154 customers

## Project Files

- customer_segmentation.py
- online_shoppers_intention.csv
- results.txt
- README.md

## How to Run

```bash
python customer_segmentation.py
```

## Output

- PCA dimensionality reduction
- Elbow Method graph
- Customer segmentation graph
- Cluster distribution analysis
- Customer personas
- Results summary

## Conclusion

The project successfully segmented customers into three meaningful groups using PCA and K-Means Clustering. The Elbow Method and Silhouette Score were used to identify the optimal number of clusters, helping businesses better understand customer behavior and engagement patterns.