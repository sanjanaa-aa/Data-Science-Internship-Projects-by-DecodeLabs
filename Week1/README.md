# Week 1: Advanced EDA and Feature Engineering

## Description

This project focuses on data cleaning, preprocessing, and feature engineering to prepare raw data for machine learning applications. The dataset was cleaned by handling missing values, removing outliers, and creating new useful features.

## Dataset

- Dataset: Retail Sales Dataset
- Original Shape: (1200, 14)
- Final Shape: (1192, 17)

## Tasks Performed

### 1. Missing Value Handling
- Identified missing values in the CouponCode column.
- Filled missing values using the most frequent coupon code.

### 2. Outlier Detection and Removal
- Applied the IQR (Interquartile Range) method.
- Removed outliers from the TotalPrice column.

### 3. Feature Engineering
Created the following new features:
- AverageItemPrice
- OrderMonth
- OrderYear

### 4. Data Export
- Saved the cleaned dataset as `cleaned_dataset.xlsx`.

## Technologies Used

- Python
- Pandas
- NumPy
- OpenPyXL
- VS Code

## Project Files

- `project1.py`
- `cleaned_dataset.xlsx`
- `README.md`

## How to Run

```bash
python project1.py
```

## Output

- Cleaned dataset generated successfully.
- Missing values handled.
- Outliers removed.
- New features created.
- Dataset exported to Excel format.

## Conclusion

The raw dataset was successfully transformed into a clean and machine-learning-ready dataset through preprocessing and feature engineering techniques.