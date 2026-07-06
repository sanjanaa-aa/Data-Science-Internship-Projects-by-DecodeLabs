# DecodeLabs Week 1 Project

## Project Title
Advanced EDA and Feature Engineering

## Objective
Transform raw data into a clean dataset ready for machine learning.

## Tasks Performed

### 1. Missing Value Handling
- Identified missing values in CouponCode column.
- Filled missing values using the most frequent coupon code (FREESHIP).

### 2. Outlier Detection and Removal
- Used the IQR (Interquartile Range) method.
- Removed outliers from TotalPrice column.

### 3. Feature Engineering
Created the following new features:
- AverageItemPrice
- OrderMonth
- OrderYear

### 4. Final Dataset
- Original Shape: (1200, 14)
- Final Shape: (1192, 17)

## Tools Used
- Python
- Pandas
- NumPy
- VS Code

## Files
- project1.py
- cleaned_dataset.xlsx
- README.md