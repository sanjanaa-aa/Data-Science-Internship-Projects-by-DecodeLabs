import pandas as pd

df = pd.read_excel("Dataset for Data Analytics (2).xlsx")

# Handle missing values
mode_coupon = df["CouponCode"].mode()[0]
df["CouponCode"] = df["CouponCode"].fillna(mode_coupon)

# Find IQR
Q1 = df["TotalPrice"].quantile(0.25)
Q3 = df["TotalPrice"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

# Remove outliers
df = df[
    (df["TotalPrice"] >= lower_limit) &
    (df["TotalPrice"] <= upper_limit)
]
df["AverageItemPrice"] = df["TotalPrice"] / df["Quantity"]

df["Date"] = pd.to_datetime(df["Date"])

df["OrderMonth"] = df["Date"].dt.month
  
df["OrderYear"] = df["Date"].dt.year

print(df[["Date", "OrderMonth", "OrderYear"]].head())

print(df.shape)
 
df.to_excel("cleaned_dataset.xlsx", index=False)

print("Dataset saved successfully!")