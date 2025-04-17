import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv("D:/Downloads/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69.csv")

#Remove duplicates
df_cleaned = df.drop_duplicates()

#Handling NA values in pollutant
pollutant_cols = ['pollutant_min', 'pollutant_max', 'pollutant_avg']
for col in pollutant_cols:
    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())

#Standardize city/state names
df_cleaned['city'] = df_cleaned['city'].str.title()
df_cleaned['state'] = df_cleaned['state'].str.title()

# Data Visualization

# 1. Bar plot – Top 10 cities with highest average pollutant
top_cities = df_cleaned.groupby('city')['pollutant_avg'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
top_cities.plot(kind='bar', color='salmon')
plt.title("Top 10 Cities by Avg Pollutant Level")
plt.xlabel("City")
plt.ylabel("Pollutant Avg")
plt.tight_layout()
plt.show()

# 2. Histogram – Distribution of average pollutants
plt.figure(figsize=(10, 6))
sns.histplot(df_cleaned['pollutant_avg'], kde=True, bins=30, color='skyblue')
plt.title("Distribution of Pollutant Average")
plt.tight_layout()
plt.show()

# 3. Scatter plot – Min vs Max pollutant
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_cleaned, x='pollutant_min', y='pollutant_max', hue='pollutant_id')
plt.title("Pollutant Min vs Max")
plt.tight_layout()
plt.show()

# 4. Heatmap – Correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(df_cleaned[pollutant_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 5. Box plot – Avg pollutant by pollutant_id
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_cleaned, x='pollutant_id', y='pollutant_avg')
plt.title("Boxplot of Avg Pollutant by Type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Linear Regression Model 
# Features & target
X = df_cleaned[['pollutant_min', 'pollutant_max']]
y = df_cleaned['pollutant_avg']

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split into train-test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Plot prediction
plt.figure(figsize=(8, 6))
sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
plt.xlabel("Actual Pollutant Avg")
plt.ylabel("Predicted Pollutant Avg")
plt.title(f"Linear Regression Prediction with Line\nMSE: {mse:.2f}, R²: {r2:.2f}")
plt.grid(True)
plt.tight_layout()
plt.show()
