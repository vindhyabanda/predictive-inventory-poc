import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Uses Random Forest Model

# Loading the data
file_path = "sample_inventory_data.xlsx"
df = pd.read_excel(file_path)

# Data Preprocessing
df['Date'] = pd.to_datetime(df['Date'])
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Month'] = df['Date'].dt.month

# Convert categorical data to numerical (this is One-Hot Encoding)
df = pd.get_dummies(df, columns=['Category', 'Promotion', 'Supplier'])

# Feature we're considering, target to be populated: Quantity Sold
features = df.drop(columns=['Date', 'Product ID', 'Product Name', 'Quantity Sold'])
target = df['Quantity Sold']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')

# Create a sample row to predict future quantity sold
future_df = pd.DataFrame({
    'Stock Level': [150],
    'Reorder Point': [50],
    'Lead Time (Days)': [3],
    'Sales Price': [899.99],
    'DayOfWeek': [2],  # Example: Wednesday
    'Month': [8],  # Example: August
    'Category_Electronics': [1],
    'Category_Clothing': [0],
    'Category_Groceries': [0],
    'Promotion_No': [1],
    'Promotion_Yes': [0],
    'Supplier_Supplier 1': [1],
    'Supplier_Supplier 2': [0],
    'Supplier_Supplier 3': [0],
    'Supplier_Supplier 4': [0],
    'Supplier_Supplier 5': [0],
})

# Ensures that future_df has the same columns as the training features
future_df = future_df.reindex(columns=X_train.columns, fill_value=0)

# Predicts future quantity sold
prediction = model.predict(future_df)
print(f'Predicted Quantity Sold: {prediction[0]}')