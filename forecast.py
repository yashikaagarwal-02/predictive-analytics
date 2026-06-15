import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("sales_data.csv")

# Features and target
X = df[['Month']]
y = df['Sales']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict future sales
future_months = pd.DataFrame({'Month': [11, 12, 13]})
predictions = model.predict(future_months)

print("Predicted Sales:")
for month, sale in zip([11, 12, 13], predictions):
    print(f"Month {month}: {sale:.2f}")

# Plot graph
plt.scatter(df['Month'], df['Sales'], label="Actual Sales")
plt.plot(df['Month'], model.predict(X), label="Regression Line")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Sales Forecasting")
plt.legend()
plt.show()