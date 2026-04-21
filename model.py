import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
df = pd.read_csv("house_data.csv")

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# City mapping
city_map = {
    'chandigarh': 0,
    'mohali': 1,
    'delhi': 2
}

df['city'] = df['city'].str.lower().map(city_map)
df['city'] = df['city'].fillna(0)

# Features (NOW 5 FEATURES)
X = df[['sqft_living', 'bedrooms', 'bathrooms', 'floors', 'city']]
y = df['price']

# Train
model = LinearRegression()
model.fit(X, y)

# Save
pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model retrained with 5 features!")