import streamlit as st
import pickle
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Free geocoder
geolocator = Nominatim(user_agent="house_price_app")

st.set_page_config(page_title="House Price Predictor", layout="wide")

# ---------- CITY MAP ----------
city_map = {
    'delhi': 0,
    'new delhi': 0,
    'chandigarh': 1,
    'mohali': 2,
    'sahibzada ajit singh nagar': 2
}

# ---------- FUNCTION (FIXED) ----------
def get_city(address):
    try:
        location = geolocator.geocode(address)
        if location:
            data = geolocator.reverse((location.latitude, location.longitude))
            address_data = data.raw['address']

            # 🔥 handle all cases
            city = (
                address_data.get('city') or
                address_data.get('town') or
                address_data.get('village') or
                address_data.get('county') or
                address_data.get('state_district') or
                address_data.get('state')
            )

            if city:
                return city.lower()

    except:
        return None

    return None

# ---------- UI ----------
st.title("🏠 House Price Prediction (FREE VERSION)")
st.write("### Enter house details 👇")

# TEXT INPUT
address = st.text_input("📍 Enter Location (Delhi, Pune, Mohali, Village, etc.)")

city = None

if address:
    city = get_city(address)

    if city:
        st.success(f"📍 Detected Location: {city.title()}")
    else:
        st.warning("⚠️ Could not detect location")

# ---------- INPUT ----------
col1, col2 = st.columns(2)

with col1:
    sqft_living = st.number_input("📏 Living Area (sq ft)", min_value=500)

with col2:
    bedrooms = st.number_input("🛏 Bedrooms", min_value=1)
    bathrooms = st.number_input("🛁 Bathrooms", min_value=1)
    floors = st.number_input("🏢 Floors", min_value=1)

# ---------- PREDICTION ----------
if st.button("🚀 Predict Price"):

    if not city:
        st.error("❌ Please enter valid location")
    else:
        city_value = city_map.get(city, 0)

        input_data = np.array([[sqft_living, bedrooms, bathrooms, floors, city_value]])

        prediction = model.predict(input_data)

        st.success(f"💰 Estimated Price: ₹ {prediction[0]:,.0f}")

        if city not in city_map:
            st.warning("⚠️ City not in training data → prediction may be less accurate")

# ---------- DATA ----------
df = pd.read_csv("house_data.csv")

st.subheader("📊 Data Insights")

col3, col4 = st.columns(2)

with col3:
    st.bar_chart(df['price'])

with col4:
    st.line_chart(df[['sqft_living', 'price']].set_index('sqft_living'))

# ---------- FOOTER ----------
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")