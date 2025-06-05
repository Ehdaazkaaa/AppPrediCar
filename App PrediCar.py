import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

# ---------------------- CSS UI Styling ----------------------
st.markdown("""
    <style>
    body {
        background-color: #f5f7fa;
    }
    .main {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    h1, h2, h3, .st-bc {
        color: #2e344d;
    }
    .stButton > button {
        background-color: #ecc159;
        color: #2e344d;
        font-weight: bold;
        border: none;
        border-radius: 12px;
        padding: 12px 28px;
        font-size: 16px;
        margin-top: 15px;
        transition: 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #df5849;
        color: white;
        transform: scale(1.03);
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------- Header ----------------------
st.image(
    "https://drive.google.com/file/d/1L8df0VxJWKNK3GfxaeDPbNRnl-70I3Ge/view?usp=sharing",
    use_container_width=True
)
st.markdown('<div class="main">', unsafe_allow_html=True)
st.title("🚗 Prediksi Harga Mobil Bekas Toyota 🚗")
st.markdown("**🎯 Prediksi Harga Mobil Toyota Bekas Anda Menggunakan Deep Learning (ANN)**")

# ---------------------- Load dan Proses Data ----------------------
df = pd.read_csv("Toyota (1).csv")
df.dropna(inplace=True)

features = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
X = df[features]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'scaler.pkl')

# ---------------------- Build & Train Model ANN ----------------------
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train_scaled, y_train, epochs=100, batch_size=16, verbose=0)
model.save("model_toyota_dl.h5")

# ---------------------- Currency Conversion Rate ----------------------
EUR_TO_IDR = 18617.72

# ---------------------- Form Input User ----------------------
st.subheader("📝 Masukkan Spesifikasi Mobil")
year = st.number_input("Tahun Mobil", min_value=1990, max_value=2025, value=2018)
mileage = st.number_input("Jarak Tempuh (km)", min_value=0, max_value=300000, value=50000)
tax = st.number_input("Pajak (Euro)", min_value=0, max_value=600, value=150)
mpg = st.number_input("Konsumsi BBM (mpg)", min_value=10.0, max_value=100.0, value=40.0)
engineSize = st.number_input("Ukuran Mesin (liter)", min_value=0.5, max_value=6.0, value=1.6, step=0.1)

# ---------------------- Prediksi Harga Mobil ----------------------
if st.button("🔍 Prediksi Harga"):
    input_df = pd.DataFrame([[year, mileage, tax, mpg, engineSize]], columns=features)
    model = load_model("model_toyota_dl.h5")
    scaler = joblib.load("scaler.pkl")
    input_scaled = scaler.transform(input_df)

    prediction_euro = model.predict(input_scaled)[0][0]
    prediction_rupiah = prediction_euro * EUR_TO_IDR
    tax_rupiah = tax * EUR_TO_IDR

    st.markdown(f"💶 **Pajak Mobil:** Rp {tax_rupiah:,.2f}")
    st.success(f"💰 **Estimasi Harga Mobil Bekas:** Rp {prediction_rupiah:,.2f}")

st.markdown('</div>', unsafe_allow_html=True)