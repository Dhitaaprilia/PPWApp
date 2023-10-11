import streamlit as st
import pandas as pd
import numpy as np
from numpy import array
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from datetime import datetime

st.title('Prediksi Harga Emas Menggunakan Metode Linear Regression')

# Mendapatkan data harga emas dari URL
df = pd.read_csv('https://raw.githubusercontent.com/DwiAqilahP/kolaborasipro/main/PTRO.JK.csv')
df.isnull().sum()
df['Open'] = df['Open'].fillna(value=df['Open'].median())
df['High'] = df['High'].fillna(value=df['High'].median())
df['Low'] = df['Low'].fillna(value=df['Low'].median())
df['Close'] = df['Close'].fillna(value=df['Close'].median())
df['Adj Close'] = df['Adj Close'].fillna(value=df['Adj Close'].median())
df['Volume'] = df['Volume'].fillna(value=df['Volume'].median())


# Konversi kolom 'Date' menjadi tipe datetime
df['Date'] = pd.to_datetime(df['Open'])

# Menampilkan data harga emas
st.write('Data:', df)

# Memisahkan data menjadi data train dan data test
data = df['Open']
n = len(data)
sizeTrain = (round(n * 0.8))
data_train = pd.DataFrame(data[:sizeTrain])
data_test = pd.DataFrame(data[sizeTrain:])

# Praproses data dengan MinMaxScaler
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(data_train)
test_scaled = scaler.transform(data_test)

# reshaped_data = data.reshape(-1, 1)
train = pd.DataFrame(train_scaled, columns = ['data'])
train = train['data']
# st.write('Data Train',train)

test = pd.DataFrame(test_scaled, columns = ['data'])
test = test['data']

# Menggabungkan train dan test menjadi satu tabel
merged_data = pd.concat([train, test], axis=1)
merged_data.columns = ['Train', 'Test']
# Menampilkan tabel hasil penggabungan
st.write('Hasil Normalisasi:')
st.write(merged_data)

# Membentuk sequence untuk training dan testing
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    X = array(X)
    y = array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Perubahan bentuk data
    return X, y

# Menyiapkan data train
X_train, Y_train = split_sequence(train_scaled, 2)
x_train = pd.DataFrame(X_train.reshape((X_train.shape[0], X_train.shape[1])), columns=['xt-2', 'xt-1'])  # Perubahan bentuk data
y_train = pd.DataFrame(Y_train, columns=['xt'])
dataset_train = pd.concat([x_train, y_train], axis=1)
X_train = dataset_train.iloc[:, :2].values
Y_train = dataset_train.iloc[:, -1].values

# Menyiapkan data test
test_x, test_y = split_sequence(test_scaled, 2)
x_test = pd.DataFrame(test_x.reshape((test_x.shape[0], test_x.shape[1])), columns=['xt-2', 'xt-1'])  # Perubahan bentuk data
y_test = pd.DataFrame(test_y, columns=['xt'])
dataset_test = pd.concat([x_test, y_test], axis=1)
# st.write('Dataset Test:', dataset_test)
X_test = dataset_test.iloc[:, :2].values
Y_test = dataset_test.iloc[:, -1].values

# Melakukan prediksi dengan Linear Regression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
knn_preds = regressor.predict(X_test)

from sklearn.tree import DecisionTreeRegressor
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, Y_train)
dt_preds = dt_model.predict(X_test)

from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor()
rf_model.fit(X_train, Y_train)
rf_preds = rf_model.predict(X_test)

from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(X_train, Y_train)
lr_preds = lr_model.predict(X_test)

from sklearn.svm import SVR
svr_model = SVR()
svr_model.fit(X_train, Y_train)
svr_preds = svr_model.predict(X_test)

# Mengembalikan skala data ke aslinya
knn_preds = scaler.inverse_transform(knn_preds.reshape(-1, 1))
dt_preds = scaler.inverse_transform(dt_preds.reshape(-1, 1))
rf_preds = scaler.inverse_transform(rf_preds.reshape(-1, 1))
lr_preds = scaler.inverse_transform(lr_preds.reshape(-1, 1))
svr_preds = scaler.inverse_transform(svr_preds.reshape(-1, 1))

reshaped_datates = Y_test.reshape(-1, 1)
actual_test = scaler.inverse_transform(reshaped_datates)

# Menyimpan hasil prediksi dan data aktual dalam file Excel
prediksi_knn = pd.DataFrame(knn_preds)
prediksi_dt = pd.DataFrame(dt_preds)
prediksi_rf = pd.DataFrame(rf_preds)
prediksi_lr = pd.DataFrame(lr_preds)
prediksi_svr = pd.DataFrame(svr_preds)

actual = pd.DataFrame(actual_test)

# Menghitung mean absolute percentage error (MAPE)
knn_mape = mean_absolute_percentage_error(knn_preds, actual_test) * 100
dt_mape = mean_absolute_percentage_error(dt_preds, actual_test) * 100
rf_mape = mean_absolute_percentage_error(rf_preds, actual_test)* 100
lr_mape = mean_absolute_percentage_error(lr_preds, actual_test) * 100
svr_mape = mean_absolute_percentage_error(svr_preds, actual_test) * 100

# Menampilkan hasil prediksi dan MAPE
# st.write("Hasil Prediksi:")
# st.write(prediksi_knn)
# st.write("Data Aktual:")
# st.write(aktual)
st.write("MAPE KNeighborsRegressor:", knn_mape)
st.write("MAPE DecisionTreeRegressor:", dt_mape)
st.write("MAPE RandomForestRegressor:", rf_mape)
st.write("MAPE LinearRegression:", lr_mape)
st.write("MAPE SVR:", svr_mape)

# Input tanggal untuk memprediksi harga emas
st.sidebar.title("Prediksi Harga Saham")
selected_date = st.sidebar.date_input("Pilih Tanggal")

if selected_date is not None:
    # Ubah tanggal menjadi format yang sesuai dengan data
    selected_date_str = selected_date.strftime("%Y-%m-%d")

    # Cari indeks tanggal terdekat dalam data
    closest_date = pd.Timestamp(selected_date)  # Convert to pd.Timestamp
    closest_date_idx = df['Date'].sub(closest_date).abs().idxmin()

    # Ambil data sebelum dan pada tanggal yang dipilih
    selected_data = df.loc[closest_date_idx-2:closest_date_idx, 'Open']

    # Praproses data dengan MinMaxScaler
    selected_data_scaled = scaler.transform(selected_data.values.reshape(-1, 1))

    # Bentuk input sequence untuk prediksi
    X_selected, _ = split_sequence(selected_data_scaled, 2)
    x_selected = pd.DataFrame(X_selected.reshape((X_selected.shape[0], X_selected.shape[1])), columns=['xt-2', 'xt-1'])
    X_selected = x_selected.values

    # Prediksi harga emas pada tanggal yang dipilih
    predicted_price_scaled = regressor.predict(X_selected)
    predicted_price = scaler.inverse_transform(predicted_price_scaled.reshape(-1, 1))

    st.sidebar.write("Prediksi Harga Saham pada Tanggal", selected_date_str)
    st.sidebar.write(predicted_price)