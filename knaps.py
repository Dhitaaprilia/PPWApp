import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from numpy import array
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from collections import OrderedDict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import altair as alt
from sklearn.utils.validation import joblib



st.title("DATA MINING")
st.write("-------------------------------------------------------------------------------------------------------------------------")
st.write("**Nama  : Dhita Aprilia Dhamayanti**")
st.write("**NIM   : 200411100102**")
st.write("-------------------------------------------------------------------------------------------------------------------------")
upload_data, desc, preporcessing, modeling, implementation = st.tabs(["Upload Data","Dataset Description", "Preprocessing", "Modeling", "Implementation"])


with upload_data:
    st.write("""# Upload File""")
    st.write("Dataset yang digunakan adalah data Klasifikasi Kardiovaskuler dari https://www.kaggle.com/code/ekramasif/cardiovasculardiseasepredictionusingml")
    uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file, sep=';')
        st.write("Nama File Anda = ", uploaded_file.name)
        st.dataframe(df)

with desc:
    st.write("""# Deskripsi""")
    st.write("Dataset ini adalah dataset menengenai penyakit cardiovascular. Kardiovaskular atau Cardiovascular Disease (CVD) merupakan penyakit yang berkaitan dengan jantung dan pembuluh darah. Pada dataset ini terdapat 70.000 data.")
    st.write("Dataset ini diambil dari https://www.kaggle.com/code/ekramasif/cardiovasculardiseasepredictionusingml")
    st.subheader("""Fitur""")
    st.write("[1].  Age")
    st.write("[2].  Height")
    st.write("[3].  Weight")
    st.write("[4].  Gender")
    st.write("[5].  Ap_hi (Systolic Blood Pressure)")
    st.write("[6].  Ap_lo (Diastolic Blood Pressure)")
    st.write("[7].  Cholesterol")
    st.write("[8].  Gluc (Glucose)")
    st.write("[9].  Smoke (Smoking)")
    st.write("[10]. Alco (Alcohol)")
    st.write("[11]. Active (Physical Activity)")
    st.subheader("""Fungsi""")
    st.write("Aplikasi ini berfungsi untuk menentukan penyakit Cardiovascular.")
    st.write("Source Code dapat diakses melalui https://github.com/Dhitaaprilia/DataminingApp.git")

with preporcessing :
    st.write("""# Preprocessing""")
    st.subheader("""Normalisasi Data""")
    st.write("""Rumus Normalisasi Data :""")
    from PIL import Image
    image = Image.open('rumus.jpeg')
    st.image(image, use_column_width=False, width=250)
    st.markdown("""
    Dimana :
    - X = data yang akan dinormalisasi atau data asli
    - min = nilai minimum semua data asli
    - max = nilai maksimum semua data asli
    """)
    df = df.drop(columns=["id"])
    #Mendefinisikan Varible X dan Y
    X = df.drop(columns=['cardio'])
    y = df['cardio'].values
    df
    X
    df_min = X.min()
    df_max = X.max()
    
    #NORMALISASI NILAI X
    scaler = MinMaxScaler()
    #scaler.fit(features)
    #scaler.transform(features)
    scaled = scaler.fit_transform(X)
    features_names = X.columns.copy()
    #features_names.remove('label')
    scaled_features = pd.DataFrame(scaled, columns=features_names)

    st.subheader('Hasil Normalisasi Data')
    st.write(scaled_features)

    st.subheader('Target Label')
    dumies = pd.get_dummies(df.cardio).columns.values.tolist()
    dumies = np.array(dumies)

    labels = pd.get_dummies(df.cardio).columns.values.tolist()

    st.write(labels)

    # st.subheader("""Normalisasi Data""")
    # st.write("""Rumus Normalisasi Data :""")
    # st.image('https://i.stack.imgur.com/EuitP.png', use_column_width=False, width=250)
    # st.markdown("""
    # Dimana :
    # - X = data yang akan dinormalisasi atau data asli
    # - min = nilai minimum semua data asli
    # - max = nilai maksimum semua data asli
    # """)
    # df.weather.value_counts()
    # df = df.drop(columns=["date"])
    # #Mendefinisikan Varible X dan Y
    # X = df.drop(columns=['weather'])
    # y = df['weather'].values
    # df_min = X.min()
    # df_max = X.max()

    # #NORMALISASI NILAI X
    # scaler = MinMaxScaler()
    # #scaler.fit(features)
    # #scaler.transform(features)
    # scaled = scaler.fit_transform(X)
    # features_names = X.columns.copy()
    # #features_names.remove('label')
    # scaled_features = pd.DataFrame(scaled, columns=features_names)

    # #Save model normalisasi
    # from sklearn.utils.validation import joblib
    # norm = "normalisasi.save"
    # joblib.dump(scaled_features, norm) 


    # st.subheader('Hasil Normalisasi Data')
    # st.write(scaled_features)

with modeling:
    st.write("""# Modeling""")
    training, test = train_test_split(scaled_features,test_size=0.2, random_state=1)#Nilai X training dan Nilai X testing
    training_label, test_label = train_test_split(y, test_size=0.2, random_state=1)#Nilai Y training dan Nilai Y testing
    with st.form("modeling"):
        st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
        naive = st.checkbox('Gaussian Naive Bayes')
        k_nn = st.checkbox('K-Nearest Neighboor')
        destree = st.checkbox('Decission Tree')
        submitted = st.form_submit_button("Submit")

        # NB
        GaussianNB(priors=None)

        # Fitting Naive Bayes Classification to the Training set with linear kernel
        gaussian = GaussianNB()
        gaussian = gaussian.fit(training, training_label)

        # Predicting the Test set results
        y_pred = gaussian.predict(test)
    
        y_compare = np.vstack((test_label,y_pred)).T
        gaussian.predict_proba(test)
        gaussian_akurasi = round(100 * accuracy_score(test_label, y_pred))
        # akurasi = 10

        #Gaussian Naive Bayes
        # gaussian = GaussianNB()
        # gaussian = gaussian.fit(training, training_label)

        # probas = gaussian.predict_proba(test)
        # probas = probas[:,1]
        # probas = probas.round()

        # gaussian_akurasi = round(100 * accuracy_score(test_label,probas))

        #KNN
        K=10
        knn=KNeighborsClassifier(n_neighbors=K)
        knn.fit(training,training_label)
        knn_predict=knn.predict(test)

        knn_akurasi = round(100 * accuracy_score(test_label,knn_predict))

        #Decission Tree
        dt = DecisionTreeClassifier()
        dt.fit(training, training_label)
        # prediction
        dt_pred = dt.predict(test)
        #Accuracy
        dt_akurasi = round(100 * accuracy_score(test_label,dt_pred))

        if submitted :
            if naive :
                st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(gaussian_akurasi))
            if k_nn :
                st.write("Model KNN accuracy score : {0:0.2f}" . format(knn_akurasi))
            if destree :
                st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(dt_akurasi))
        
        grafik = st.form_submit_button("Grafik akurasi semua model")
        if grafik:
            data = pd.DataFrame({
                'Akurasi' : [gaussian_akurasi, knn_akurasi, dt_akurasi],
                'Model' : ['Gaussian Naive Bayes', 'K-NN', 'Decission Tree'],
            })

            chart = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    alt.X("Akurasi"),
                    alt.Y("Model"),
                    alt.Color("Akurasi"),
                    alt.Tooltip(["Akurasi", "Model"]),
                )
                .interactive()
            )
            st.altair_chart(chart,use_container_width=True)
  
with implementation:
    st.write("""# Implementasi""")
    with st.form("my_form"):
        Age = st.number_input('Usia')
        Gender = st.number_input('Jenis Kelamin')
        Height = st.number_input('Tinggi Badan')
        Weight = st.number_input('Berat Badan')
        Ap_hi = st.number_input('Systolic Blood Pressure')
        Ap_lo = st.number_input('Diastolic Blood Pressure')
        Cholesterol = st.number_input('Cholesterol')
        Glucose = st.number_input('Glukosa')
        Smoke = st.number_input('Smoking')
        Alco = st.number_input('Alcohol')
        Active = st.number_input('Physical Activity')
        model = st.selectbox('Pilihlah model yang akan anda gunakan untuk melakukan prediksi?',
                ('Gaussian Naive Bayes', 'K-NN', 'Decision Tree'))

        prediksi = st.form_submit_button("Submit")
        if prediksi:
            inputs = np.array([
            Age,Gender,Height,Weight,Ap_hi,Ap_lo,Cholesterol,Glucose,Smoke,Alco,Active
            ])

            df_min = X.min()
            df_max = X.max()
            input_norm = ((inputs - df_min) / (df_max - df_min))
            input_norm = np.array(input_norm).reshape(1, -1)

            if model == 'Gaussian Naive Bayes':
                mod = gaussian
            if model == 'K-NN':
                mod = knn 
            if model == 'Decision Tree':
                mod = dt

            input_pred = mod.predict(input_norm)


            st.subheader('Hasil Prediksi')
            st.write('Menggunakan Pemodelan :', model)

            st.write(input_pred)
