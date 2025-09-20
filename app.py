import streamlit as st
import pandas as pd
from surprise import SVD, Dataset, Reader

# --- Load dataset ---
places = pd.read_csv("tourism_with_id.csv")
ratings = pd.read_csv("tourism_rating.csv")
users = pd.read_csv("user.csv")

# Filter Bandung only
places_bdg = places[places['City'].str.lower() == "bandung"]
ratings_bdg = ratings.merge(
    places_bdg[['Place_Id','Place_Name','Category','Price','Rating','City']],
    on='Place_Id',
    how='inner'
)

# --- Siapkan model SVD ---
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_bdg[['User_Id','Place_Id','Place_Ratings']], reader)

trainset = data.build_full_trainset()
model = SVD()
model.fit(trainset)

# --- Streamlit UI ---
st.title("🎯 Sistem Rekomendasi Wisata Bandung")

# Input dari user
kategori_input = st.selectbox(
    "Pilih kategori wisata yang kamu sukai:",
    places_bdg['Category'].unique()
)

usia_input = st.number_input("Masukkan usia kamu:", min_value=10, max_value=80, value=25)

if st.button("🔍 Prediksi Rekomendasi"):
    # Ambil user contoh dari kelompok umur mirip (misalnya user terdekat)
    if usia_input <= 25:
        sample_user = users[users['Age'] <= 25]['User_Id'].iloc[0]
    elif usia_input <= 35:
        sample_user = users[(users['Age'] > 25) & (users['Age'] <= 35)]['User_Id'].iloc[0]
    else:
        sample_user = users[(users['Age'] > 35)]['User_Id'].iloc[0]

    # Cari tempat yang belum dikunjungi user
    visited = ratings_bdg[ratings_bdg['User_Id'] == sample_user]['Place_Id'].tolist()
    not_visited = [p for p in places_bdg['Place_Id'].unique() if p not in visited]

    # Filter hanya kategori yang dipilih
    not_visited_cat = places_bdg[places_bdg['Place_Id'].isin(not_visited)]
    not_visited_cat = not_visited_cat[not_visited_cat['Category'] == kategori_input]

    # Prediksi rating
    preds = [(p, model.predict(sample_user, p).est) for p in not_visited_cat['Place_Id']]
    pred_df = pd.DataFrame(preds, columns=['Place_Id','predicted_rating'])

    # Gabungkan dengan info tempat
    top7 = (
        pred_df
        .merge(places_bdg[['Place_Id','Place_Name','Category','Price','Rating']], on='Place_Id')
        .sort_values('predicted_rating', ascending=False)
        .head(7)
        .reset_index(drop=True)
    )

    st.subheader("✨ Top 7 Rekomendasi Wisata untuk Kamu:")
    st.dataframe(top7)

    # Visualisasi bar chart
    st.bar_chart(top7.set_index('Place_Name')['predicted_rating'])