import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Prédiction de Population")

@st.cache_data
def charger_donnees():

    try:
        population_cleaned = pd.read_csv("population_clean.csv")
        
        id_vars = ['Country Name', 'Continent']
        year_columns = [col for col in population_cleaned.columns if col not in id_vars]

        population_verticale = pd.melt(
            population_cleaned,
            id_vars=id_vars,
            value_vars=year_columns,
            var_name='Annee',
            value_name='Population'
        )
        population_verticale['Annee'] = pd.to_numeric(population_verticale['Annee'])
        population_verticale.dropna(subset=['Population'], inplace=True) # Retirer les lignes sans données de population

        return population_verticale

    except FileNotFoundError:
        st.error("Erreur : arrrrrggggg Le fichier 'population_clean.csv' est introuvable. Veuillez vérifier le chemin d'accès.")
        return None

def creer_graphique_prediction(nom_pays, annee_coupure, df_complet):

    df_pays = df_complet[df_complet['Country Name'] == nom_pays].copy()
    if df_pays.empty: return None
    df_prophet = df_pays.rename(columns={'Annee': 'ds', 'Population': 'y'})
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], format='%Y')
    train_data = df_prophet[df_prophet['ds'].dt.year <= annee_coupure]
    test_data = df_prophet[df_prophet['ds'].dt.year > annee_coupure]
    if test_data.empty: return None
    model = Prophet()
    model.fit(train_data)
    future_dates = model.make_future_dataframe(periods=len(test_data), freq='AS')
    forecast = model.predict(future_dates)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(test_data['ds'], test_data['y'], 'r', linewidth=2, label='Données Réelles (Test)')
    model.plot(forecast, ax=ax)
    ax.plot(train_data['ds'], train_data['y'], 'k.', label='Données d\'Entraînement')
    ax.set_title(f'Prédiction vs Réalité pour {nom_pays} (après {annee_coupure})', fontsize=16)
    ax.set_xlabel('Année', fontsize=12)
    ax.set_ylabel('Population', fontsize=12)
    ax.legend()
    ax.grid(True)
    return fig

st.title("📈 Outil de Prédiction de Population")
st.markdown("Utilisez les options dans la barre latérale pour choisir un pays et une date de fin d'entraînement.")

population_verticale = charger_donnees()

if population_verticale is not None:
    st.sidebar.header("⚙️ Paramètres")
    liste_pays = sorted(population_verticale['Country Name'].unique())
    pays_choisi = st.sidebar.selectbox("Choisissez un pays :", liste_pays)
    min_annee = int(population_verticale['Annee'].min())
    max_annee = int(population_verticale['Annee'].max() - 1)
    annee_choisie = st.sidebar.slider(
        "Choisissez l'année de fin d'entraînement :",
        min_value=min_annee,
        max_value=max_annee,
        value=2005
    )
    st.subheader(f"Analyse pour : {pays_choisi}")
    figure_prediction = creer_graphique_prediction(pays_choisi, annee_choisie, population_verticale)
    if figure_prediction:
        st.pyplot(figure_prediction)
    else:
        st.warning(f"Pas de données de test disponibles après {annee_choisie} pour créer une comparaison.")
