import streamlit as st
from streamlit_lottie import st_lottie
import json  # <- Import manquant
import pandas as pd
from preprocessing import clean_text, translate_to_english
from similarity import compute_similarity
from summarizer import summarize_text
import time
import os
from groq import Groq
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="StaySmart & RealReview Assistant", layout="wide")
st.title("🏡 StaySmart & RealReview Assistant")

# Init Groq Client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Menu de navigation
menu = st.sidebar.selectbox("Navigation", ["🏠 Accueil", "🧪 Analyse Fichier", "🌐 Scraper URL VRBO"])

if menu == "🏠 Accueil":
    st.markdown("""
    ## Bienvenue 👋
    Grâce à notre application, vous avez **deux parcours possibles** pour enrichir les descriptions de logements :
    - 🏠 **Scraper une annonce précise depuis une page VRBO** : l’application extrait automatiquement la description et les avis clients liés à cette page, puis effectue tout le processus d’analyse (nettoyage, traduction, similarité, résumé, génération).
    - 📊 **Analyser une base de données complète** : vous pouvez importer un fichier Excel contenant plusieurs annonces. L’utilisateur choisit un **index** correspondant à un logement donné, et l’application applique le pipeline NLP à ce cas précis.
    ### Objectif 🎯
    - Nettoyage et traduction des données
    - Calcul de similarité sémantique
    - Résumé intelligent basé sur les avis clients
    - Génération via API Groq (LLM)
    """)
    # --- Fonction pour charger Lottie ---
    def load_lottie_file(filepath: str):
        with open(filepath, "r") as f:
            return json.load(f)

    try:
        lottie_animation = load_lottie_file("Animation.json")
        st_lottie(lottie_animation, height=300, key="lottie_intro")
    except Exception as e:
        st.warning("Le module streamlit-lottie n'est pas installé ou l'animation Lottie est introuvable.")


elif menu == "🧪 Analyse Fichier":
    st.header("Analyse à partir d'une base de données locale")
    uploaded_file = st.file_uploader("📥 Téléverser le fichier Excel contenant les descriptions et commentaires", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)

        st.subheader("1️⃣ Données brutes")
        st.dataframe(df)

        st.subheader("2️⃣ Données nettoyées")
        df["description_clean"] = df["description"].apply(clean_text)
        df["comment_clean"] = df["comment"].apply(clean_text)
        st.dataframe(df[["description_clean", "comment_clean"]])

        st.subheader("3️⃣ Indexation")
        df = df.dropna(subset=["description_clean"])
        df["index"] = df.groupby("description_clean", sort=False).ngroup()
        st.dataframe(df[["index", "description_clean", "comment_clean"]])

        st.subheader("4️⃣ Entrée d'un index")
        index_input = st.number_input("Saisir un index :", min_value=0, max_value=df["index"].max(), step=1)

        df_selected = df[df["index"] == index_input]
        if not df_selected.empty:
            desc = df_selected["description_clean"].iloc[0]
            comments = df_selected["comment_clean"].tolist()

            desc_en = translate_to_english(desc)
            comments_en = [translate_to_english(c) for c in comments]

            st.markdown("### 📄 Description traduite")
            st.write(desc_en)
            st.markdown("### 💬 Commentaires traduits")
            for c in comments_en:
                st.write("-", c)

            st.subheader("5️⃣ Similarité cosine")
            st.markdown("**DistilBERT-based similarity computation")
            similarities = [compute_similarity(desc_en, cmt) for cmt in comments_en]
            top5 = sorted(zip(comments_en, similarities), key=lambda x: x[1], reverse=True)[:5]

            st.markdown("### 🔝 Top 5 commentaires les plus similaires")
            for i, (cmt, sim) in enumerate(top5):
                st.markdown(f"**{i+1}.** Similarité: {sim:.3f}<br>{cmt}", unsafe_allow_html=True)

            st.subheader("6️⃣ Résumé avec T5")
            combined_top5 = " ".join([c[0] for c in top5])
            t5_summary = summarize_text(combined_top5)
            st.markdown("### 📝 Résumé généré des top 5 commentaires")
            st.success(t5_summary)

            st.subheader("7️⃣ Génération des déscriptions")
            st.subheader("🤖 Realistic Description Generation using LLM (LLaMA)")
            prompt = f"Voici une description initiale :\n{desc_en}\n\nVoici les 5 commentaires les plus proches :\n" + "\n".join([c[0] for c in top5]) + f"\n\nVoici un résumé de ces commentaires :\n{t5_summary}\n\nGénère une nouvelle description plus réaliste à partir de ces éléments."
            if st.button("Générer avec Groq", key="gen_groq_1"):
                with st.spinner("Génération en cours..."):
                    chat_completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="llama-3.3-70b-versatile",
                    )
                    st.markdown("### 🧠 Description générée via Groq")
                    st.success(chat_completion.choices[0].message.content)

elif menu == "🌐 Scraper URL VRBO":
    st.header("Scraping d'une page VRBO")
    url = st.text_input("Saisir l'URL d'une annonce VRBO :")
    scrape_trigger = st.button("Scraper", key="scrape_url")

    if scrape_trigger and url:
        service = Service("C:\\geckodriver\\geckodriver.exe")
        driver = webdriver.Firefox(service=service)
        wait = WebDriverWait(driver, 15)

        def scrape_listing(url):
            data = {"url": url, "description": "", "comments": []}
            try:
                driver.get(url)
                time.sleep(5)
                try:
                    wait.until(EC.presence_of_element_located((By.XPATH, "//div[@data-stid='content-markup']")))
                except:
                    st.warning("⚠️ Description non trouvée")

                soup = BeautifulSoup(driver.page_source, 'html.parser')
                desc = soup.find("div", {"data-stid": "content-markup"})
                if desc:
                    data["description"] = desc.get_text(strip=True)

                try:
                    see_reviews_btn = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable((By.XPATH, "//button[@data-stid='reviews-link']"))
                    )
                    driver.execute_script("arguments[0].click();", see_reviews_btn)
                    time.sleep(2)

                    for _ in range(5):
                        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                        time.sleep(2)

                    more_btns = driver.find_elements(By.XPATH, "//button[contains(text(), 'See more')]")
                    for btn in more_btns:
                        try:
                            driver.execute_script("arguments[0].click();", btn)
                            time.sleep(0.5)
                        except:
                            pass

                    soup = BeautifulSoup(driver.page_source, 'html.parser')
                    spans = soup.find_all("span", {"itemprop": "description"})
                    for span in spans:
                        txt = span.get_text(strip=True)
                        if txt and len(txt) > 10:
                            data["comments"].append(txt)
                except:
                    pass

            except Exception as e:
                st.error(f"Erreur : {e}")
            return data

        result = scrape_listing(url)
        driver.quit()
                
        if result["description"]:
            st.session_state["scraped_result"] = result  # Stocker en session

    if "scraped_result" in st.session_state:
        result = st.session_state["scraped_result"]
        df_scraped = pd.DataFrame({"Description": [result["description"]] * len(result["comments"]), "Commentaire": result["comments"]})
        st.subheader("📄 Résultats du scraping")
        st.dataframe(df_scraped)

        desc_clean = clean_text(result["description"])
        comments_clean = [clean_text(c) for c in result["comments"]]

        df_cleaned = pd.DataFrame({"Description nettoyée": [desc_clean] * len(comments_clean), "Commentaire nettoyé": comments_clean})
        st.subheader("🧼 Données nettoyées")
        st.dataframe(df_cleaned)

        desc_en = translate_to_english(desc_clean)
        comments_en = [translate_to_english(c) for c in comments_clean]

        st.markdown("GoogleTranslator for translations")
        st.markdown("### 📄 Description traduite")
        st.write(desc_en)
        st.markdown("### 💬 Commentaires traduits")
        for c in comments_en:
            st.write("-", c)

        similarities = [compute_similarity(desc_en, c) for c in comments_en]
        top5 = sorted(zip(comments_en, similarities), key=lambda x: x[1], reverse=True)[:5]

        st.markdown("### 🔝 Top 5 commentaires les plus similaires")
        st.markdown("**DistilBERT-based similarity computation")
        for i, (cmt, sim) in enumerate(top5):
            st.markdown(f"**{i+1}.** Similarité: {sim:.3f}<br>{cmt}", unsafe_allow_html=True)

        st.subheader("📝 Résumé des top 5 commentaires")
        combined_top5 = " ".join([c[0] for c in top5])
        t5_summary = summarize_text(combined_top5)
        st.success(t5_summary)

        st.subheader("🤖 Realistic Description Generation using LLM (LLaMA)")
        prompt = f"Voici une description initiale :\n{desc_en}\n\nVoici les 5 commentaires les plus proches :\n" + "\n".join([c[0] for c in top5]) + f"\n\nVoici un résumé de ces commentaires :\n{t5_summary}\n\nGénère une nouvelle description plus réaliste à partir de ces éléments."

        generate_trigger = st.button("Générer avec Groq", key="gen_groq_2")
        if generate_trigger:
            with st.spinner("Génération en cours..."):
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.3-70b-versatile",
                )
                st.markdown("### 🧠 Description générée via Groq")
                st.success(chat_completion.choices[0].message.content)
