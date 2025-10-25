# === Öğrenci Staj İlanı Eşleştiricisi (Final Sürüm) ===
# by Selim

# --- KÜTÜPHANELER ---
import pandas as pd
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# --- ADIM 1: VERİ YÜKLEME ---
df = pd.read_csv("staj_ilani_veri.csv")  # ✅ kendi CSV dosyan
print(f"Veri seti boyutu: {df.shape}")
print(df.head())

# --- ADIM 2: METİN TEMİZLEME ---
def temizle_metin(metin):
    metin = str(metin).lower()
    metin = re.sub(r'\d+', '', metin)
    metin = re.sub(r'[^\w\s]', '', metin)
    metin = metin.strip()
    return metin

df["temiz_metin"] = df["ilan_metni"].apply(temizle_metin)

# --- ADIM 3: TRAIN / TEST AYIRMA ---
X = df["temiz_metin"]
y = df["etiket"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- ADIM 4: MODELLER VE PARAMETRELER ---
modeller = [
    ("Naive Bayes", Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=['ve', 'ile', 'ama', 'için'])),
        ('model', MultinomialNB())
    ]), {
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'model__alpha': [0.1, 0.5, 1.0]
    }),

    ("SVM", Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=['ve', 'ile', 'ama', 'için'])),
        ('model', SVC(probability=True))
    ]), {
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'model__C': [0.1, 1, 10],
        'model__kernel': ['linear', 'rbf']
    }),

    ("Random Forest", Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=['ve', 'ile', 'ama', 'için'])),
        ('model', RandomForestClassifier(random_state=42))
    ]), {
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'model__n_estimators': [50, 100],
        'model__max_depth': [None, 10, 20]
    })
]

en_iyi_modeller = {}

# --- ADIM 5: MODEL EĞİTİMİ + GRIDSEARCH ---
for ad, pipeline, param_grid in modeller:
    print(f"\n{ad} modeli eğitiliyor...")
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print(f"✅ {ad} En iyi parametreler: {grid.best_params_}")
    print(f"✅ {ad} F1 (CV): {grid.best_score_:.4f}")
    en_iyi_modeller[ad] = grid.best_estimator_

# --- ADIM 6: TEST DEĞERLENDİRME ---
sonuclar = []
for ad, model in en_iyi_modeller.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = classification_report(y_test, y_pred, output_dict=True)['1']['f1-score']
    print(f"\n--- {ad} ---")
    print(classification_report(y_test, y_pred, target_names=['0 - Uygun Değil', '1 - Uygun']))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{ad} Karışıklık Matrisi")
    plt.show()
    sonuclar.append({'Model': ad, 'Accuracy': acc, 'F1-Score': f1})

sonuclar_df = pd.DataFrame(sonuclar).sort_values(by='F1-Score', ascending=False)
print("\nModel Performansları:")
print(sonuclar_df)

# --- ADIM 7: EN İYİ MODELİ KAYDET ---
en_iyi_model_adi = sonuclar_df.iloc[0]['Model']
final_model = en_iyi_modeller[en_iyi_model_adi]
joblib.dump(final_model, "en_iyi_model.pkl")
print(f"🎯 En iyi model kaydedildi: {en_iyi_model_adi}")

# --- ADIM 8: YENİ İLAN TAHMİNİ ---
yeni_ilan = [
    "Yazılım stajyeri olarak React projelerinde görev alacak, öğrenmeye açık öğrenciler arıyoruz.",
    "Kıdemli C# geliştirici aranıyor. En az 6 yıl tecrübe ve takım liderliği gereklidir."
]

for ilan in yeni_ilan:
    temiz = temizle_metin(ilan)
    tahmin = final_model.predict([temiz])[0]
    print(f"\n{ilan}\n→ Tahmin: {'Uygun (1)' if tahmin==1 else 'Uygun Değil (0)'}")
