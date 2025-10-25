import pandas as pd
import re
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- ADIM 1: Veri Yükleme ---
# 3 kişilik ekibinizin topladığı verileri buraya yükleyin.
# Örnek olarak küçük bir veri seti (DataFrame) oluşturuyorum.
# Gerçek projede: df = pd.read_csv('topladiginiz_ilanlar.csv') kullanabilirsiniz.
data = {
    'metin': [
        "Temel C# ve Unity bilgisine sahip, öğrenmeye hevesli stajyer arıyoruz.", # Junior/Stajyer (1)
        "React ve Node.js konusunda en az 5 yıl tecrübeli senior developer aranıyor.", # Senior (0)
        "Yeni mezun, kendini geliştirmek isteyen, SQL bilen junior yazılımcı.", # Junior/Stajyer (1)
        "Ekibimize liderlik edecek, AWS ve mikroservis mimarilerinde uzman lead engineer.", # Senior (0)
        "Firmamız bünyesinde yetiştirilmek üzere stajyer alınacaktır.", # Junior/Stajyer (1)
        "GoLang ve Kubernetes ile 3+ yıl deneyimli backend developer.", # Senior (0)
        "Python, Django bilgisi olan ve makine öğrenmesi projelerinde yer almak isteyen yeni mezun." # Junior/Stajyer (1)
    ],
    'etiket': [1, 0, 1, 0, 1, 0, 1] # 1 = Uygun (Staj/Junior), 0 = Uygun Değil (Senior)
}
df = pd.DataFrame(data)

print(f"Veri seti boyutu: {df.shape}")
print("\nVeri Seti Dağılımı (Etiketler):")
print(df['etiket'].value_counts())


# --- ADIM 2: Veri Ön İşleme ve EDA (Kılavuz Bölüm C) ---
# Kılavuz, "eksik değerleri, yinelenenleri" [cite: 31] ve "kategorik değişkenleri" [cite: 32] ele almayı belirtir.
# NLP için bu, metin temizlemeyi içerir.

def temizle_metin(metin):
    """Temel metin temizleme fonksiyonu"""
    metin = metin.lower() # Küçük harfe çevir
    metin = re.sub(r'\d+', '', metin) # Sayıları kaldır
    metin = re.sub(r'[^\w\s]', '', metin) # Noktalamayı kaldır
    metin = metin.strip() # Boşlukları kaldır
    return metin

df['temiz_metin'] = df['metin'].apply(temizle_metin)

# ÖNEMLİ: Gerçek "Stage 2 EDA Submission"  için kılavuzda belirtilen
# "görsel özetler (korelasyon grafikleri, histogramlar, kutu grafikleri)" 
# gibi daha detaylı analizler (örn: Word Cloud, en sık geçen kelimeler) eklemelisiniz.

print("\nTemizlenmiş Veri Örneği:")
print(df[['temiz_metin', 'etiket']].head())


# --- ADIM 3: Veri Setini Ayırma ---
X = df['temiz_metin']
y = df['etiket']

# Veriyi %80 train, %20 test olarak ayıralım
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# --- ADIM 4: Model Uygulama (Kılavuz Bölüm D) ---
# Kılavuz, "farklı ailelerden en az üç model" [cite: 35] ve
# "hiperparametre ayarı (GridSearchCV)" [cite: 40] kullanılmasını istiyor.

# 1. Model: Multinomial Naive Bayes (Genellikle metin için iyi bir temel modeldir)
pipeline_nb = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=['ve', 'ile', 'ama', 'için'])), # Basit Türkçe stop words
    ('model', MultinomialNB())
])

# Naive Bayes için Hiperparametre Ağı
param_grid_nb = {
    'tfidf__ngram_range': [(1, 1), (1, 2)], # Tekli kelimeler veya ikili kelime grupları
    'model__alpha': [0.1, 0.5, 1.0] # Laplace smoothing parametresi
}

# 2. Model: Support Vector Machine (SVM) (Metin sınıflandırmada güçlüdür)
pipeline_svm = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=['ve', 'ile', 'ama', 'için'])),
    ('model', SVC(probability=True)) # Olasılıklar için probability=True
])

# SVM için Hiperparametre Ağı
param_grid_svm = {
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'model__C': [0.1, 1, 10], # Düzenlileştirme parametresi
    'model__kernel': ['linear', 'rbf'] # Denenecek kernel tipleri
}

# 3. Model: Random Forest (Kılavuzda belirtilen bir "Ensemble" yöntem )
pipeline_rf = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=['ve', 'ile', 'ama', 'için'])),
    ('model', RandomForestClassifier(random_state=42))
])

# Random Forest için Hiperparametre Ağı
param_grid_rf = {
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'model__n_estimators': [50, 100], # Ağaç sayısı
    'model__max_depth': [None, 10, 20] # Ağaç derinliği
}

# Modelleri ve parametre ağlarını bir listede toplayalım
modeller = [
    ("Naive Bayes", pipeline_nb, param_grid_nb),
    ("SVM", pipeline_svm, param_grid_svm),
    ("Random Forest", pipeline_rf, param_grid_rf)
]

# En iyi modelleri saklamak için bir sözlük
en_iyi_modeller = {}

print("\n--- Model Eğitimi ve Hiperparametre Ayarı Başlatılıyor ---")

for ad, pipeline, param_grid in modeller:
    print(f"\n{ad} modeli için GridSearchCV çalıştırılıyor...")
    
    # Kılavuzda belirtildiği gibi cross-validation [cite: 38] (cv=5) ve 
    # hiperparametre ayarı [cite: 40] (GridSearchCV) yapılıyor.
    # Kılavuz metrik olarak 'Accuracy, Precision, Recall, F1' [cite: 42] belirttiği için 'f1' 'e göre optimize edelim.
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    print(f"{ad} için en iyi parametreler: {grid_search.best_params_}")
    print(f"{ad} için en iyi F1 skoru (CV): {grid_search.best_score_:.4f}")
    
    # En iyi modeli sakla
    en_iyi_modeller[ad] = grid_search.best_estimator_

print("\n--- Model Eğitimi Tamamlandı ---")


# --- ADIM 5: Sonuçlar ve Tartışma (Kılavuz Bölüm E) ---
# Modelleri test verisi üzerinde değerlendirip karşılaştıralım 

print("\n--- Test Seti Değerlendirme Sonuçları ---")

results = []

for ad, model in en_iyi_modeller.items():
    y_pred = model.predict(X_test)
    
    # Kılavuzda istenen metrikler [cite: 42]
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n--- {ad} Model Raporu ---")
    print(classification_report(y_test, y_pred, target_names=['0 - Uygun Değil', '1 - Uygun']))
    
    results.append({
        'Model': ad,
        'Accuracy': acc,
        'F1-Score (Class 1)': classification_report(y_test, y_pred, output_dict=True)['1 - Uygun']['f1-score']
    })
    
    # Karışıklık Matrisi (Confusion Matrix) - Rapor  ve Sunum  için görsel
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Tahmin 0', 'Tahmin 1'], 
                yticklabels=['Gerçek 0', 'Gerçek 1'])
    plt.title(f'{ad} Karışıklık Matrisi')
    plt.ylabel('Gerçek Etiket')
    plt.xlabel('Tahmin Edilen Etiket')
    plt.show()

# Modelleri karşılaştırmak için tablo 
results_df = pd.DataFrame(results).sort_values(by='F1-Score (Class 1)', ascending=False)
print("\n--- Modellerin Karşılaştırmalı Tablosu (Test Verisi) ---")
print(results_df)


# --- ADIM 6: Yeni Bir Veri ile Tahmin Yapma ---
print("\n--- Yeni İlan Tahmini Örneği ---")

# En iyi modeli seç (Örn: F1'e göre en yüksek olan)
en_iyi_model_adi = results_df.iloc[0]['Model']
final_model = en_iyi_modeller[en_iyi_model_adi]

print(f"Kullanılan final model: {en_iyi_model_adi}")

# Örnek Yeni İlan Metinleri
yeni_ilan_1 = "Şirketimizde java ve spring boot bilen, en az 6 yıl deneyimli kıdemli yazılımcı arayışımız bulunmaktadır."
yeni_ilan_2 = "Photoshop bilen, yaratıcı, yetiştirilmek üzere junior grafiker stajyer arıyoruz."

# Metinleri temizle
temiz_ilan_1 = temizle_metin(yeni_ilan_1)
temiz_ilan_2 = temizle_metin(yeni_ilan_2)

# Tahmin yap
tahmin_1 = final_model.predict([temiz_ilan_1])[0]
tahmin_2 = final_model.predict([temiz_ilan_2])[0]

# Tahmin olasılıklarını al (eğer model destekliyorsa, SVM ve RF destekler)
try:
    olasilik_1 = final_model.predict_proba([temiz_ilan_1])[0]
    olasilik_2 = final_model.predict_proba([temiz_ilan_2])[0]
    
    print(f"İlan 1 Tahmin: {'Uygun (1)' if tahmin_1 == 1 else 'Uygun Değil (0)'} (Skor: {max(olasilik_1):.2f})")
    print(f"İlan 2 Tahmin: {'Uygun (1)' if tahmin_2 == 1 else 'Uygun Değil (0)'} (Skor: {max(olasilik_2):.2f})")

except AttributeError: # Naive Bayes 'predict_proba'yı farklı kullanabilir
    print(f"İlan 1 Tahmin: {'Uygun (1)' if tahmin_1 == 1 else 'Uygun Değil (0)'}")
    print(f"İlan 2 Tahmin: {'Uygun (1)' if tahmin_2 == 1 else 'Uygun Değil (0)'}")
