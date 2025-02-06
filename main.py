import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import seaborn as sns
import yaml



with open('main.yaml', 'r') as file:
    loaded_data = yaml.safe_load(file)

settings = loaded_data.get('settings', {})
models = loaded_data.get('models', {})

kmeans_features = models.get('kmeans', {}).get('features', [])
if kmeans_features:
    print("K-means özellikleri:", kmeans_features)
else:
    print("K-means özellikleri bulunamadı.")

matplotlib_style = settings.get('matplotlib_style', None)
if matplotlib_style:
    print("Matplotlib stili:", matplotlib_style)
else:
    print("Matplotlib stili belirtilmemiş.")
# Matplotlib ayarları
import matplotlib
plt.style.use('seaborn-v0_8')

# Veri setini okuma
data = pd.read_csv('accidents_2017_to_2023_english.csv', low_memory=False)

# İlk birkaç satırı yazdırma
print("Veri setinin ilk birkaç satırı:")
print(data.head(5))

# Genel istatistiksel bilgileri yazdırma
descriptive_statistics = data.describe()
print("\nGenel İstatistiksel Bilgiler:")
print(descriptive_statistics)

# Null değerlerin sayısını yazdırma
null_statistics = data.isnull().sum()
print("\nNull Değerlerin Sayısı:")
print(null_statistics)

# df dataların temizlenmiş halini temsil ediyor
df = data.dropna()

# 'km' sütunundaki virgülleri noktaya çevirip sayısal türe dönüştürme
df.loc[:, 'km'] = df['km'].str.replace(',', '.').astype(float)

# Saat ve zaman bilgisini datetime formatına çevirme
df.loc[:, 'hour'] = pd.to_datetime(df['hour'], format='%H:%M:%S').dt.hour  # Sadece saat bilgisi alıyoruz

# Tarih bilgisini datetime formatına çevirme
df['inverse_data'] = pd.to_datetime(df['inverse_data'], errors='coerce')

# Dönüştürülmüş veriyi yazdırma
print("Dönüştürülmüş Veri:")
print(df.head(13))

# K-means için kullanılacak özellikler
features_for_clustering = ['road_id', 'km', 'latitude', 'longitude', 'total_injured', 'hour', 'people']
X = df[features_for_clustering].copy()

# Verileri ölçeklendirme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Metodu için
inertias = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Elbow Grafiği
plt.figure(figsize=(12, 8))
plt.plot(K, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Küme Sayısı (k)', fontsize=12)
plt.ylabel('Inertia', fontsize=12)
plt.title('Elbow Metodu ile Optimal Küme Sayısı Belirleme', fontsize=14)
plt.grid(True)

# Optimal küme sayısını belirleme
knee_locator = KneeLocator(K, inertias, curve='convex', direction='decreasing')
optimal_k = knee_locator.elbow
print(f"Optimal küme sayısı: {optimal_k}")

# Optimal noktayı grafikte gösterme
plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k: {optimal_k}')
plt.legend()
plt.tight_layout()
plt.show()

# K-means modeli oluşturma ve eğitme
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# Kümeleri tahmin etme
clusters = kmeans.predict(X_scaled)

# Kümeleri görselleştirme
plt.figure(figsize=(12, 8))
plt.scatter(df['total_injured'], df['hour'], c=clusters, cmap='viridis', marker='o', alpha=0.5)
plt.scatter(kmeans.cluster_centers_[:, features_for_clustering.index('total_injured')],
            kmeans.cluster_centers_[:, features_for_clustering.index('hour')],
            s=300, c='red', marker='x', label='Küme Merkezleri')
plt.title('K-Means Kümeleme Sonuçları (KM ve Saat)')
plt.xlabel('KM')
plt.ylabel('Saat')
plt.legend()
plt.show()

# Regresyon modeli için kullanılacak özellikler ve hedef değişken
features_for_regression = ['km', 'latitude', 'longitude', 'hour', 'people', 'deaths','road_id','vehicles_involved']
target_variable = 'total_injured'

# Özellikler ve hedef değişkeni ayırma
X_reg = df[features_for_regression]
y_reg = df[target_variable]

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Lineer regresyon modeli oluşturma ve eğitme
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Tahmin yapma
y_pred = regressor.predict(X_test)

# Modelin performansını değerlendirme
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nRegresyon Modeli Performansı:")
print(f"Ortalama Kare Hata (MSE): {mse}")
print(f"R-kare (R^2): {r2}")

# Modelin katsayılarını yazdırma
print("\nModel Katsayıları:")
for feature, coef in zip(features_for_regression, regressor.coef_):
    print(f"{feature}: {coef}")

# Modelin kesişim noktası
print(f"Kesişim Noktası: {regressor.intercept_}")

# Regresyon modelinin tahminlerini ve gerçek değerleri karşılaştıran grafik
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')
plt.title('Gerçek vs Tahmin Edilen Değerler (Lineer Regresyon)')
plt.show()

# Karar ağacı modeli oluşturma ve eğitme
tree_regressor = DecisionTreeRegressor(random_state=42)
tree_regressor.fit(X_train, y_train)

# Karar ağacı ile tahmin yapma
y_tree_pred = tree_regressor.predict(X_test)

# Karar ağacı modelinin performansını değerlendirme
tree_mse = mean_squared_error(y_test, y_tree_pred)
tree_r2 = r2_score(y_test, y_tree_pred)
tree_mae = mean_absolute_error(y_test, y_tree_pred)

print(f"\nKarar Ağacı Modeli Performansı:")
print(f"Ortalama Kare Hata (MSE): {tree_mse}")
print(f"R-kare (R^2): {tree_r2}")
print(f"Ortalama Mutlak Hata (MAE): {tree_mae}")

# Karar ağacı modelinin tahminlerini ve gerçek değerleri karşılaştıran grafik
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_tree_pred, alpha=0.5, color='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')
plt.title('Gerçek vs Tahmin Edilen Değerler (Karar Ağacı)')
plt.show()

# Sadece sayısal sütunları seçme
numeric_df = df.select_dtypes(include=[np.number])

# Korelasyon matrisi oluşturma ve görselleştirme
correlation_matrix = numeric_df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Korelasyon Matrisi')
plt.show()

# Kaza nedenlerinin dağılımı
cause_counts = df['cause_of_accident'].value_counts()

# Bar grafiği oluşturma
plt.figure(figsize=(20, 12))
cause_counts.plot(kind='bar', color='skyblue')
plt.title("Kazaların Nedenleri", fontsize=14)
plt.xlabel("Kaza Nedeni", fontsize=10)
plt.ylabel("Kaza Sayısı", fontsize=10)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.tight_layout()
plt.show()

# Pasta grafiği oluşturma
plt.figure(figsize=(12, 12))
cause_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
plt.title("Kazaların Nedenlerinin Dağılımı", fontsize=14)
plt.ylabel("")
plt.tight_layout()
plt.show()

# 'Animals on the road' kazaları analizi
animals_on_road_df = df[df['cause_of_accident'].str.contains('animals on the road', case=False)]

# Kazalarda ölen ve yaralanan kişileri hesaplayın
animals_on_road_df.loc[:, 'total_injured'] = animals_on_road_df['slightly_injured'] + animals_on_road_df['severely_injured']
animals_on_road_df.loc[:, 'total_affected'] = animals_on_road_df['deaths'] + animals_on_road_df['total_injured']

# Road ID'ye göre ölü ve yaralı sayılarının toplamını hesaplayın
road_injuries = animals_on_road_df.groupby('road_id').agg(
    total_deaths=('deaths', 'sum'),
    total_slightly_injured=('slightly_injured', 'sum'),
    total_severely_injured=('severely_injured', 'sum'),
    total_injured=('total_injured', 'sum'),
    total_affected=('total_affected', 'sum')
)

# Grafik oluşturun
plt.figure(figsize=(20, 12))
road_injuries[['total_deaths', 'total_injured']].plot(kind='bar', stacked=True, color=['red', 'orange'], ax=plt.gca())
plt.title('Animals on the Road - Road ID Injury and Death Distribution', fontsize=14)
plt.xlabel('Road ID', fontsize=10)
plt.ylabel('Number of People Affected', fontsize=10)
plt.xticks(rotation=45, fontsize=8)
plt.tight_layout()
plt.show()

# "driver's lack of attention to conveyance" nedenine bağlı kazaları filtreleme
attention_lack_df = df[df['cause_of_accident'].str.contains("driver's lack of attention to conveyance", case=False)]

# Saat ve yol kimliğine göre kazaların sayısını hesaplama
attention_lack_pivot = attention_lack_df.pivot_table(index='hour', columns='road_id', values='cause_of_accident', aggfunc='count', fill_value=0)

# Isı haritası oluşturma
plt.figure(figsize=(14, 10))
sns.heatmap(attention_lack_pivot, cmap='YlGnBu', annot=False, cbar=True)
plt.title("Driver's Lack of Attention to Conveyance - Saat ve Yol ID'ye Göre Kaza Dağılımı")
plt.xlabel('Yol ID')
plt.ylabel('Saat')
plt.show()

# Hava durumu sütununu kontrol edin (örneğin, 'weather_condition' olarak varsayıyorum)
# Hava durumu ve kaza sayıları arasındaki ilişkiyi inceleme
weather_accidents = df['wheather_condition'].value_counts()

# Bar grafiği oluşturma
plt.figure(figsize=(12, 8))
weather_accidents.plot(kind='bar', color='skyblue')
plt.title("Hava Durumuna Göre Kaza Sayıları")
plt.xlabel("Hava Durumu")
plt.ylabel("Kaza Sayısı")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Saat ve gün bazında kaza yoğunluğu
hour_day_pivot = df.pivot_table(index='hour', columns='week_day', values='cause_of_accident', aggfunc='count', fill_value=0)

# Isı haritası oluşturma
plt.figure(figsize=(12, 8))
sns.heatmap(hour_day_pivot, cmap='YlOrRd', annot=False, cbar=True)
plt.title('Saat ve Güne Göre Kaza Yoğunluğu', fontsize=14)
plt.xlabel('Gün', fontsize=10)
plt.ylabel('Saat', fontsize=10)
plt.xticks(rotation=45, fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.show()

# Günlere göre kaza sayısını hesaplama
daily_accidents = df['week_day'].value_counts().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

# Bar grafiği oluşturma
plt.figure(figsize=(12, 6))
daily_accidents.plot(kind='bar', color='skyblue')
plt.title('Günlere Göre Kaza Dağılımı')
plt.xlabel('Gün')
plt.ylabel('Kaza Sayısı')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Yaralanma şiddetine göre dağılım
injury_severity = df[['slightly_injured', 'severely_injured', 'deaths']].sum()

# Bar grafiği oluşturma
plt.figure(figsize=(12, 6))
injury_severity.plot(kind='bar', color=['yellow', 'orange', 'red'])
plt.title('Yaralanma Şiddetine Göre Dağılım')
plt.xlabel('Yaralanma Tipi')
plt.ylabel('Kişi Sayısı')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Aylara göre kaza sayısını hesaplama
df['month'] = df['inverse_data'].dt.month
monthly_accidents = df['month'].value_counts().sort_index()

# Çizgi grafiği oluşturma
plt.figure(figsize=(12, 6))
plt.plot(monthly_accidents.index, monthly_accidents.values, marker='o')
plt.title('Aylara Göre Kaza Dağılımı')
plt.xlabel('Ay')
plt.ylabel('Kaza Sayısı')
plt.grid(True)
plt.show()

# Saatlere göre kaza sayısını hesaplama
hourly_accidents = df['hour'].value_counts().sort_index()

# Çizgi grafiği oluşturma
plt.figure(figsize=(12, 6))
plt.plot(hourly_accidents.index, hourly_accidents.values, marker='o')
plt.title('Saatlere Göre Kaza Dağılımı')
plt.xlabel('Saat')
plt.ylabel('Kaza Sayısı')
plt.grid(True)
plt.show()
import yaml


