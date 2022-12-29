import pandas as pd
# linear regression için gerekli modülümüzü import ettik
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from numpy import sqrt 

# csv türündeki dosyayı pandas kütüphanesiyle okuma
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# sütun başlıklarını yazdırma
print(train.columns)

# model oluşturma
model = LinearRegression()
X_train = train.drop('SalePrice', axis=1)
y_train = train.loc[:,'SalePrice']
model.fit(X_train,y_train)

X_test = test.drop('SalePrice', axis=1)
y_test = test.loc[:,'SalePrice']
predictions = model.predict(X_test)

rmse = sqrt(mean_squared_error(y_test, predictions))
print(rmse)

comparison = pd.DataFrame({"Actual Values": y_test,"Predictions": predictions})
# baştan istenilen kadar veri okuma
print(comparison.head(10))
# sondan istenilen kadar veri okuma
print(comparison.tail(7))

print(train.corr()["SalePrice"].sort_values(ascending=False).head(10))

correlations = train.corr()
print(correlations)

saleprice_correlations = correlations["SalePrice"]
print(saleprice_correlations)

print(saleprice_correlations.sort_values(ascending=False).head(10))

"""
GÖZETİMLİ ÖĞRENME(Supervised Learning):

Bu öğrenme çeşidinde algoritmalar öğrendiklerinden yola çıkarak tahmin yapmak için etiketli verileri kullanır. Eğitimde kullanılacak veriler önceden bilinir.
Bu bilgi ile sistem öğrenir ve yeni gelen veriyi yorumlar.
Buna göre sistem yaptığı hatalardan ders çıkararak bu hataları öğrenmek için kullanır.
Örneğin bir emlakçıyı ele alalım. Emlakçı bir stajyer’e ev fiyatlarının nasıl belirlendiğini
öğretmek istiyor. Bunun için daha önceden sattığı evlerin konumları, oda sayıları ve 
metrekareleri gibi bilgilerin yanında evlerin fiyatlarını da vererek satacağı diğer evlerin fiyatlarını belirlemesini istiyor. 
Bu şekilde çalışan bir algoritma Gözetimli Öğrenme algoritmasıdır.
"""