# %% veri setini içeriye aktar, preprocessing(padding)
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.datasets import imdb # veri seti
from tensorflow.keras.preprocessing.sequence import pad_sequences #padding için
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
## embedding: Modelin kelimeleri/simgeleri ANLAMLI sayılara dönüştürmesini sağlar.

from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report,roc_curve,auc

import keras_tuner as kt
from keras_tuner.tuners import RandomSearch

#KerasTuner → Hiperparametre arama kütüphanesi
#RandomSearch → Hiperparametreleri rastgele kombinasyonlarla dener
#Modelini manuel olarak ayarlamak yerine en iyisini otomatik bulur

import warnings
warnings.filterwarnings("ignore")

# veri seti yukle, imdb 50000 sample var. etiketleri : (0 =>olumsuz yorum), (1=>olumlu yorum)


# IMDb veri setinde her kelimeye kullanım sıklığına göre bir ID verilir; num_words=10000 ile en sık kullanılan 10.000 kelime alınır, nadir kelimeler atılır.
# ID kelimenin anlamını değil, sıklığını gösterir; embedding layer bu ID’leri anlamlı vektörlere çevirir.
(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words = 10000) # num_words:en çok kullanılan 10000 kelimeyi al


#her bir metne padding işlemi uygulamak veri ön işleme: yorumları aynı uzunluğu getirmek için padding yontemi kullanımı

#maxlen=100 → tüm yorumlar 100 kelime uzunluğuna getirilir
#Kısa yorumlar → başına veya sonuna 0 eklenir (default: başına)
#Uzun yorumlar → kesilir (truncate)

#her bir yorum 100 kelime içeriyor
maxlen = 100
x_train = pad_sequences(x_train, maxlen=maxlen) # train verisi uznluğu ayarla
x_test=pad_sequences(x_test,maxlen=maxlen) # test verisi uznluğu ayarla


# %% create and compile RNN modeli

def build_model(hp): #hp:hyperparameter
    model=Sequential() #base model
    
    #embedding katmanı: kelimeleri vektöre dönüştüren katman
    
    #input_dim=10000:Bu, modelin tanıyacağı kelime sayısı.
    #output_dim = hp.Int("embedding_output",min_value=32, max_value=128, step=32),:Bu, her kelimeyi kaç sayılık bir vektöre çevireceğini söyler
    #input_lengt= maxlen)):her bir yorum uzunluğu eşit olucak.
    
    model.add(Embedding(input_dim=10000,
                        output_dim = hp.Int("embedding_output",min_value=32, max_value=128, step=32), #32 64 96 128 olabilir
                        input_length= maxlen))
    
    
    # SimpleRNN katmanı: rnn katmani   
    #Girdi Alma: Kendisinden önceki katmandan (Embedding) gelen, sıraya dizilmiş kelime vektörlerini (dizi verisini) tek tek alır.

    #Hafıza Oluşturma: Her bir kelimeyi işlerken, bir önceki kelimeden gelen gizli durum (hidden state) adı verilen iç hafızayı kullanır
    #ve bu bilgiyi mevcut kelimeyle birleştirerek yeni bir gizli durum oluşturur.

    #Kapasite Belirleme: units (birim sayısı) parametresi (32, 64, 96, 128) bu gizli durum vektörünün boyutunu belirler.
    #Bu boyut, katmanın bir diziden ne kadar detaylı ve karmaşık bilgi (hafıza) saklayabileceğini gösterir.
    model.add(SimpleRNN(units=hp.Int("rnn_units",min_value=32,max_value=128,step=32)))# hafıza özeti detay boyutu belirleme 32 64 96 128 olabilir
    
    
    # dropout katmani: overffitingi engellemek için rastgele bazi nöronları kapatır.
    model.add(Dropout(hp.Float("dropout_rate", min_value=0.2, max_value=0.5, step=0.1)))#ëoverfittingi önlemek için 
    #simpleRNN katmani: rnn katmani
    
    
    #cikti katmani: 1 cell ve sigmoid
    model.add(Dense(1,activation="sigmoid"))
    
    
    #modelin derlenmesi
    model.compile(     
        optimizer=hp.Choice("optimizer",["adam","rmsprop"]), # adam veya rmsprop kullanılabilir
        loss="binary_crossentropy", # ikili sınıflandırma için kullanılan loss fonksiyonu
        metrics=["accuracy","AUC"]
        )
    
    return model
    

# %% hyperparamter search, model train

# hyperparametre search:random search ile hiperparametre aranacak.
tuner=RandomSearch(
    build_model, #optimize edilecek model fonksiyonu
    objective= "val_loss", #bu hp parametrelerini denerken val_lossu en çok düşüşüren en iyisi der. eğer val_accuracy kullansaydık vall_accuracy i en yüksek yapan değer eniyisidr.
    
    #Belirlediğim tüm hiperparametre aralıklarını kullanarak rastgele sadece 2 farklı model kombinasyonu dene ve bu ikisi arasından val_loss değeri en düşük olanı seç.
    max_trials=5, # maximum 2 farklı model deneyecek
    executions_per_trial=1, #her model için 1 eğitim denmesi yapılacak
    directory=".", 
    project_name="imdb_rnn" #projenin adı
    )

# early stopping: val loss a göre erken durdurma.vall loss duzelmezse (azalmazsa) eğitimi durdur.
early_stopping=EarlyStopping(monitor="val_loss",patience=3,restore_best_weights=True)#monitor val_lossu izleyecek 3 epoch boyunca iyleşme azalmazsa durduracak True ise en iyi modelin ağırlıklarını geri yükleyecek.


# model training:model eğitimi:tuner.search() birden çok modelin eğitimini yöneten bir komutken, model.fit() tek bir modelin eğitimini başlatan komuttur.
tuner.search(x_train,y_train,
             epochs=15,
             validation_split=0.2,# eğitim veri setinin  %20 si validation olacak
             callbacks=[early_stopping]
             )




# %% evulate best  model

# en iyi modelin alınması
 
#get_best_models(num_models=1): Bana birinci en iyi modeli içeren bir liste ver.
#[0]: O listedeki ilk (ve tek) modeli al ve kullan. 
best_model=tuner.get_best_models(num_models=1)[0] # en iyi performans gösteren model

# en iyi modeli kullanarak test et
#model.evaluate() metodunun çıktısı, modelinizi derlerken (model.compile()) belirttiğiniz metriklerin tam sırasını takip eder:
loss,accuracy,AUC=best_model.evaluate(x_test,y_test)
print(f"test loss:{loss:.3f}, accuracy:{accuracy:.3f}, test auc:{AUC:.3F}")

# tahmin yapma ve modelin performansını değerlendirme
y_pred_prob=best_model.predict(x_test) # 0 ile 1 arasında tahmin değerleri gelir 0.5 ten küçükse olumsuz 0.5 ten büyükse olumlu
y_pred= (y_pred_prob>0.5).astype("int32") # tahmin edilen değer 0.5 ten buyukse 1 e yuvarlanır yani olumlu küçükse 0 a yuvarlanır yani olumsuz.


#precision, recall, f1-score değerleri tablosu
print(classification_report(y_test, y_pred))

# roc eğrisi hesaplama
fpr,tpr,_=roc_curve(y_test, y_pred_prob) # roc eğrisi için fpr(false positive rate) ve tpr(true pozitive rate hesaplanır.)
roc_auc=auc(fpr,tpr) # roc eğrisin altında kalan alan hesaplanır

# roc eğrisini görselleştirme
plt.figure()
plt.plot(fpr,tpr,color="darkorange",label="roc curve (area=%0.2f)" % roc_auc) #roc eğrisi
plt.plot([0,1],[0,1],color="blue",linestyle="--") #rastgele tahmin çizgisi
plt.xlim([0,1])
plt.ylim([0,1.05])
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("roc eğrisi")
plt.legend()
plt.show()

