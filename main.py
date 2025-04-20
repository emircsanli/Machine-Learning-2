from statistics import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler



#Proje 7 Dizi film vb öneri
#columns_name=["user_id","item_id","rating","timestamp"]
#df=pd.read_csv('users.data',sep="\t",names=columns_name)

#movie_titles=pd.read_csv("movie_id_titles.csv")
#df=pd.merge(df,movie_titles,on="item_id")

#moviemat=df.pivot_table(index="user_id",columns="title",values="rating")#titlları sütuna dönüştürdü rating değerleri doldurdu
#starwars_rating=moviemat["Star Wars (1977)"]
#starwars_rating = starwars_rating.dropna()

#similar_starwars_rating=moviemat.corrwith(starwars_rating) #star warsa göre colerasyonunu hesaplıyor
#corr_starwars=pd.DataFrame(similar_starwars_rating,columns=["Correlation"])
#corr_starwars.dropna(inplace=True)
#df.drop(["timestamp"],axis=1,inplace=True)

#rating=pd.DataFrame(df.groupby("title")["rating"].mean())

#rating["rating_number"]=df.groupby("title")["rating"].count()

#corr_starwars=corr_starwars.join(rating["rating_number"])
#filter_corr_starwars=corr_starwars[corr_starwars["rating_number"] > 100].sort_values(by="Correlation",ascending=False)
#print(filter_corr_starwars.head(10))

#Proje6 Müşreti Gruplama (unsupervised learning)

#df=pd.read_csv("Avm_Musterileri.csv")
#plt.scatter(df["Annual Income (k$)"],df["Spending Score (1-100)"])
#plt.xlabel("Annual Income")
#plt.ylabel("Spending Score")
##plt.show()
#df.rename(columns={"Annual Income (k$)":"income"}, inplace=True)
#df.rename(columns={"Spending Score (1-100)":"score"}, inplace=True)

#scaler=MinMaxScaler()
#scaler.fit(df[["income"]])
#df["income"]=scaler.transform(df[["income"]])
#scaler.fit(df[["score"]])
#df["score"]=scaler.transform(df[["score"]])

#K nın belirlenmesi
#k_range=range(1,11)
#dist=[]
#for k in k_range:
#    k_model=KMeans(n_clusters=k)
#    k_model.fit(df[["income","score"]])
#    dist.append(k_model.inertia_)
#plt.xlabel("K")
#plt.ylabel("Inertia")
#plt.plot(k_range,dist)
#plt.show()

#k_model = KMeans(n_clusters=5)
#y=k_model.fit_predict(df[["income", "score"]])
#df["cluster"]=y
#k_model.cluster_centers_

#df1=df[df["cluster"]==0]
#df2=df[df["cluster"]==1]
#df3=df[df["cluster"]==2]
#df4=df[df["cluster"]==3]
#df5=df[df["cluster"]==4]

#plt.xlabel("income")
#plt.ylabel("score")
#optimal_k=5
#plt.scatter(df1["income"], df1["score"], color="green", label="Cluster 0")
#plt.scatter(df2["income"], df2["score"], color="red", label="Cluster 1")
#plt.scatter(df3["income"], df3["score"], color="black", label="Cluster 2")
#plt.scatter(df4["income"], df4["score"], color="orange", label="Cluster 3")
#plt.scatter(df5["income"], df5["score"], color="purple", label="Cluster 4")
#plt.scatter(k_model.cluster_centers_[:, 0], k_model.cluster_centers_[:, 1], color="blue", marker="x", label="Centroids")
#plt.legend()
#plt.show()
#colors = ["green", "red", "black", "orange", "purple"]
#for i in range(optimal_k):
#    cluster_data = df[df["cluster"] == i]
#    plt.scatter(cluster_data["income"], cluster_data["score"], color=colors[i], label=f"Cluster {i}")

#plt.scatter(k_model.cluster_centers_[:, 0], k_model.cluster_centers_[:, 1], color="blue", marker="x", label="Centroids")
#plt.legend()
#plt.show()
#Proje 5 İş başvuru yapay zeka

#df=pd.read_csv('DecisionTreesClassificationDataSet.csv')
#duzelt={"Y":1,"N":0}
#df["IseAlindi"]=df["IseAlindi"].map(duzelt)
#df["SuanCalisiyor?"]=df["SuanCalisiyor?"].map(duzelt)
#df["Top10 Universite?"]=df["Top10 Universite?"].map(duzelt)
#df["StajBizdeYaptimi?"]=df["StajBizdeYaptimi?"].map(duzelt)
#learn={"Bs":0,"MS":1,"PhD":2}
#df["Egitim Seviyesi"]=df["Egitim Seviyesi"].map(learn)
#y=df["IseAlindi"]
#x=df.drop("IseAlindi",axis=1)

#clf=tree.DecisionTreeClassifier()

#clf=clf.fit(x,y)
#print(clf.predict([[5,1,3,0,0,0]]))

#clf=clf.fit(x,y)
#new_data = pd.DataFrame([[0, 0, 0, 1, 0,0]], columns=x.columns) #model eğitilirken sütun isimleri var tahminde de karşıklık olmaması için olmalı
#print(clf.predict(new_data))
#Proje 4 Iris çiçeği
#url="pca_iris.data"
#df=pd.read_csv(url,names=["sepal lenght","sepal width","petal lenght","petal width","target"])

#features=["sepal lenght","sepal width","petal lenght","petal width"]
#x=df[features]
#y=df["target"]
#x=StandardScaler().fit_transform(x) #scale işlemi yapması için benzer aralığa getirir değerleri

#4 boyuttan 2 ye indiriyoruz yeni 2 boyuta indiriyor
#pca=PCA(n_components=2)
#principalComponents=pca.fit_transform(x)
#principalDf=pd.DataFrame(data=principalComponents,columns=["principal component1","principal component2"])
#finaldf=pd.concat([principalDf,df[["target"]]],axis=1)
#print(finaldf)

#dfsetosa=finaldf[finaldf["target"]=="Iris-setosa"]
#dfvirginica=finaldf[finaldf["target"]=="Iris-virginica"]
#dfversicolor=finaldf[finaldf["target"]=="Iris-versicolor"]
#plt.xlabel("principal component1")
#plt.ylabel("principal component2")
#plt.scatter(dfsetosa["principal component1"],dfsetosa["principal component2"],color="green")
#plt.scatter(dfvirginica["principal component1"],dfvirginica["principal component2"],color="red")
#plt.scatter(dfversicolor["principal component1"],dfversicolor["principal component2"],color="blue")
#plt.show()
#kolay yol

#targets=["Iris-setosa","Iris-versicolor","Iris-virginica"]
#colors=["green","blue","red"]

#for target,col in zip(targets,colors):
#    dftemp=finaldf[finaldf["target"]==target]
#    plt.scatter(dftemp["principal component1"],dftemp["principal component2"],color=col)

#plt.show()

#lose=pca.explained_variance_ratio_
#print(lose)
#totallose=pca.explained_variance_ratio_.sum()
#print(totallose)

#Proje 3 Maaş skalası
#df=pd.read_csv('polynomial.csv',sep=";")
#dereceyi belirtiyoruz başta 2 ye kadar
#poly=PolynomialFeatures(degree=2)

#plt.scatter(df.deneyim,df.maas)
#plt.xlabel("Deneyim")
#plt.ylabel("Maas")
#plt.show()

#reg=LinearRegression()
#reg.fit(df[["deneyim"]],df["maas"])
#plt.scatter(df.deneyim, df.maas, color='blue', label='Veri noktaları')
#x=df.deneyim
#y=df.maas
#plt.plot(x,y,color='red',label="linear")
#plt.xlabel("Deneyim")
#plt.ylabel("Maas")
#plt.title("Deneyim ve Maaş İlişkisi")
#plt.legend()
#plt.show()

#poly = PolynomialFeatures(degree=2)
#x_poly = poly.fit_transform(df[["deneyim"]])
#reg = LinearRegression()
#reg.fit(x_poly, df["maas"])
#y_poly_pred = reg.predict(x_poly)

# Veri noktalarını ve polinomal regresyonu görselleştir
#plt.scatter(df.deneyim, df.maas, color='blue', label='Veri noktaları')
#plt.plot(df.deneyim, y_poly_pred, color='green', label="Polinomal Regresyon")
#plt.xlabel("Deneyim")
#plt.ylabel("Maaş")
#plt.title("Deneyim ve Maaş İlişkisi")
#plt.legend()
#plt.show()

# 4.5 yıllık deneyim için tahmin yapma
#experience = pd.DataFrame({'deneyim': [4.5]})
#experience_poly = poly.transform(experience)
#salary_pred = reg.predict(experience_poly)
#print(f"4.5 yıllık deneyim için tahmin edilen maaş: {salary_pred[0]}")

#diabetes example Proje1

#data=pd.read_csv('diabetes.csv')

#sick=data[data.Outcome==1]
#well=data[data.Outcome==0]
#plt.scatter(well.Age,well.Glucose,color='green',label='well',alpha=0.5) #alpha koyuluk
#plt.scatter(sick.Age,sick.Glucose,color='red',label='diabetes',alpha=0.5)
#plt.xlabel('Age')
#plt.ylabel('Glucose')
#plt.legend()
#plt.show()

#y=data.Outcome.values
#x=data.drop('Outcome',axis=1)
#x2=(x-np.min(x))/(np.max(x)-np.min(x))#algoritma düzgün çalışsın diye normalizasyon
#print(x.head())
#print(x2.head())

#x_train,x_test,y_train,y_test=train_test_split(x2,y,test_size=0.2,random_state=1)

#knn=KNeighborsClassifier(n_neighbors=3)  #k değerini verdik
#knn.fit(x_train,y_train)  #fit ayarla eğit
#predict=knn.predict(x_test)
#print(knn.score(x_test,y_test))

#en iyi k için
#i=1
#for k in range(1,11):
#    knn_new=KNeighborsClassifier(n_neighbors=k)
#    knn_new.fit(x_train,y_train)
    #print(i," " ,"Doğruluk %:", knn_new.score(x_test,y_test)*100)
#    i+=1


#multi lineer rengression Proje2

#df=pd.read_csv('multilinearregression.csv',sep=";")

#reg=linear_model.LinearRegression()
#reg.fit(df[['alan','odasayisi','binayasi']],df['fiyat']) #ilk köşeli parantez bağımsız değişkenler
#input_data = pd.DataFrame([[230, 4, 10],[250,5,10],[300,6,0]], columns=["alan", "odasayisi", "binayasi"])
#predicted_price = reg.predict(input_data)
#print(predicted_price)

#print(reg.coef_)#katsayıları hesapladı
#print(reg.intercept_)#sabit değeri hesapladı


# Yeni bir hasta tahmini için:
#from sklearn.preprocessing import MinMaxScaler

# normalization yapıyoruz - daha hızlı normalization yapabilmek için MinMax  scaler kullandık...
#sc = MinMaxScaler()
#sc.fit_transform(x2)

#new_prediction = knn.predict(sc.transform(np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])))
#new_prediction[0]

"""""
plt.plot([1,2,3],[1,4,9])

x=[1,2,3]
y=[1,4,9]

plt.plot(x,y)

plt.xticks(x)
plt.yticks(y)
plt.title("Scatter Plot")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x,y)
plt.show()

plt.plot(x,y,label="x*2",color="red")
plt.xticks([1,2,3,4,5])
plt.yticks([1,4,9,16,25])
plt.title("Scatter Plot")
plt.xlabel("x")
plt.ylabel("y")
plt.legend() #x*2 için grafiğin üzerinde gözüksün diye
plt.show()

plt.plot(x,y,label="x*2",color="red",linewidth=2,linestyle="dashed",marker="o")
plt.xticks([1,2,3,4,5])
plt.yticks([1,4,9,16,25])
plt.title("Scatter Plot")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

plt.plot(x,y,label="x*2",color="red",linewidth=2,linestyle="dashed",marker="o")
plt.xticks([1,2,3,4,5]) #x eksenine yazılacak değerler
plt.yticks([1,4,9,16,25])
plt.title("Scatter Plot")
plt.xlabel("x") # x ekseninin adı
plt.ylabel("y")

x2=np.arange(0,5,0.5)
plt.plot(x2,x2*2,label="2*x",color="green",linewidth=2,marker="o")
plt.legend()
plt.show()

x=["Ankara","İstanbul","İzmir"]
y=[120,178,87]

plt.bar(x,y)
plt.show()

x=["Ankara","İstanbul","İzmir"]
y=[120,178,87]

sticks=plt.bar(x,y)
sticks[1].set_hatch("/") #desen verir
sticks[0].set_hatch("*")
plt.show()
"""""

#K-Nearest Neighbours knn

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


# Veri yükleme ve ön işleme
def load_and_preprocess_data(data_dir, img_size=(224, 224), batch_size=32):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )

    # Veri artırma ve önbelleğe alma
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ])

    # Normalizasyon
    normalization_layer = layers.Rescaling(1. / 255)

    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds


# CNN modeli oluşturma
def create_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


# Elastic Net düzenlileştirmesi
def elastic_net_regularizer(l1=0.01, l2=0.01):
    return tf.keras.regularizers.L1L2(l1=l1, l2=l2)


# Dinamik öğrenme oranı
class DynamicLearningRate(tf.keras.callbacks.Callback):
    def __init__(self, initial_lr=0.001):
        super(DynamicLearningRate, self).__init__()
        self.initial_lr = initial_lr

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < 10:
            lr = self.initial_lr
        elif epoch < 20:
            lr = self.initial_lr / 2
        else:
            lr = self.initial_lr / 4
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        print(f"\nEpoch {epoch + 1}: Learning rate set to {lr}")


# Ana fonksiyon
def train_and_evaluate_model(data_dir, epochs=30, batch_size=32):
    # Veri yükleme
    train_ds, val_ds = load_and_preprocess_data(data_dir, batch_size=batch_size)

    # Model oluşturma
    input_shape = next(iter(train_ds))[0].shape[1:]
    num_classes = len(train_ds.class_names)
    model = create_cnn_model(input_shape, num_classes)

    # Elastic Net düzenlileştirmesi ekleme
    for layer in model.layers:
        if isinstance(layer, layers.Dense):
            layer.kernel_regularizer = elastic_net_regularizer()

    # Optimizer ve derleme
    optimizer = optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Dinamik öğrenme oranı
    lr_scheduler = DynamicLearningRate()

    # Eğitim
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=[lr_scheduler]
    )

    # Değerlendirme
    test_loss, test_acc = model.evaluate(val_ds)
    print(f"Test accuracy: {test_acc:.4f}")

    # Tahmin
    predictions = model.predict(val_ds)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.concatenate([y for x, y in val_ds], axis=0)

    # Sınıflandırma raporu
    class_names = train_ds.class_names
    print(classification_report(true_classes, predicted_classes, target_names=class_names))

    # Karmaşıklık matrisi
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# Kullanım
if __name__ == "__main__":
    import os

    # Kullanıcıdan veri seti yolunu al
    data_dir = input("Lütfen veri seti klasörünün tam yolunu girin: ")

    # Yolun var olup olmadığını kontrol et
    if not os.path.exists(data_dir):
        print(f"Hata: {data_dir} yolu bulunamadı. Lütfen doğru yolu girdiğinizden emin olun.")
    else:
        print(f"Veri seti yolu: {data_dir}")
        train_and_evaluate_model(data_dir, epochs=30, batch_size=32)