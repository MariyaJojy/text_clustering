from gensim.parsing.preprocessing import remove_stopwords
import os
import re
import pandas as pd
import numpy as np
from pandas import DataFrame

#extracting and cleaning text data
data=[]
list=os.listdir('Level2')
for file in list:
    if "sents_dict" not in file:
        path=os.path.join("C:/Users/Jojy/tcs2/Level2/"+file)
        f = open(path, "r", encoding="utf8")
        for x in f:
            x = x.lower()
            x = re.sub(r"[',]", ' ', x)
            x = re.sub(r'["]', '', x)
            x = x.replace("[ ", "")
            x = x.replace(" ]", "")
            x = x.replace("_ _", "")
            x = x.replace("_", "")
            x = x.replace("-", "")
            x = x.replace("\n", "")
            a = x.split("    ")
            if remove_stopwords((a[0].replace("[", "")).replace(",", "")) not in data:
                a[0] = a[0].replace("[", "")
                a[0] = remove_stopwords(a[0].replace(",", ""))
                data.append(a[0])
            l = len(x.split("    "))
            if l == 3:
                if remove_stopwords((a[2].replace("[", "")).replace(",", "")) not in data:
                    a[2] = a[2].replace("[", "")
                    a[2] = remove_stopwords(a[2].replace(",", ""))
                    data.append(a[2])
        f.close()

#extracting wikidatainfo
f = open("wiki.txt", "r", encoding="utf8")

dat=[]
res={}
desc=[]
w=f.read()
w=w.replace("{","")
w=w.replace("'","")
w=w.replace("[","")
w=w.split("], ")
for i in w:
    i=i.lower()
    dat.append(i.split(": ")[0])
    desc.append(i.split(": ")[1])
    res[i.split(": ")[0]]=i.split(": ")[1]
d = {'Name':dat,'Desc':desc}
df=DataFrame(d)
f.close()

#using decriptions wherever it is available and leaving the word as it is if description not available
sentences=[]
for i in data:
    sentences.append(i.split(" "))
v=[]
for i in sentences:
    cnt=0
    s=""
    for word in i:


        if word in dat:

            s=s+res[word]
    s=s.replace(",","")
    s=remove_stopwords(s)
    if s=="":
        for word in i:
            s=s+" "+word
            s = remove_stopwords(s)
        v.append(s)
    else:
        v.append(s)

#function to convert word/phrases to average vector form
def sent_vector(sent, model):
    sent_vec = []
    num = 0
    for w in sent:
        try:
            if num == 0:
                sent_vec = model[w]
            else:
                sent_vec = np.add(sent_vec, model[w])
            num += 1
        except:
            pass

    return np.asarray(sent_vec) / num

#normalization and converting to average vector form
from gensim.models import Word2Vec
from sklearn import cluster
model = Word2Vec(sentences, min_count=1)
X = []
for sentence in sentences:
    X.append(sent_vector(sentence, model))
a=np.asarray(X)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
b = scaler.fit_transform(a)

print("===========AGGLOMERATIVE=============")
#plot for dendrogram
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))
plt.title("Dendrograms")
dend = shc.dendrogram(shc.linkage(b, method='ward'))
plt.show()

#model training
from sklearn.cluster import AgglomerativeClustering
agglo = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward')
clusters=agglo.fit_predict(b)

#dataframe for clusters
d = { 'data': data, 'description':v, 'cluster': clusters }
frame = pd.DataFrame(d, columns=['data', 'description','cluster'] )
m=[]
for i in set(clusters):
    print("cluster ", i)
    m.append(len(frame[(frame["cluster"] == i)]))
    print(frame[(frame["cluster"]==i)])
frame.sort_values(by=['cluster'], inplace=True)

#converting to csv
frame.to_csv('clust_agglo_6.csv')

#no. of samples in each cluster
print(m)