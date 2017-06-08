from __future__ import division
import getdir
import user
import d_2
import random
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity   
from numpy import *

from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import PCA
from sklearn import preprocessing

from nltk.stem.snowball import SnowballStemmer
import re

import nltk
import pprint

import os
tau=0
diff = {}
timestamp={}
Z = {}
user_all = []
user_diff = []
kscore = {}
userls = []
aaaa = 0
dd = 30
map = num = 0
filenumber=0;
linenumber=0;
listofNodes = []
node=list()
listofEdges=[]
edge=[]
node_id=1
listofWeb=list()
timestamp={}

diffusion=0
start=0
end=0
diffusion=[]
count=-1
di=dj=nbProbas=0
epsilon=0.0001
T=1000


corpus = []

G = nx.DiGraph()
def sigmoid(ui,uj):
	zu=Z[ui]
	wv=Z[uj]
	sum=0
	for i in range(1,dd):
		sum+=(zu[i]*wv[i])
	#sum=sum+(zu[0]*wv[0])
	
	sum=1+math.exp(sum)
	return (1/sum)


list1 = getdir.GetFileList('C:\Users\User\Desktop\code\dataset\Test', [])  # enter the name of folder
for ee in list1:  # create the dic of diffusion
    diffusionnumber = 0
    f = open(ee)
    lines = f.readlines()
    seq1 = lines[0].find('\n')  # find the position of \n
    count = filenumber
    diff[filenumber]={}
    i=0;
    timestamp[filenumber] = {}
    user_1 = []
    file_split = re.split(r'\s+',lines[0])
    file_split = [x.lower() for x in file_split]
    porter = nltk.PorterStemmer()  
    file_split = [porter.stem(x) for x in file_split]

    wnl = nltk.WordNetLemmatizer()  
    file_split = [wnl.lemmatize(x) for x in file_split]

    file_split = ' '.join(file_split)

    corpus.append(file_split)



    diffusion.append([filenumber,lines])
    print(ee)

    i=0


    



    # timestamp[item] = {}
    line = lines[1].strip('')  # preprocessing->'234,0\t236,0\t437,2'
    linels = line.split(',')  # preprocessing again->['234,0', '236,0', '437,2']



    node.append(line)
    for i in listofNodes:
        if i not in listofNodes:
            node.append(i)


    #  print(linels)



    for l in linels:
        seq2 = l.find(',')
        url=linels[0][0:seq2]


        linenumber=linenumber+1;





        url=l[14:]

        l.split(',')
        for i in l.split():


              timestamptime=l.split(None,1)[0]
              web=l.split(None,0)[0]

              web=l[14:]
              #print(web)

              if [web] not in listofWeb:
                  listofNodes.append([node_id,web])

                  listofWeb.append(web)

                  user_all.append(listofWeb.index(web))
                  userl_all= list(set(user_all))




                  G.add_edge(node_id,node_id+1)
                  edge.append([node_id,node_id+1])
                  node_id=node_id+1
                  if(web in listofWeb):



                          G.add_edge(listofWeb.index(web),node_id+1)

                          edge.append([listofWeb.index(web),node_id+1])


        timestamp[filenumber][int(listofWeb.index(web))] = []
        for l in linels:
            #diff[filenumber][i].append(listofWeb.index(web))

            timestamp[filenumber][int(listofWeb.index(web))].append(timestamptime)
       # timestamp.append([(listofWeb.index(web)), timestamptime])

        user_1.append(listofWeb.index(web))






        diff[filenumber] =user_1

        l.strip()

        start= l[:14]
        start.strip()
        end = l[len(start):]

        end.strip()





        # print(l[14:])
        # print(l[:14])



        # diff[item][l[0:seq2]] = l[seq1 ][seq1:seq2]
        # print(diff[item][l[0:seq2+1]] )

        #timestamp[item][int(l[seq2 + 1:])] = []
    for l in linels:
        seq2 = l.find(',')
        #timestamp[item][int(l[seq2 + 1:])].append(l[0:seq2])
        #timestamp[item][int(l[seq2 + 1:])]=l[seq2+2:]


    filenumber = filenumber + 1


    f.close()

vectorizer=CountVectorizer()
transformer=TfidfTransformer() 
tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))  
word=vectorizer.get_feature_names() 
weight=tfidf.toarray()  

min_max_scaler = preprocessing.MinMaxScaler()
weight = min_max_scaler.fit_transform(weight)


pca = PCA(n_components=dd,whiten=True)
newData = pca.fit_transform(weight)

min_max_scaler = preprocessing.MinMaxScaler()
newData = min_max_scaler.fit_transform(newData)







    #nx.draw(G)

   
#user_all=listofWeb
#print(user_all)


#print(lisofweb)
#print(filenumber)
#print(linenumber)
randomvalz=[]
for u in range(0,len(listofWeb)+1): # user value init
	for i in range(dd):
		randomvalz.append(random.uniform(-1,1))
	Z[u]=randomvalz
        randomvalz=[]


alpha=0.01

while tau<T:
	dname=random.choice(diff.keys())
         
	d=diff[dname] #sample diffusion

	for u in diff[dname]:
		user_diff.append(u)
	uin=timestamp[dname].keys()
	uin.sort()
	udiff=timestamp[dname][uin[0]]
	#print 'diffusion',d

	#sc=random.choice(udiff) #sample s
	sc=diff[dname][0]
   
	ddic=d_2.D_1(d,user_all) #D_(1)
	dsc=d_2.D_1(d,user_diff)
	Dd=len(diff)
	D_1_num=len(ddic)
	print newData[dname]
	try:
		Z[sc]=Z[sc]+newData[dname]
	except:
		pass
	
	#alpha=Dd*D_1_num/nbProbas
	if len(dsc)==0:
		continue
	ui=random.choice(user_diff)  #sample ui
	uj=random.choice(user_all)  #sample uj
	try:
		dsi=sigmoid(ui,sc)
		dsj=sigmoid(uj,sc)
		dij=sigmoid(ui,uj)
	except:
			pass

	for i in range(dd):
		di+=pow(Z[ui][i]-Z[sc][i],2)
	for i in range(dd):
		dj+=pow(Z[uj][i]-Z[sc][i],2)
	deltad=dj-di

	if deltad<0:
		for i in range(dd):
			 Z[ui][i]=Z[ui][i]+alpha*epsilon*dsi*(1-dsi)
			 Z[uj][i]=Z[uj][i]+alpha*epsilon*dsj*(1-dsj)
			 Z[sc][i]=Z[sc][i]+alpha*epsilon*dij*(1-dij)
		    
	tau+=1
	print tau
	
	

zs=str(Z)
fz=file('C:\Users\User\Desktop\code\Z.txt','w')
fz.write(zs)
fz.close()

