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
dd = 100
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


list1 = getdir.GetFileList('C:\Users\User\Desktop\code\dataset\meme_test', [])  # enter the name of folder
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
#user.users(ee,userls)
userls=set(userls)
fz=open('C:\Users\User\Desktop\code\Z.txt')
Z=fz.read()
Z=eval(Z)


for i in range(1,7851):
	user_all.append(i)
	
for dname in diff:
#dname=random.choice(diff.keys())
    
	uin=timestamp[dname].keys()
	uin.sort()
	
	udiff=timestamp[dname][uin[0]]
	for u in diff[dname]:
		if u not in udiff:
			user_diff.append(int(u))
	for i in range(1,7851):
		kscore[i]=0
	if len(user_diff)==0:
		continue
#print 'diffusion',diff[dname]
	

	for v in user_all:
		
		for u in udiff:
			kscore[v]=0
			for i in range(dd):
				
				#kscore[v]+=pow(Z[str(v)][i]-Z[1+diff[dname][0]][i],2)
				try:
					kscore[v]+=pow(Z[v][i]-Z[diff[dname][0]][i],2)
				except:
					pass
		kscore[v]=kscore[v]/len(diff)
	kscore_sort=sorted(kscore.iteritems(),key=lambda x:x[1],reverse=True)



	for u in user_diff:
		aaaa+=kscore_sort.index((int(u),kscore[int(u)]))/7851
	print aaaa/len(diff[dname])


	map=map+aaaa/len(diff[dname])
	aaaa=0
	kscore={}
	user_diff=[]
print 'map',map/15
print len(Z)
