from __future__ import division
import getdir
import user
import d_1
import random


import dcalc

import d_1
import fuv
import d1
import posu
import negu
import posv
import negv
import math
import reald


diff={}
timestamp={}
Z={}
user_all=[]
user_diff=[]
kscore={}  
userls=[]
aaaa=0
dd=50
map=num=0
diffusion_number=1
#-----------------------
mse=0
map=0

p=r=F1=0
userls=[]
puv={}
diff={}
nbProbas=0 
L1=L=L2=n=nn=0
nn=0
u_num=900
list1=getdir.GetFileList('C:/Users/User/Desktop/code/dataset/irvine/irvine_test', []) #enter the name of folder
print len(list1)

for l in list1:
	nn+=1
	f=open(l)
	dif=eval(f.read())
	diff[nn]=dif
fz=open('C:\Users\User\Desktop\code\Z.txt')

Z=fz.read()
Z=eval(Z)

#----------------------
for dif in diff:   #calculate the value of nbProbas
	timeseq=[]
	timestamp[dif]={}
	for u in diff[dif]:
		timestamp[dif][int(u)]=[]
	for u in diff[dif]:
		timeseq.append(int(diff[dif][u]))
		timestamp[dif][int(u)].append(int(diff[dif][u]))
	for u in diff[dif]:
		tD=int(diff[dif][u])
		nbProbas+=(u_num-dcalc.D_calc(diff,dif,u,tD,timeseq))



for i in range(1,900):
	user_all.append(i)
	
for dname in diff:
#dname=random.choice(diff.keys())
	uin=timestamp[dname].keys()
	uin.sort()
	udiff=timestamp[dname][uin[0]]
	sdiff=[]
	sc=random.choice(udiff)
	for u in diff[dname].keys():
		if sc==int(diff[dname][u]):
		       sdiff.append(u)
	
	

   

	sc=u #sample sc
	for u in diff[dname].keys():
		if u not in sdiff:
			user_diff.append(int(u))
	for i in range(1,900):
		kscore[i]=0
	if len(user_diff)==0:
		continue
    #print 'diffusion',diff[dname]
#print 'diff000',udiff
    

	for v in user_all:
		for u in sdiff:
			for i in range(dd):
				kscore[v]+=pow(Z[str(v)][i]-Z[u][i],2)
		kscore[v]=kscore[v]/len(diff)
		
	kscore_sort=sorted(kscore.iteritems(),key=lambda x:x[1],reverse=True)
	


	for u in user_diff:
		aaaa+=kscore_sort.index((int(u),kscore[int(u)]))/900
	print aaaa/len(diff[dname])
	
	diffusion_number=diffusion_number+1
	


	map=map+aaaa/len(diff[dname])
	
	aaaa=0
	kscore={}
	user_diff=[]
print 'map',map/26

