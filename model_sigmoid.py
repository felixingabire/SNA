from __future__ import division
import getdir
import random
import user
import d_1
import math
import d1
import dcalc

tau=0
list1 = getdir.GetFileList('C:\Users\User\Desktop\code\dataset\digg\digg_30_1458_learn', []) #enter the name of folder
userls=[]  #create user list
dd=100  #25 dimesionuser parameter Z
randomvalz=[] # random valuez{}
Z={} #
T=1000
diff={}
timestamp={}
user_all=[]
user_diff=[]
di=dj=nbProbas=0
epsilon=0.0001
count2=1


def sigmoid(ui,uj):
	zu=Z[ui]
	wv=Z[uj]
	sum=0
	for i in range(1,5):
		sum+=(zu[i]*wv[i])
	sum=sum+(zu[0]*wv[0])
	sum=1+math.exp(sum)
	return (1/sum)


for ee in list1:  #create the dic of diffusion
	f=open(ee)
	lines=f.readlines()
	seq1=lines[0].find('\n') #find the position of \n
	item=lines[0][2:seq1]
	diff[item]={}
	timestamp[item]={}
	line=lines[-1].replace('D','\t').strip()  #preprocessing->'234,0\t236,0\t437,2'
	linels=line.split('\t')  #preprocessing again->['234,0', '236,0', '437,2']
	for l in linels:
		seq2=l.find(',')
		diff[item][l[0:seq2]]=l[seq2+1:]
		timestamp[item][int(l[seq2+1:])]=[]
	for l in linels:
		seq2=l.find(',')
		timestamp[item][int(l[seq2+1:])].append(l[0:seq2])

	f.close()

for e in list1:  #create user list
	user.users(e,userls)
userls=list(set(userls))
userls.sort()

u_num=userls[-1]
for i in range(1,u_num+1):
	user_all.append(i)

for u in range(1,u_num+1): # user value init
	for i in range(dd):
		randomvalz.append(random.uniform(-1,1))
	Z[str(u)]=randomvalz
	randomvalz=[]

for dif in diff:   #calculate the value of nbProbas
	timeseq=[]
	for u in diff[dif]:
		timeseq.append(int(diff[dif][u]))
	for u in diff[dif]:
		tD=int(diff[dif][u])
		nbProbas+=(u_num-dcalc.D_calc(diff,dif,u,tD,timeseq))


while tau<T:
	dname=random.choice(diff.keys())
	d=diff[dname] #sample diffusion
	for u in diff[dname].keys():
		user_diff.append(u)
	uin=timestamp[dname].keys()
	uin.sort()
	udiff=timestamp[dname][uin[0]]
#	print 'diffusion',d

	sc=random.choice(udiff) #sample sc
	sc_t=diff[dname][sc]
	ddic=d_1.D_1(d,user_all) #D_(1)
	dsc=d_1.D_1(d,user_diff)
	Dd=len(diff) 
	D_1_num=len(ddic)
	alpha=Dd*D_1_num/nbProbas
	if len(dsc)==0:
		continue
	ui=random.choice(dsc.keys())  #sample ui
	uj=random.choice(ddic.keys())  #sample uj
	dsi=sigmoid(ui,sc)
	dsj=sigmoid(uj,sc)
	dij=sigmoid(ui,uj)
	
	
	deltad=dsj-dsi
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
