import getdir
import dcalc
import random
import d_1
import fuv
import d1
import posu
import negu
import posv
import negv
import math
import reald
kscore={}

list1 = getdir.GetFileList('C:/Users/User/Desktop/code/dataset/irvine/learn', []) #enter the name of folder
diff={}
tau=0
T=3000
timestamp={}
user_all=[]
user_diff=[]
userls=[]  #create user list
randomvalz=[] # random valuez
randomvalo=[] # random valueo
dd=50  #25 dimesion
freq=100 # frequency
epsilon=0.0001 # learning step
Z={} # user parameter Z 
OMIG={} #user parameter O
diff={}  #diffusion dictionary
ppuv={}
nbProbas=0 
L1=L=L2=n=nn=0
for l in list1: #create the dic of diffusion
	nn+=1
	f=open(l)
	dif=eval(f.read())
	diff[nn]=dif

	    

for i in range(1,900): #create the user list
	userls.append(str(i))
for u in range(1,900): # user value init
	for i in range(dd):
		randomvalz.append(random.uniform(-1,1))
	for i in range(dd):
		randomvalo.append(random.uniform(-1,1))
	Z[str(u)]=randomvalz
	OMIG[str(u)]=randomvalo
	randomvalo=[]
	randomvalz=[]
oldL=-9999999999999
it=0
u_num=900
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
#---------------------
for i in range(1,900):
	kscore[i]=0
for dname in diff:
	
	d=diff[dname] #sample diffusion
	
	for u in diff[dname].keys():
		user_diff.append(u)
	
	uin=timestamp[dname].keys()
	uin.sort()
	udiff=timestamp[dname][uin[0]]
	
	sc=random.choice(udiff)
	for u in diff[dname].keys():
		if sc==int(diff[dname][u]):
		       sc=u
	
	

   

	sc=u #sample sc
	#sc_t=diff[dname][sc]
	ddic=d_1.D_1(d,user_all) #D_(1)
	dsc=d_1.D_1(d,user_diff)
	Dd=len(diff) 
	D_1_num=len(ddic)
	alpha=Dd*D_1_num/nbProbas
	if len(dsc)==0:
		continue
	if len(ddic)==0:
		continue
	for u in diff[dname]:
	  if u in diff[dname]:
		  kscore[int(u)]=(kscore[int(u)]+1)
		  
	kscore[int(u)]=(kscore[int(u)])/len(diff)
		
print kscore,len(diff)	

	
zs=str(kscore)
fz=file('C:\Users\User\Desktop\code\kscore.txt','w')
fz.write(zs)
fz.close()
