from __future__ import division
import getdir
import user
import random
import dcalc
import d_1
import d1
import fuv
import posu
import negu
import posv
import negv
import math
import reald

ii=0
list1 = getdir.GetFileList('C:\Users\User\Desktop\code\dataset\digg\digg_30_1458_learn', []) #enter the name of folder
userls=[]  #create user list
randomvalz=[] # random valuez
randomvalo=[] # random valueo
dd=25  #25 dimesion
freq=5000 # frequency
epsilon=0.0005 # learning step
Z={} # user parameter Z 
OMIG={} #user parameter O
diff={}  #diffusion dictionary
ppuv={}
user_all=[]
nbProbas=L1=L=L2=n=0 
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

zs_random=str(Z)

fz=file('C:\Users\User\Desktop\code\zs_random.txt','w')

fz.write(zs_random)

fz.close()


oldL=-99
it=0
for ee in list1:  #create the dic of diffusion
	f=open(ee)
	lines=f.readlines()
	seq1=lines[0].find('\n') #find the position of \n
	item=lines[0][2:seq1]
	diff[item]={}
	line=lines[-1].replace('D','\t').strip()  #preprocessing->'234,0\t236,0\t437,2'
	linels=line.split('\t')  #preprocessing again->['234,0', '236,0', '437,2']

	for l in linels:
		seq2=l.find(',')
		diff[item][l[0:seq2]]=l[seq2+1:]

	f.close()
for dif in diff:   #calculate the value of nbProbas
	timeseq=[]
	for u in diff[dif]:
		timeseq.append(int(diff[dif][u]))
	for u in diff[dif]:
		tD=int(diff[dif][u])
		nbProbas+=(u_num-dcalc.D_calc(diff,dif,u,tD,timeseq))

while True :
	dname=random.choice(diff.keys())
	d=diff[dname] #sample diffusion
	
	ddic=d_1.D_1(d,user_all) #D_(1)
	Dd=len(diff) 
	D_1_num=len(ddic)
	alpha=Dd*D_1_num/nbProbas
	
	
	v_name=random.choice(ddic.keys()) #sample node
	v_t=ddic[v_name]  #tD(v)
	
	ppuv[v_name]={}
	if v_t!='999':
		print'also rechead'
		DtD=d1.D1(v_t,d) #DtD(v)
		PvDd=1
		for u in DtD:
			Puv=fuv.puvcalc(Z,Z,u,v_name)
			ppuv[v_name][u]=Puv
			PvDd=PvDd*(1-Puv)
		PvDd=1-PvDd
	else:
		DtD=d
	for u in DtD:
		print 'rechead'
		Xiupos=posu.Posu(Z,Z,u,v_name)
		Xiuneg=negu.Negu(Z,Z,u,v_name)
		
		if v_t!='999':
			for i in range(dd):
				Z[u][i]=Z[u][i]+alpha*epsilon*((ppuv[v_name][u]/PvDd)*Xiupos[i]+(1-ppuv[v_name][u]/PvDd)*Xiuneg[i])
				
		else:
			for i in range(dd):
				Z[u][i]=Z[u][i]+alpha*epsilon*Xiuneg[i]
	it+=1
	if (it % freq)==0:
		print 'written to file'
		zs=str(Z)
		
		fz=file('C:\Users\User\Desktop\code\Z.txt','w')
			
		fz.write(zs)
			
		fz.close()
		break
			
		for dname in diff:
			dre=reald.realD(diff[dname],userls)
			print it
			for u in dre:
				if dre[u]!='999':
					dreu=d1.D1(dre[u],diff[dname])
					if len(dreu)==0:
						continue
					else:
						p=1
						for uu in dreu:
							p_uv=fuv.puvcalc(Z,Z,uu,u)
							p=p*(1-p_uv)
						p=1-p
					L1+=math.log(p)
					print 'l1',L1
				else:
					p=1
					for uu in diff[dname]:
						p_uv=fuv.puvcalc(Z,Z,uu,u)
						p=p*(1-p_uv)
					L2+=math.log(p)
					print 'l2', L2
		L=L1+L2
		print L
		if L<=oldL:
			print oldL
			
			zs=str(Z)
			ws=str(OMIG)
			fz=file('C:\Users\User\Desktop\code\zs.txt','w')
			
			fz.write(zs)
			
			fz.close()
			
			
			break
		oldL=L
		L=L1=L2=0
