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

list1 = getdir.GetFileList('C:/Users/User/Desktop/code/dataset/irvine/irvine_clean', []) #enter the name of folder
diff={}
userls=[]  #create user list
randomvalz=[] # random valuez
randomvalo=[] # random valueo
dd=25  #25 dimesion
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
	for u in diff[dif]:
		timeseq.append(int(diff[dif][u]))
	for u in diff[dif]:
		tD=int(diff[dif][u])
		nbProbas+=(u_num-dcalc.D_calc(diff,dif,u,tD,timeseq))
while True :
	dname=random.choice(diff.keys())
	d=diff[dname] #sample diffusion
	ddic=d_1.D_1(d,userls) #D_(1)
	Dd=len(diff) 
	D_1_num=len(ddic)
	alpha=Dd*D_1_num/nbProbas
	v_name=random.choice(ddic.keys()) #sample node
	v_t=ddic[v_name]  #tD(v)
	print 'v_name'
	print v_name
	print v_t
	ppuv[v_name]={}
	if v_t!='999':
		DtD=d1.D1(v_t,d) #DtD(v)
		PvDd=1
		for u in DtD:
			Puv=fuv.puvcalc(Z,OMIG,u,v_name)
			ppuv[v_name][u]=Puv
			PvDd=PvDd*(1-Puv)
		PvDd=1-PvDd
	else:
		DtD=d
	for u in DtD:
		Xiupos=posu.Posu(Z,OMIG,u,v_name)
		Xiuneg=negu.Negu(Z,OMIG,u,v_name)
		Xivpos=posv.Posv(Z,OMIG,u,v_name)
		Xivneg=negv.Negv(Z,OMIG,u,v_name)

		if v_t!='999':

			for i in range(dd):
				Z[u][i]=Z[u][i]+alpha*epsilon*((ppuv[v_name][u]/PvDd)*Xiupos[i]+(1-ppuv[v_name][u]/PvDd)*Xiuneg[i])
				OMIG[u][i]=OMIG[u][i]+alpha*epsilon*((ppuv[v_name][u]/PvDd)*Xivpos[i]+(1-ppuv[v_name][u]/PvDd)*Xivneg[i])
		else:
			for i in range(dd):
				Z[u][i]=Z[u][i]+alpha*epsilon*Xiuneg[i]
				OMIG[u][i]=OMIG[u][i]+alpha*epsilon*Xivneg[i]
	it+=1
	if (it % freq)==0:
		for dname in diff:
			dre=reald.realD(diff[dname],userls)
			for u in dre:
				if dre[u]!='999':
					dreu=d1.D1(dre[u],diff[dname])
					if len(dreu)==0:
						continue
					else:
						p=1
						for uu in dreu:
							p_uv=fuv.puvcalc(Z,OMIG,uu,u)
							p=p*(1-p_uv)
						p=1-p
					L1+=math.log(p)
				else:
					p=1
					for uu in diff[dname]:
						p_uv=fuv.puvcalc(Z,OMIG,uu,u)
						p=p*(1-p_uv)
					L2+=math.log(p)
		L=L1+L2
		if L<=oldL:
			print oldL/len(diff)
			zs=str(Z)
			ws=str(OMIG)
			fz=file('C:/Users/User/Desktop/code/dataset/irvine/irvine_zs.txt','w')
			fw=file('C:/Users/User/Desktop/code/dataset/irvine/irvine_ws.txt','w')
			fz.write(zs)
			fw.write(ws)
			fz.close()
			fw.close()
			break
		oldL=L
		L=L1=L2=0


