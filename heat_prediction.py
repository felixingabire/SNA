from __future__ import division
import getdir
import user
import d_1
import random

diff={}
timestamp={}
Z={}
user_all=[]
user_diff=[]
kscore={}  
userls=[]
aaaa=0
dd=100
map=num=0
diffusion_number=1

list1 = getdir.GetFileList('C:\Users\User\Desktop\code\dataset\digg\digg_30_1458_test_2', []) #enter the name of folder
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
	user.users(ee,userls)
userls=set(userls)
fz=open('C:\Users\User\Desktop\code\Z.txt')
Z=fz.read()
Z=eval(Z)


for i in range(1,1454):
	user_all.append(i)
	
for dname in diff:
#dname=random.choice(diff.keys())
	uin=timestamp[dname].keys()
	uin.sort()
	udiff=timestamp[dname][uin[0]]
	for u in diff[dname].keys():
		if u not in udiff:
			user_diff.append(int(u))
	for i in range(1,1455):
		kscore[i]=0
	if len(user_diff)==0:
		continue
    #print 'diffusion',diff[dname]
#print 'diff000',udiff


	for v in user_all:
		for u in udiff:
			for i in range(dd):
				kscore[v]+=pow(Z[str(v)][i]-Z[u][i],2)
		kscore[v]=kscore[v]/len(diff)
	kscore_sort=sorted(kscore.iteritems(),key=lambda x:x[1],reverse=True)
  


	for u in user_diff:
		aaaa+=kscore_sort.index((int(u),kscore[int(u)]))/1454
	print aaaa/len(diff[dname])
	print diffusion_number
	diffusion_number=diffusion_number+1



	map=map+aaaa/len(diff[dname])
	aaaa=0
	kscore={}
	user_diff=[]
print 'map',map/diffusion_number



