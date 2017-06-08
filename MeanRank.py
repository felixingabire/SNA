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
dd=5
map=num=0
kscore2={}
for i in range(1,1454):
		kscore[i]=0	
list1 = getdir.GetFileList('C:\Users\Administrator\Desktop\Felix doc\Learn\code\dataset\digg\Test_digg', []) #enter the name of folder
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


for i in range(1,1454):
	user_all.append(i)

	
for dname in diff:
	i=1
#dname=random.choice(diff.keys())
	uin=timestamp[dname].keys()
	uin.sort()
	udiff=timestamp[dname][uin[0]]
	print uin
	print udiff
	for u in diff[dname].keys():
	
			    if i <len(uin):
				  kscore[u]=kscore[int(u)]+1/uin[i]
				  print 'u, i,uin',kscore[int(u)],len(uin)
				  print u,i,uin[i]
				  user_diff.append(int(u))
				  i=i+1
				  

	
	



print 'kscore23'


zs=str(kscore)
fz=file('C:\Users\Administrator\Desktop\Felix doc\Learn\code\kscore2.txt','w')
fz.write(zs)
fz.close()
#print 'diffusion',diff[dname]
#	print 'diff000',udiff
	
