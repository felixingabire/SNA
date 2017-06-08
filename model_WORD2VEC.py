from __future__ import division
import getdir
import random
import user
import d_1
import math
import d1
import dcalc
import numpy as np

tau=0
list1 = getdir.GetFileList('C:\Users\User\Desktop\code\dataset\digg\digg_30_1458_learn', []) #enter the name of folder
userls=[]  #create user list
dd=10
randomvalz=[] # random valuez{}
Z={} #
T=1000
diff={}
timestamp={}
user_all=[]
user_diff=[]
di=dj=nbProbas=0
epsilon=0.0001
words=[]



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
		if(user_diff.index(u)<len(user_diff)-1):
		    words.append(u)
	uin=timestamp[dname].keys()
	uin.sort()
	udiff=timestamp[dname][uin[0]]
	
	#print 'diffusion',d
	print tau,'ok'
	tau=tau+1
   

	
	
	

	

#-----------------------------------




def cos_similarity(x1, x2):
    inner_product = np.dot(x1, x2)
    x1_l2 = np.linalg.norm(x1, 2)
    x2_l2 = np.linalg.norm(x2, 2)
    return inner_product / (x1_l2 * x2_l2)

if __name__ == '__main__':
    
    dic = dict()
    min_reduce = 1  # preserving all words doesn't improve results
    for word in words:
        if word in dic.keys():
            dic[word] += 1
        else:
            dic[word] = 1
    for key in dic.keys():
        if dic[key] < min_reduce:
            dic.pop(key)
    vocab_size = len(dic.keys())
    i = 0
    vocab = []
    for key in dic.keys():
        dic[key] = i
        vocab.append(key)
        i += 1

    window = 5
    negative = 5
    alpha = 0.025
    total_iter = 5
    cur_iter = 0
    losses = [float(0)] * total_iter
    cnt_f = 0
    sentence_position = 0
    word_count = len(words)
    print(word_count)
    print(words)
    layer1_size = 10
    syn0 = (np.random.rand(1454, layer1_size) - 0.5) / layer1_size  # centralize
    syn1neg = np.zeros((1454, layer1_size), dtype=float)

    EXP_TABLE_SIZE = 100
    MAX_EXP = 6
    expTable = [0.0] * EXP_TABLE_SIZE  # accelerate obviously!
    for i in range(EXP_TABLE_SIZE):
        expTable[i] = math.exp((i / float(EXP_TABLE_SIZE) * 2 - 1) * MAX_EXP)
        expTable[i] /= (expTable[i] + 1)

    while True:
        if sentence_position % 100 == 0:
            print sentence_position
        if sentence_position >= word_count:
		
            losses[cur_iter] /= cnt_f
            cnt_f = 0
            cur_iter += 1
            sentence_position = 0
            print "-------------------"
            if cur_iter >= total_iter:
                break
        word = words[sentence_position]
        if word not in dic.keys():
            sentence_position += 1
            continue
        cur_vocab_index = dic[word]
        neu1 = np.zeros((1, layer1_size))
        neu1e = np.zeros((1, layer1_size))

        cw = 0
        for a in range(0, window * 2 + 1):
            if a != window:
                c = sentence_position - window + a
                if c < 0 or c >= word_count:
                    continue
                last_word = words[c]
                if last_word in dic.keys():
                    last_word_index = dic[last_word]
                else:
                    continue
                neu1 += syn0[last_word_index, :]
                cw += 1
        if cw:
            neu1 /= cw
            # out -> hidden
            for d in range(negative + 1):
                if d == 0:
                    target = cur_vocab_index
                    label = 1
                else:
                    target = random.randint(0, vocab_size - 1)
                    if target == cur_vocab_index:
                        continue
                    label = 0
                f = np.dot(neu1, syn1neg[target, :])
                if f > MAX_EXP:
                    sigmoid_f = 1
                    g = (label - 1) * alpha
                elif f < MAX_EXP:
                    sigmoid_f = 0
                    g = (label - 0) * alpha
                else:
                    sigmoid_f = expTable[int((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                    g = (label - sigmoid_f) * alpha
                if d == 0:
                    if sigmoid_f:
                        losses[cur_iter] += math.log(sigmoid_f)
                    else:
                        losses[cur_iter] += -6
                else:
                    if sigmoid_f != 1:
                        losses[cur_iter] += math.log(1 - sigmoid_f)
                    else:
                        losses[cur_iter] += -6
                cnt_f += 1
                neu1e += g * syn1neg[target, :]
                syn1neg[target, :] = syn1neg[target, :] + g * neu1

            # hidden -> in
            for a in range(0, window * 2 + 1):
                if a != window:
                    c = sentence_position - window + a
                    if c < 0 or c >= word_count:
                        continue
                    last_word = words[c]
                    if last_word in dic.keys():
                        last_word_index = dic[last_word]
                    else:
                        continue
                    syn0[last_word_index, :] = syn0[last_word_index, :] + neu1e

        sentence_position += 1

    queries = ['501', '37', '', '297', '', '1124', '826']

    for query in queries:
        print query
        print '----------'
        if query in dic.keys():
            query_index = dic[query]
            print 'ok'
        else:
			
            continue
            
        query_vec = syn0[query_index, :]
        similarity_result = np.zeros(vocab_size)
        for i in range(vocab_size):
            similarity_result[i] = cos_similarity(syn0[i], query_vec)
        sorted_indices = np.argsort(similarity_result)
        for i in range(vocab_size - 2, vocab_size - 12, -1):
            print vocab[sorted_indices[i]]
        print '----------'
       
    print losses
print syn0[1]   

# (K, d) dimension.-----------------------------------------testing


diff={}
timestamp={}
Z={}
user_all=[]
user_diff=[]
user_diff2=[]
kscore={}  
userls=[]
aaaa=0
diffusion_number=1
d=10

map=num=0

list1 = getdir.GetFileList('C:\Users\User\Desktop\code\dataset\digg\digg_30_1458_test_1', []) #enter the name of folder
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
#dname=random.choice(diff.keys())
	uin=timestamp[dname].keys()
	uin.sort()
	udiff=timestamp[dname][uin[0]]
	for u in diff[dname].keys():
		if u not in udiff:
			user_diff.append(int(u))
	for i in range(len(user_diff)):
		 user_diff2.append(i)		
	for i in range(0,1454):
		kscore[i]=0
	
	for v in user_all:
		for u in udiff:
			for i in range(dd):
				
				kscore[v]+=pow(syn0[v][i]-syn0[int(u)][i],2)
		kscore[v]=kscore[v]/len(diff)
		print kscore[v]
	kscore_sort=sorted(kscore.iteritems(),key=lambda x:x[1],reverse=True)
	print '----------------'


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

	
