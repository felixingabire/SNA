from __future__ import division
import getdir
import random
import user
import d_1
import math
import d1
import dcalc
import numpy as np
from collections import defaultdict
from scipy.io import loadmat
from itertools import izip
import numpy as np
from sklearn.multiclass  import OneVsRestClassifier
from sklearn.linear_model  import LogisticRegression
from scipy.io  import loadmat
from numpy import linalg as LA
from sklearn.utils import shuffle as skshuffle
import time
inputVectors={}
d={}

tau=0
list1 = getdir.GetFileList('C:\Users\User\Desktop\code\dataset\digg\digg_30_1458_learn', []) #enter the name of folder
userls=[]  #create user list
dd=30 #25 dimesionuser parameter Z
randomvalz=[] # random valuez{}
Z={} #
T=4000
diff={}
timestamp={}
user_all=[]
user_diff=[]
di=dj=nbProbas=0
epsilon=0.0001
edges=[]
starttime = time.time()
def sigmoid(ui,uj):
	zu=Z[ui]
	wv=Z[uj]
	sum=0
	for i in range(1,5):
		sum+=(zu[i]*wv[i])
	sum=sum+zu[0]+wv[0]
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
		if(user_diff.index(u)<len(user_diff)-1):
		    edges.append([u,user_diff[user_diff.index(u)+1]])
	uin=timestamp[dname].keys()
	uin.sort()
	udiff=timestamp[dname][uin[0]]
	
	#print 'diffusion',d
   

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
	
	
	tau+=1
	print(tau)

dname=random.choice(diff.keys())
d=diff[dname]
print len(edges)
print u_num

#-----------------
for u in range(1,u_num+1): # user value init
	for i in range(dd):
		randomvalz.append(random.uniform(-1,1))
	inputVectors[str(u)]=randomvalz
	randomvalz=[]
class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            all_labels.append(labels)
        return all_labels


def sigmoid(x):
    """
    Computes sigmoid function

    @param x : x is a matrix
    """
    x = 1/(1 + np.exp(-x))
    return x


def normalize_rows(x):
    """
    Row normalization function

    normalizes each row of a matrix to have unit lenght
    @param x : x is a matrix
    """
    row_sums = np.sqrt(np.sum(x**2,axis=1,keepdims=True))
    x = x / row_sums
    return x


def negSampObjFuncAndGrad(predicted, target, outputVectors, K=2):
    """
    Negative sampling cost function for word2vec models

    computes cost and gradients for one predicted word vector and one
    target word vector using the negative sampling technique.

    Inputs:
        - predicted: numpy ndarray, predicted word vector (\har{r} in
                     the written component)
        - target   : integer, the index of the target word
        - outputVectors: "output" vectors for all tokens
        - K: it is sample size

    Outputs:
        - cost: cross entropy cost for the softmax word prediction
        - gradPred: the gradient with respect to the predicted word vector
        - grad: the gradient with restpect to all the other word vectors
    """
    gradPred = np.zeros_like(predicted)
    grad = np.zeros_like(outputVectors)
    cost = 0
    z = sigmoid(outputVectors[int(target)].dot(predicted))
    cost -= np.log(z)
    gradPred += outputVectors[int(target)] * (z-1.)
    grad[int(target)]+= predicted * (z-1.)

    for k in range(K):
        sampled_idx = random.randint(0, 4)
        z = sigmoid(outputVectors[sampled_idx].dot(predicted))
        cost -= np.log(1 - z)
        gradPred += z * outputVectors[sampled_idx]
        grad[sampled_idx] += z * predicted

    return cost, gradPred, grad


# def sgrad_desc(f, x0, step, iterations, postprocessing = None, useSaved = False, PRINT_EVERY=10):
#     """
#     Stochastic Gradient Descent

#     Inputs:
#        - f: the function to optimize, should take a single argument
#             and yield two outputs, a cost and the gradient with
#             respect to the arguments
#        - x0: the initial point to start SGD from
#        - step: the step size for SGD
#        - iterations: total iterations to run SGD for
#        - postprocessing: postprocessing function for the parameters
#             if necessary. In the case of word2vec we will need to
#             normalize the word vectors to have unit length.
#        - PRINT_EVERY: specifies every how many iterations to output
#     Output:
#        - x: the parameter value after SGD finishes
#     """
#     # Anneal learning rate every several iterations
#     ANNEAL_EVERY = 20000

#     if useSaved:
#         start_iter, oldx, state = load_saved_params()
#         if start_iter > 0:
#             x0 = oldx;
#             step *= 0.5 ** (start_iter / ANNEAL_EVERY)

#         if state:
#             random.setstate(state)
#     else:
#         start_iter = 0

#     x = x0

#     if not postprocessing:
#         postprocessing = lambda x: x

#     expcost = None

#     for iter in xrange(start_iter + 1, iterations + 1):
#         x = postprocessing(x)
#         cost, grad = f(x)
#         x -= step * grad

#         if iter % PRINT_EVERY == 0:
#             print "iteration=%d cost=%f" % (iter, cost)

#         if iter % SAVE_PARAMS_EVERY == 0 and useSaved:
#             save_params(iter, x)

#         if iter % ANNEAL_EVERY == 0:
#             step *= 0.5

#     return x


def sparse2graph(x):
    G = defaultdict(lambda: set())
    cx = x.tocoo()
    for i,j,v in izip(cx.row, cx.col, cx.data):
        G[i].add(j)
    return {str(k): [str(x) for x in v] for k,v in G.iteritems()}



# lookup tabel for vertices
vertex = {}
index = 0


# make edge matrix (dictionary)
edge = {}
for node in range(len(edges)):
	v_pair = (edges[node][0], edges[node][1])
	print v_pair
	edge[v_pair] = 1




# Generally converges in linear time with number of edges
EDGE_COUNT =len(edges)
V = len(user_all)
print V
d = 30
bsample = 5
eta = 0.01
outputVectors = np.random.normal(-1, 1, (V+1, d))
inputVectors = np.random.normal(-1, 1, (V+1, d))
a=0

for step in range(1,EDGE_COUNT):
    # sample mini batch edges
    # and updates the model parameter.
    # for loop

    # sample b edges
    
    sample_edges = []
    for i in range(bsample):
        edge1 = random.choice(edge.keys())
        
        sample_edges.append(edge1)

    tgradPred = np.zeros_like(inputVectors[0])
    tgradj = np.zeros_like(outputVectors[0])
    tgrad = np.zeros_like(outputVectors) # (K, d) dimension.
    

    for i, j in sample_edges:
        # normalize rows of inputs and output matrix
        inputVectors = normalize_rows(inputVectors)
        outputVectors = normalize_rows(outputVectors)

        cost, gradPred, grad = negSampObjFuncAndGrad(inputVectors[int(i)], j, outputVectors)
        tgradPred += (gradPred * 1)#w[i][j]
        tgradj += (grad[int(j)] * 1) #w[i][j]
        tgrad += (grad * 1)  #w[i][j]

    inputVectors[int(i)] += eta*tgradPred
    outputVectors[int(j)] += eta*tgradj
    outputVectors += eta*tgrad
   
print "==========feature learning done========"
print "writing features "



 


print "embeddings written to file"
fz=file('Z.txt','wb')
for i in range(len(user_all)):
    
    fz.write(str(inputVectors[i]))
fz.close()


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

x=inputVectors[1191][4]
y=inputVectors[1][4]


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
	for i in range(1,1455):
		kscore[i]=0
	if len(user_diff)==0:
		continue
		
    
   

	for v in user_all:
		for u in udiff:
			for i in range(dd):
				
					kscore[v]+=pow(inputVectors[v-1][i]-outputVectors[int(u)-1][i],2)
				
			       
		     
			
			
		    
				   
		kscore[v]=kscore[v]/len(diff)
	kscore_sort=sorted(kscore.iteritems(),key=lambda x:x[1],reverse=True)
	


	for u in user_diff:
		aaaa+=kscore_sort.index((int(u),kscore[int(u)]))/1454
	print dname, aaaa/len(diff[dname])
	
	diffusion_number=diffusion_number+1


	map=map+aaaa/len(diff[dname])
	aaaa=0
	kscore={}
	user_diff=[]
print 'map',map/diffusion_number
endtime = time.time()
usedtime = endtime-starttime
print usedtime,'finished!'
print v
