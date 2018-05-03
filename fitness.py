import pandas as pd
import numpy as np
import math
import constants
import readdata
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from readdata import  get_file_contents

def get_final_fitness_trig(fitness):
	got_back = get_final_fitness(fitness)
	if(readdata.delta_x.get(constants.URL)!=None):
		got_back = got_back + readdata.delta_x.get(constants.URL)
	return got_back
	
def get_final_fitness(fitness): #final one valeu fitness of a candidate
	final_fitness = np.zeros(len(fitness))
	for i,x in zip(range(0,final_fitness.shape[0]),fitness):
		final_fitness[i]=np.sum(np.array(fitness[i])) # should be changed after each function addition
	return final_fitness

def get_pop_distri(x_dataset, population, K):
	X=np.array(x_dataset)
	pop=np.array(population)
	pop_distri = []
	for i in pop:
		distri = []
		for j in range(0,K):
			distri.append([])
		for pp in X:
			distance=-1
			ind=-1
			lol=0
			for centroid in i:
				dist = centroid - pp 
				dist=dist*dist
				dist=np.sum(dist)
				if distance < 0:
					distance = dist
					ind = lol
				elif distance > dist: 
					distance = dist 
					ind = lol
				lol = lol + 1
			distri[ind].append(np.array(pp))
		for j in range(0,K):
			distri[j]=np.array(distri[j])
		pop_distri.append(np.array(distri))
	# print("pop distributions:::\n",pop_distri)
	# pop_distri=np.array(pop_distri)
	return pop_distri;

def get_pop_distri_label(x_dataset, centroidset, K): #
	X=np.array(x_dataset)
	label_alot = np.zeros(len(x_dataset))
	# print("cs :: ",centroidset)
	for pp,point_id in zip(X,range(0,len(X))):
		distance=-1
		ind=-1
		lol=0
		for centroid,cid in zip(centroidset,range(0,len(centroidset))):
			dist = centroid - pp 
			dist=dist*dist
			dist=np.sum(dist)
			if distance < 0:
				distance = dist
				ind = lol
			elif dist < distance:
				distance = dist 
				ind = lol
			lol = lol + 1
		label_alot[point_id]=ind
	# print("pop distributions:::\n",pop_distri)
	# pop_distri=np.array(pop_distri)
	return label_alot;


def find_fitness(x_dataset, population, K):
	X=np.array(x_dataset)
	pop=np.array(population)
	fitness = np.zeros((len(population),constants.funcs)) # number of functions
	# fitness = fitness.reshape(fitness.shape + (constants.funcs,)) # reshape the fitness array 
	pop_distri = get_pop_distri(x_dataset, population, K)


	#print(silhoutte(X,pop_distri[0],0,K))


	for cc,i in zip(range(0,pop.shape[0]),pop):
		# gotback = silhoutte(X,pop_distri[cc],i,K)
		# print("got back is ::: ",gotback)
		# print("type of return ::",type(gotback))
		# type(gotback)
		#fitness[cc][0]=silhoutte(X,pop_distri[cc],i,K)
		#fitness[cc][1]=gen(X,pop_distri[cc],i,K)
		#fitness[cc][1]=0
		#fitness[cc][2]=5*(1-C_Index(X,pop_distri[cc],i,K))
		fitness[cc][3],fitness[cc][4],fitness[cc][6]=BallandHall_CnH(X,pop_distri[cc],i,K)
		fitness[cc][3] = 10*fitness[cc][3]
		fitness[cc][4] = fitness[cc][4]/10
		#fitness[cc][3] = 0
		#fitness[cc][5] = Dunn(X,pop_distri[cc],i,K)

	for cc,i in zip(range(0,pop.shape[0]),pop):
		labelled = get_pop_distri_label(x_dataset,pop[cc],constants.K)
		ul = np.unique(labelled)
		if ul.size != constants.K :
			fitness[cc]=-(1e9)

	return fitness

def dist_calc(a, b):
	a_ = np.array(a,dtype=np.float64)
	b_ = np.array(b,dtype=np.float64)
	# return np.sqrt(np.sum((a_-b_)*(a_-b_)))
	return np.linalg.norm(a_-b_)

def silhoutte(X,pop_distri,centroids,K):
	a=np.zeros(X.shape[0])
	b=np.zeros(X.shape[0])
	S=np.zeros(K)
	ind=0
	for cc,i in zip(range(0,K),np.array(pop_distri)):
		i=np.array(i)
		if len(i) == 0: 
			continue
		if len(i) == 1: 
			a[ind]=0  # this is to be checked...
			ind = ind + 1
		else:
			for cno1,point in zip(range(len(i)),i):
				ss=0.0
				for cno2,point_ in zip(range(len(i)),i):
					if cno1 != cno2 :
						ss=ss+math.sqrt(np.sum((point-point_)*(point-point_)))
				ss=float(ss)
				ss=ss/(len(i)-1)
				a[ind]=ss
				ind = ind + 1
	ind = 0 
	for cc,i in zip(range(0,K),pop_distri):
		for pp in i:
			b[ind] = -1 
			for cc_,i_ in zip(range(0,K),pop_distri):
				i_=np.array(i_)
				sm = 0
				if len(i_) == 0:
					continue
				if cc_ == cc: 
					continue
				for pp_ in i_:
					sm=sm+dist_calc(pp_,pp)
				sm=float(sm)
				sm = sm/len(i_)
				if b[ind]<0 or b[ind]>sm:
					b[ind]=sm

			ind = ind + 1


	ind = 0
	overall_sum=0
	count = 0
	for i in range(0,S.shape[0]):
		if pop_distri[i].shape[0] == 0: 
			continue
		count = count + 1
		for j in pop_distri[i]:
			S[i] = S[i]+(b[ind]-a[ind])/max(b[ind],a[ind])
			ind = ind + 1
		S[i]=S[i]/pop_distri[i].shape[0]
		overall_sum = overall_sum+S[i]
	overall_sum = overall_sum / count
	# print("overall_sum shape ",overall_sum.shape)
	# print("overall_sum value is :: ",overall_sum)
	return overall_sum

def gen(X,pop_distri,centroids,K):
	overall_sum = 0
	for i in range(0,K):
		for pp in pop_distri[i]:
			overall_sum = overall_sum + dist_calc(pp,centroids[i])
	return -overall_sum

def C_Index(X,pop_distri,centroids,K):	
	Sw=0
	Smx=0
	Smn=0
	Nw=0
	Nt=len(X)*(len(X)-1)//2
	all_distances=[]
	for i in pop_distri:
		# print("i is :: ",i, " when centroids are ::: ",centroids)
		Nw+=len(i)*((len(i))-1)//2
		for j in range(0,len(i)):
			for k in range(j+1,len(i)):
				Sw=Sw+dist_calc(i[j],i[k])
				# print("Sw is ::",Sw, " ",i[j]," ",i[k]," is :: ",dist_calc(i[j],i[k]))

	for i in range(0,len(X)):
		for j in range(i+1,len(X)):
			all_distances.append(dist_calc(X[i],X[j]))

	all_distances.sort()
	Smn=np.sum(np.array(all_distances[:Nw]))
	Smx=np.sum(np.array(all_distances[-Nw:]))
	# print("vals :: ",Sw,Smn,Smx)
	if (Smx == Smn):
		return -999999999
	return (Sw-Smn)/(Smx-Smn)

def SumofSquareWithincluster(X,pop_distri,centroids,K):

	SSW = 0

	for i,x in zip(pop_distri,range(0,len(centroids))):
		# print("i is :: ",i, " when centroids are ::: ",centroids)
		for j in range(0,len(i)):
			SSW = SSW + dist_calc(i[j],centroids[x])

	SSW = 1.0 * SSW / len(X)
	return SSW 


def SumofSquareBetweenClusters(X,pop_distri,centroids,K):
	Mem = np.zeros(len(X[0]))
	for i in X:
		Mem = Mem + np.array(i)
	Mem = Mem/len(X)
	SSB = 0
	for i,x in zip(pop_distri,range(0,len(centroids))):
		SSB = SSB + len(i)*(dist_calc(centroids[x],Mem))
	SSB = SSB / len(X)
	return SSB

def BallandHall_CnH(X,pop_distri,centroids,K):
	SSW = SumofSquareWithincluster(X,pop_distri,centroids,K)
	SSB = SumofSquareBetweenClusters(X,pop_distri,centroids,K)
	return (-SSW/K,(SSB*(len(X)-K))/((K-1)*SSW),math.log((SSB/SSW),10))

def Dunn(X,pop_distri,centroids,K):

	dmin = 1e15
	f=0
	for i in range(0,len(pop_distri)):
		for j in range(i+1,len(pop_distri)):
			for x in range(0,len(pop_distri[i])):
				for y in range(0,len(pop_distri[j])):
					dmin = min(dmin,dist_calc(pop_distri[i][x],pop_distri[j][y]))
					f=1
	dmax = 0.0
	for i in pop_distri:
		for x in range(0,len(i)):
			for y in range(x+1,len(i)):
				dmax=max(dmax,dist_calc(i[x],i[y]))

	if (f == 1):
		return dmin/dmax
	else:
		return -(1e9)


def fmeasure(label,classdata):
	set1 = set()
	label = label.astype(int)
	for x in range(0,label.size) :
		for y in range(x+1,label.size):
			if label[x]==label[y]:
				set1.add((x,y))

	#print(set1)
	classdata = np.array(classdata)
	set2 = set()
	for x in range(0,label.size) :
		for y in range(x+1,label.size):
			if classdata[x]==classdata[y]:
				set2.add((x,y))

	#print(set2)
	a = set1.intersection(set2).__len__()
	#print(a)
	b = (set1-set2).__len__()
	#print(b)
	c = (set2-set1).__len__()
	#print(c)

	print('F MEASURE  : ',2*a/(2*a+b+c))