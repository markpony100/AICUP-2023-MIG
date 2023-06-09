import os
import json
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
import math
#-------global variables-------
_file_ouput=False
_reduced_data=True
_float_size=4
embedder = SentenceTransformer('hfl/chinese-roberta-wwm-ext-large')
#------------------------------
#-------helper functions-------
def find_center(_cluster,id_lst):
	global vec_cos_sim
	e=2.718281828459
	import math

	temp=[[0 if i==j else math.log(abs(1-vec_cos_sim[id_lst.index(i)][id_lst.index(j)])+0.000000001,e) for j in _cluster] for i in _cluster]
	_max=-10000000000
	tp=''
	for i in range(len(temp)):
		if sum(temp[i])>_max:
			tp=_cluster[i]
			_max=sum(temp[i])
	return tp
def dict_index(v,d,_all=False):
	return [i for i in d if v in d[i]] if _all else [i for i in d if v in d[i]][0]
def cos_sim(a,b):
	global _reduced_data,_float_size
	return round(float(np.dot(a,b)/(norm(a)*norm(b))),_float_size) if _reduced_data else float(np.dot(a,b)/(norm(a)*norm(b)))
def jl(p):
	with open(p,'r',encoding='utf8') as f:
		lst=list(f)
	return [json.loads(s) for s in lst]
def sub_string_2_l(l1,l2):
	
	s1=''
	s2=''
	for i in l1:
		s1+=i
	for i in l2:
		s2+=i
	sub1=s1
	sub2=s2
	

	anchor=0
	_max_sub_len=0
	sub_s=''
	temp_s=''
	sub_str_lst=[]
	jd=0
	for i in range(len(s1)):
		temp=[s2.find(s1[i:i+j]) for j in range(len(s2))]
		anchor=temp.index(-1)-1 if -1 in temp else len(s2)
		if anchor>0 and len(s1[i:i+anchor])>1:
			sub_str_lst.append(s1[i:i+anchor])
	for i in sub_str_lst:
		if(len(i)>1):
			sub1=sub1.replace(i,'')
			sub2=sub2.replace(i,'')

	return sub1,sub2
def sub_string(s1,s2):
	sub1=''
	sub2=''
	for i in range(len(s1)):
		if s1[i]!=s2[i]:
			sub1+=s1[i]
			sub2+=s2[i]
	return sub1,sub2
def sub_string_2(s1,s2):
	sub1=s1
	sub2=s2
	
	anchor=0
	_max_sub_len=0
	sub_s=''
	temp_s=''
	sub_str_lst=[]
	jd=0
	for i in range(len(s1)):
		temp=[s2.find(s1[i:i+j]) for j in range(len(s2))]
		anchor=temp.index(-1)-1 if -1 in temp else len(s2)
		if anchor>0 and len(s1[i:i+anchor])>1:
			sub_str_lst.append(s1[i:i+anchor])
	for i in sub_str_lst:
		if(len(i)>1):
			sub1=sub1.replace(i,'')
			sub2=sub2.replace(i,'')

	return sub1,sub2
def relation_compare(sub1,sub2,m_v):
	global embedder
	p1=embedder.encode(sub1)
	p2=embedder.encode(sub2)
	_x=cos_sim((p1-p2),m_v)
	return _x

def logistic(x_r,y_r,x_e,_rt=False):
	from sklearn import linear_model
	from sklearn.inspection import permutation_importance
	model=linear_model.LogisticRegression(max_iter=100000)
	global flag_d,_p_e,_tp_lst
	model.fit(x_r,y_r)


	p_e=model.predict(x_e)
	
	#s=model.score(x_e,y_e)
	if _rt:
		return s
	else:
		return p_e

def most_sim():
	global train_test_sim,train_id,test_id
	temp=[]
	temp_max=[]
	ncs_lst=np.array(train_test_sim)
	for e in range(len(test_id)):
		
		_max=-1
		_max_id=0
		mtp=list(ncs_lst[:,e])	
		_max=max(mtp)
		_max_id=train_id[mtp.index(_max)]
		
		#print(_max_id)
		#print(test_id[e])
		temp.append(_max_id)
		temp_max.append(_max)
	return temp,temp_max
def voting(cluster,id_lst,train_lst,claim_id,claim,pred_cluster):

	global _l,_l_id,train_test_sim,train_id,test_id
	vote_slot={'supports':0,'refutes':0,'NOT ENOUGH INFO':0}
	temp_evidence=[]
	for i in cluster[pred_cluster]:
		tp=train_lst[id_lst.index(i)]
		#print(tp)
		
		w=train_test_sim[train_id.index(i)][test_id.index(claim_id)]
		#w=cos_sim(vtp,claim_vec)
		#print(w)
		vote_slot[tp['label']]+=w
		etp=[]
		if tp['label']!='NOT ENOUGH INFO':
			etp=[i[2:] for i in tp['evidence'][0]]
		temp_evidence.append([tp['label'],w,etp])

	pred_label='NOT ENOUGH INFO' if _l[_l_id.index(claim_id)]!='ENOUGH' else sorted(temp_evidence[:2],key=lambda s:s[1],reverse=True)[0][0]

	padding_evidence=[i for i in temp_evidence if i[0]!=pred_label]
	padding_evidence=sorted(padding_evidence,key=lambda s:s[1],reverse=True)
	padding_pred_evidence=[]

	temp_evidence=[i for i in temp_evidence if i[0]==pred_label]
	temp_evidence=sorted(temp_evidence,key=lambda s:s[1],reverse=True)
	pred_evidence=[]
	
	#print(temp_evidence)
	if pred_label!='NOT ENOUGH INFO':
		for i in temp_evidence:
			pred_evidence+=i[-1]#i[2]
		for i in padding_evidence:
			if i[0]!='NOT ENOUGH INFO':
				padding_pred_evidence+=i[-1]
	tp_evidence=list_set(pred_evidence)+list_set(padding_pred_evidence)
	tp_evidence=list_set(tp_evidence)
	pred_evidence=tp_evidence[:5]
	#print(pred_label)
	#print(pred_evidence)
	return pred_label,pred_evidence
def list_set(lst):
	tp=[]
	for i in lst:
		if i not in tp:
			tp.append(i)
	return tp
#------------------------------
#-------processing functions-------
def train_claim_to_cos_sim(f):
	global embedder,_file_ouput
	
	vec_lst=[embedder.encode(e['claim']) for e in f]
	id_lst=[e['id'] for e in f]
	vec_cos_sim=[[0 for j in range(len(id_lst))] for i in range(len(id_lst))]
	for i in range(len(id_lst)):
		#print(i)
		for j in range(len(id_lst)):
			if i==j:
				vec_cos_sim[i][j]=1
			elif(i<j):
				tp=cos_sim(vec_lst[i],vec_lst[j])
				vec_cos_sim[i][j]=tp
				vec_cos_sim[j][i]=tp
	if _file_ouput:
		file= open("./processing_file/train_cos_sim.json","w",encoding='utf8')
		json.dump({"vec_sim":vec_cos_sim,"id":id_lst},file,ensure_ascii=False)
		file.close()

	return vec_cos_sim,id_lst
def cc_clusters(o_id_lst,v_sim_lst,r,r_r=2):
	global _file_ouput
	id_lst=sorted(o_id_lst)
	sim_lst=[[(id_lst[i],id_lst[j]) for j in range(len(v_sim_lst)) if i!=j and v_sim_lst[o_id_lst.index(id_lst[i])][o_id_lst.index(id_lst[j])]>=r]for i in range(len(v_sim_lst))]
	n_clusters={}
	ct=0
	jd=[]
	rr=r**r_r
	jjd=[]
	for i in range(len(id_lst)):
		#print(i)
		if sim_lst[i]==[] and (id_lst[i] not in jd):
			
			ct+=1
			temp=[id_lst[i]]
			jd.append(id_lst[i])
			n_clusters[str(ct)]=temp
		else:
			
			tp=[e[1] for e in sim_lst[i]]
			if id_lst[i] not in jd:

				ct+=1
				jd.append(id_lst[i])
				temp=[id_lst[i]] 
				for e in tp:
					if e not in jd:
						temp.append(e)
						jd.append(e)
						ttp=[ee[1] for ee in sim_lst[id_lst.index(e)]]
						for ee in ttp:
							if ee not in jd:
								if v_sim_lst[o_id_lst.index(ee)][o_id_lst.index(id_lst[i])]>=rr:
									temp.append(ee)
									jd.append(ee)
				n_clusters[str(ct)]=temp
			else:
				
				ttps=dict_index(id_lst[i],n_clusters,1==1)
				#ttp=dict_index(id_lst[i],n_clusters)
				#id=n_clusters[ttp][0]
				#id=find_center(n_clusters[ttp],id_lst)
				ids=[find_center(n_clusters[e],id_lst) for e in ttps]

				for e in tp:
					if e not in jd:
						max_cs=-1
						max_id=-1
						for _e in ids:
							_tp=v_sim_lst[o_id_lst.index(_e)][o_id_lst.index(e)]
							if _tp>=rr and _tp>max_cs:
								max_id=_e
						if max_id!=-1:
						
							jd.append(e)
							n_clusters[ttps[ids.index(max_id)]].append(e)
						else:							
							ct+=1
							temp=[e]
							jd.append(e)
							n_clusters[str(ct)]=temp
	if _file_ouput:
		file= open("./processing_file/cc_clusters_"+str(r)[2:]+".json","w",encoding='utf8')
		json.dump({'clusters':n_clusters},file,ensure_ascii=False)
		file.close()
	
	return n_clusters
def train_test_cos_sim(train_data,test_data):
	global embedder,_file_ouput
	train_vec=[embedder.encode(i['claim']) for i in train_data]
	test_vec=[embedder.encode(i['claim']) for i in test_data]
	train_id_lst=[i['id'] for i in train_data]
	test_id_lst=[i['id'] for i in test_data]

	for i in range(len(train_data)):
		for j in range(len(test_data)):
			tp=cos_sim(train_vec[i],test_vec[j])
			vec_sim_lst[i][j]=tp
			#vec_sim_lst[i][j]=round(float(tp),5)

	if _file_ouput:
		file= open("./processing_file/train_test_vec_sim.json","w",encoding='utf8')
		json.dump({'vec_sim':vec_sim_lst,'train_id':train_id_lst,'test_id':test_id_lst},file,ensure_ascii=False)
		file.close()
	return vec_sim_lst,train_id_lst,test_id_lst
def enough_train_f(_cluster,train_claim,train_id,train_label):
	global embedder,_file_ouput

	p1=embedder.encode('支持')
	p2=embedder.encode('反對')
	m_v=p1-p2
	relation_dt=[]
	ct=0

	
	p_e=[]
	p_s=['']
	for i in _cluster:
		ct+=1
		#print(ct)
		if len(_cluster[i])>1:
			c_tp=find_center(_cluster[i],train_id)
			c_ctp=train_claim[train_id.index(c_tp)].replace(' ','')
			c_ltp=train_label[train_id.index(c_tp)]
			for e in _cluster[i]:
				#print(e)
				ctp=train_claim[train_id.index(e)].replace(' ','')
				if c_ctp!=ctp:
					ltp=train_label[train_id.index(e)]
					sub1,sub2=sub_string(c_ctp,ctp) if len(c_ctp)==len(ctp) else sub_string_2(c_ctp,ctp)
					temp=relation_compare(sub1,sub2,m_v)
					
					if math.isnan(temp):
						temp=-2
					else:
						temp=float(temp)
					
					temp=[int(i)]+[vec_sim_lst[train_id.index(c_tp)][train_id.index(e)]]+[temp,(1 if c_ltp!= 'NOT ENOUGH INFO' else 0),(1 if ltp!= 'NOT ENOUGH INFO' else 0)]
					relation_dt.append(temp)

	if _file_ouput:
		file= open("./processing_file/label_classifier_train.json","w",encoding='utf8')
		json.dump({'data':temp},file,ensure_ascii=False)
		file.close()
	return relation_dt
def enough_test_pred(tr,te,train_id_lst,test_id_lst,vec_cos_sim_lst,_cluster,label_train):
	global embedder,_file_ouput
	p1=embedder.encode('支持')
	p2=embedder.encode('反對')
	m_v=p1-p2
	x_r=np.array(label_train)[:,:-1]
	y_r=np.array(label_train)[:,-1]
	te_f=[]
	for i in range(len(test_id_lst)):
		#print(i)
		tp=list(vec_cos_sim_lst[:,i])
		ttp=tp.index(max(tp))

		cid=dict_index(train_id_lst[ttp],_cluster)
		tec=te[i]['claim'].replace(' ','')
		trc=tr[ttp]['claim'].replace(' ','')
		l1=1 if tr[ttp]['label'] != 'NOT ENOUGH INFO' else 0 
		sub1,sub2=sub_string(trc,tec) if len(trc)==len(tec) else sub_string_2(trc,tec)
		r_temp=relation_compare(sub1,sub2,m_v)
		if math.isnan(r_temp):
			r_temp=-2
		else:
			r_temp=float(r_temp)
		trid=train_id_lst.index(tr[ttp]['id'])
		teid=test_id_lst.index(te[i]['id'])
		_f=[int(cid),vec_cos_sim_lst[trid][teid],r_temp,l1]
		te_f.append(_f)
	
	p_l=logistic(x_r,y_r,te_f)
	if _file_ouput:
		file= open("./processing_file/label_classifier_te_f.json","w",encoding='utf8')
		json.dump({'test_features':te_f,'id':test_id_lst,'pred':["ENOUGH" if e==1 else "NOT ENOUGH INFO" for e in p_l]},file,ensure_ascii=False)
		file.close()
	#"""
	return ["ENOUGH" if e==1 else "NOT ENOUGH INFO" for e in p_l]
	#for i in range(len(p_l)):
	#	print(test_id_lst[i],'ENOUGH' if p_l[i]==1 else 'NOT ENOUGH INFO')
def pred_output(train_data,test_data,train_id,test_id,cluster):
	ns_claim_lst=[i['claim'].replace(' ','') for i in train_data]
	
	temp,temp_max=most_sim()
	test_max_cs_cluster=[dict_index(_e,cluster) for _e in temp]
	
	with open('./ans_cc_cluster.jsonl','w') as f:
		for e in range(len(test_data)):
			_e=temp[e]
			
			
			tp={}
			tp["id"]=test_data[e]["id"]
			if test_data[e]["claim"].replace(' ','') not in ns_claim_lst:
				tp_l,tp_e=voting(cluster,train_id,train_data,test_id[e],test_data[e]['claim'],test_max_cs_cluster[e])
				tp["predicted_label"]=tp_l
				tp["predicted_evidence"]=tp_e
			
			else:

				if test_data[e]["claim"].replace(' ','') in ns_claim_lst:
					tp_e=ns_claim_lst.index(test_data[e]["claim"].replace(' ',''))
					_e=train_id[tp_e]

				tp["predicted_label"]=train_data[train_id.index(_e)]['label']
				ttp=[]
				if train_data[train_id.index(_e)]['label']!="NOT ENOUGH INFO":
					ttp=train_data[train_id.index(_e)]['evidence']
					ttp=[i[2:] for i in ttp[0]]
				tp["predicted_evidence"]=ttp
				#print(tp)
			f.write(str(tp).replace("\'","\"")+'\n')
#------------------------------
if "processing_file" not in os.listdir('.'):
    print("open directory for processing_file")
    os.mkdir("processing_file")
	#os.popen("mkdir processing_file")
    
else:
	...
file_check=os.listdir('./processing_file')

_file_ouput=(1==1)
train_data=jl("../train_all.jsonl")#"train_all.jsonl"
print("claim_to_cos_sim")
if "train_cos_sim.json" in file_check:
	f=json.load(open('./processing_file/train_cos_sim.json','r'))
	vec_cos_sim=f['vec_sim']
	id_lst=f['id']
else:
	vec_cos_sim,id_lst=train_claim_to_cos_sim(train_data)
vec_sim_lst=vec_cos_sim

print("cos_sim_to_clusters")
_r=0.96
if "cc_clusters_"+str(_r)[2:]+".json" in file_check:
	train_cluster=json.load(open('./processing_file/cc_clusters_'+str(_r)[2:]+'.json','r'))['clusters']
else:
	train_cluster=cc_clusters(id_lst,vec_cos_sim,_r,2)

test_data=jl("../test_all.jsonl")#"../test_all.jsonl"
print("train_test_cos_sim")
if "train_test_vec_sim.json" in file_check:
	f=json.load(open('./processing_file/train_test_vec_sim.json','r'))
	train_test_sim=f['vec_sim']
	train_id=f['train_id']
	test_id=f['test_id']
	tr_id=[e['id'] for e in train_data]
	te_id=[e['id'] for e in test_data]
	train_data=[train_data[tr_id.index(train_id[i])] for i in range(len(train_data))]
	test_data=[test_data[te_id.index(test_id[i])] for i in range(len(test_data))]
else:
	train_test_sim,train_id,test_id=train_test_cos_sim(train_data,test_data)
train_label=[e['label'] for e in train_data]
train_claim=[e['claim'] for e in train_data]

print("not_enough_predicting")
if "label_classifier_te_f.json" in file_check:
	test_revise_label_pred=json.load(open('./processing_file/label_classifier_te_f.json','r'))['pred']
else:
	enough_train=enough_train_f(train_cluster,train_claim,train_id,train_label)
	test_revise_label_pred=enough_test_pred(train_data,test_data,train_id,test_id,np.array(train_test_sim),train_cluster,enough_train)

_l=test_revise_label_pred
_l_id=test_id

print("test_pred_output")
pred_output(train_data,test_data,train_id,test_id,train_cluster)
