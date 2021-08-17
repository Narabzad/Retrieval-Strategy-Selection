import pickle 
from sentence_transformers.cross_encoder import CrossEncoder

model_name="sparse_vs_hybrid_bert-base-uncased_e1_b16"
model = CrossEncoder("models/"+model_name, num_labels=1)

sentences = []
queries=[]

qfile=open('queries+firstdoc.small.dev.tsv','r').readlines()
for line in qfile:
    qid,qtext,doctext=line.rstrip().split('\t')
    sentences.append([qtext,doctext])
    queries.append(qid)

scores=model.predict(sentences)


out=open('results/prediction_'+model_name+'.dev.small.tsv','w')
for i in range(len(sentences)):
    out.write(queries[i]+'\t'+sentences[i][0]+'\t'+str(scores[i])+'\n')
out.close()

