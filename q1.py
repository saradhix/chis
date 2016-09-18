import os
import re
import utils
import sys
from sklearn.naive_bayes import MultinomialNB

def main():

  filename = 'does_sun_exposure_cause_skin_cancer_final.txt'

  fd = open(filename,'r')
  num_samples = 0
  num_correct = 0
  num_supp_correct = 0

  features=['skin','cancer','sun','exposure','melanoma','risk','sunlight','uv','ultraviolet',
'radiation','cell','damage','light','people','more','developing','sunburn','rays','exposed','workers']
  x=[]
  y=[]
  for line in fd:
    X=[]
    
    (document, relevance, support) = line.split('\t')
    relevance = relevance.strip()
    support = support.strip()
    if relevance == "relevant":
      y_rel = 1
    else:
      y_rel = 0
    document = preprocess_doc(document)

    for feature in features:
      if feature in document.split(' '):
        X.append(1)
      else:
        X.append(0)

    x.append(X)
    y.append(y_rel)

  print x, y
  gnb = MultinomialNB()
  y_pred = gnb.fit(x,y).predict(x)
  print("Number of mislabeled points out of a total %d points : %d"
       % (len(x),(y != y_pred).sum()))
'''    if pred_rel == relevance:
      num_correct = num_correct + 1
    if pred_supp == support:
      num_supp_correct = num_supp_correct + 1
    num_samples = num_samples + 1
    print  num_samples, document, "Pred rel=",pred_rel, "Act rel=", relevance, "Pred supp=", pred_supp, "Act supp=", support
    print "-"*30
  accuracy = num_correct / float(num_samples)
  supp_accuracy = num_supp_correct / float(num_samples)
  print accuracy, supp_accuracy

  #print supp_dict
  #print opp_dict
  print sorted(supp_dict.items(), key=lambda x:x[1], reverse=True)
  print sorted(opp_dict.items(), key=lambda x:x[1], reverse=True)
'''
def get_support(document, relevance):
  support_words=['because', 'therefore', 'after',
          'for', 'since', 'when', 'assuming',
          'so', 'accordingly', 'thus', 'hence',
          'then', 'consequently']
  oppose_words=['however', 'but', 'though',
          'except', 'not', 'never', 'no',
          'whereas', 'nonetheless', 'yet',
          'despite','inverse']

  if relevance == "irrelevant":
    return "neutral"

  for word in support_words:
    if word in document.split(' '):
      print word ,"present->Support"
      return "support"

  for word in oppose_words:
    if word in document.split(' '):
      print word ,"present->Oppose"
      return "oppose"

  return "neutral"



def preprocess_doc(doc):
  #lower case, stemming, stop word removal, special character removal

  #lower case conversion
  doc = doc.lower()

  #Special character removal
  doc = re.sub('\W+', ' ', doc)
  return doc

def classify_doc(document):
  groups = ['sun','skin','cancer']

  synonyms={}

  synonyms['sun']=['sun','sunlight','uv','uva','uvb','sunbathers','sunburns',
          'sunburn','exposure','sunbather','indoor','radiation','outdoors','outside',
          'temperature','light','solarium']
  synonyms['skin']=['skinned','burn','melanoma','melanomas', 'damages','causes',
          'exposure','this','skin']
  synonyms['cancer']=['cancer','melanoma','melanomas','cancerous','carcinoma',
          'health problem','this']
  q = 'does sun exposure cause skin cancer'
  sim=0
  for word in groups:
    for syn in synonyms[word]:
      if syn in document:
          sim = sim + 1
          print syn
          break
    if sim >=3:
      pred_rel = "relevant"
    else:
      pred_rel = "irrelevant"
    pred_supp = get_support(document, pred_rel)
  return (pred_rel, pred_supp)

if __name__ == "__main__":
  main() 
