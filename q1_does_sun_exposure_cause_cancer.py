import os
import re
import utils
import sys

def main():

  filename = 'does_sun_exposure_cause_skin_cancer_final.txt'

  fd = open(filename,'r')
  num_samples = 0
  num_correct = 0
  num_supp_correct = 0
  supp_dict={}
  opp_dict = {}
  neut_dict = {}

  for line in fd:
    (document, relevance, support) = line.split('\t')
    relevance = relevance.strip()
    support = support.strip()
    document = preprocess_doc(document)
    if support=="support":
      for word in document.split(' '):
        supp_dict[word]=supp_dict.get(word,0)+1

    if support=="oppose":
      for word in document.split(' '):
        opp_dict[word]=opp_dict.get(word,0)+1
      
    (pred_rel, pred_supp) = classify_doc(document)
    if pred_rel == relevance:
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
  #print sorted(supp_dict.items(), key=lambda x:x[1], reverse=True)
  #print sorted(opp_dict.items(), key=lambda x:x[1], reverse=True)
def get_support(document, relevance):
  support_words=['because', 'therefore', 'after',
          'since', 'when', 'assuming',
          'so', 'accordingly', 'thus', 'hence',
          'then', 'consequently', 'increase', 'intense','cause','evidence',
'increased','harmful','develops','exposed','exposure','causes','associated','more common',
'postulated']
  oppose_words=['however', 'though',
          'except', 'not', 'never', 'no',
          'whereas', 'nonetheless', 'yet',
          'despite','inverse','less risk','lower risk']

  if relevance == "irrelevant":
    return "neutral"

  for word in support_words:
    if word in document:
      print word ,"present->Support"
      return "support"

  for word in oppose_words:
    if word in document:
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
          #print syn
          break
    if sim >=3:
      pred_rel = "relevant"
    else:
      pred_rel = "irrelevant"
    pred_supp = get_support(document, pred_rel)
  return (pred_rel, pred_supp)

if __name__ == "__main__":
  main() 
