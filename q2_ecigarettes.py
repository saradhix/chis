import os
import re
import utils
import sys

def main():

  filename = 'ecigarettes_final.txt'

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
    (pred_rel, pred_supp) = classify_doc(document)
    if pred_rel == relevance:
      num_correct = num_correct + 1
    if pred_supp == support:
      num_supp_correct = num_supp_correct + 1
    num_samples = num_samples + 1
    print  num_samples, document, "PR=",pred_rel, "AR=", relevance, "PS=", pred_supp, "AS=", support
    print "-"*30
  accuracy = num_correct / float(num_samples)
  supp_accuracy = num_supp_correct / float(num_samples)
  print accuracy, supp_accuracy

def get_support(document, relevance):
  support_words=['safe', 'therefore', 'healthier',
          'harmless', 'reduced', 'assuming',
          'accordingly', 'thus', 'hence',
          'then', 'consequently', 'increase', 'intense','cause','evidence',
'increased','harmful','develops','exposed','exposure','causes','associated','more common',
'postulated','proven']
  
  oppose_words=['however', 'though',
          'whereas', 'nonetheless', 'yet',
          'despite',]

  reverse_words = ['no ',' no ','oppose',' not ', 'does not','least', 'less','nothing','except','decreased','never','although','inverse','weak','lowest', 'cannot', 'cause','damage', 'harmful','increased risk','harms','adverse','negative effects','accident','hazard','poison','prohibit','explosion','overdose','dangerous','potential risks','addiction','carcino','concerns','deadly','formaldehyde','popcorn','risk of cancer','irritate','anxiety' ]
  reverse = False
  support = 0
  oppose = 0
  if relevance == "irrelevant":
    return "neutral"

  for word in support_words:
    if word in document:
      print word ,"present->Support"
      support = support + 1 

  for word in oppose_words:
    if word in document:
      print word ,"present->Oppose"
      oppose = oppose + 1
  
  for word in reverse_words:
    if word in document:
      print word,"Reversed meaning", "s=",support,"o=",oppose
      reverse=True
  if reverse and oppose:
    return "support"
  if reverse and support:
    return "oppose"

  if support:
    return "support"
  if oppose:
    return "oppose"

  return "neutral"



def preprocess_doc(doc):
  #lower case, stemming, stop word removal, special character removal

  #lower case conversion
  doc = doc.lower()

  #Special character removal
  doc = re.sub('\W+', ' ', doc)
  doc = doc.replace('eciggarette','ecigarette')
  doc = doc.replace('e cig','ecig')
  doc = doc.replace(' s ','s ')
  return doc

def classify_doc(document):
  groups = ['ecigarettes',]

  synonyms={}

  synonyms['ecigarettes']=['ecigarette','ecigarettes','vaping','nicotine','liquid','toxic','aldehyde',
          'cigarettes','diacetyl','exposure','vaper','ecig','inhal','vapor','cancer','carcin',
          'electronic cigarette','solarium']
  q = 'does sun exposure cause skin cancer'
  sim=0
  for word in groups:
    for syn in synonyms[word]:
      if syn in document:
          sim = sim + 1
          #print syn
          break
    if sim >=1:
      pred_rel = "relevant"
    else:
      pred_rel = "irrelevant"
    pred_supp = get_support(document, pred_rel)
  return (pred_rel, pred_supp)

if __name__ == "__main__":
  main() 
