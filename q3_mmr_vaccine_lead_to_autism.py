import os
import re
import utils
import sys

def main():

  filename = 'MMRVaccineLeadToAutism_final.txt'

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
  support_words=['causal', 'measles', 'development','suppress',
          'increase', 'cause','evidence', 'increased','associated','more common','true','side effect','factor','causing','link','develop','autism','exposure','heightened','exhibit','abnormal','autis']
  
  oppose_words=['however', 'though',
          'whereas', 'nonetheless', 'yet',
          'despite']

  reverse_words = ['no ',' no ','oppose',' not ', 'does not','least', 'less','nothing','except','decreased','never','although','inverse','weak','lowest', 'harmless' ]
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
#    print "Checking for word", word
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
  return doc

def classify_doc(document):
  groups = ['vaccine','autism']

  synonyms={}

  synonyms['vaccine']=['this','vaccine','mmr', 'measles mumps and rubella','measles vaccination','thimerosal','vaccin', 'immun','measles mumps or rubella','mercury']
  synonyms['autism']=['this', 'autism','asd','autistic','disorder','incidence', 'outcome','deficit','risk','disease','development','problem','autistic spectrum disorders','neurotoxic']
  q = 'does MMR vaccine lead to autism'
  sim=0
  for word in groups:
    for syn in synonyms[word]:
      if syn in document:
          sim = sim + 1
          print "Synonym=",syn
          break
    if sim >=2:
      pred_rel = "relevant"
    else:
      pred_rel = "irrelevant"
    pred_supp = get_support(document, pred_rel)
  return (pred_rel, pred_supp)

if __name__ == "__main__":
  main() 
