import os
import re
import utils
import sys

import mylib

def get_synonyms(sentence):
  syn = ['sunlight','uva','solar','sunbathers','sunburns','melanoma','uv', 
          'sunburn', 'melanomas','exposure','sun']
  return syn

def main():

  filename = 'does_sun_exposure_cause_skin_cancer.tsv'
  q = 'does sun exposure cause skin cancer'
  imp_nouns = ['sun', 'skin', 'cancer']
  syn = get_synonyms(q)
  q = preprocess_doc(q)
  q_nv = mylib.get_nouns_and_adjs(q)
  q_nv.extend(syn)
  print q_nv
  #sys.exit(1)

  fd = open(filename,'r')
  num_samples = 0
  num_correct = 0

  for line in fd:
    (document, relevance) = line.split('\t')
    relevance = relevance.strip()
    document = preprocess_doc(document)
    nv = mylib.get_nouns_and_verbs(document)
    #document = utils.simple_preprocess(document)
    print q_nv
    print nv
    #sim = mylib.jaccard_similarity(q_nv, nv)
    sim = len(set.intersection(set(q_nv),set(nv)))
    if sim > 2:
      pred = "relevant"
    else:
      pred = "irrelevant"
    if pred == relevance:
      num_correct = num_correct + 1
    num_samples = num_samples + 1
    print  document, "Pred=",pred, "Truth=", relevance, sim
    print "-"*30
  accuracy = num_correct / float(num_samples)
  print accuracy

def preprocess_doc(doc):
  #lower case, stemming, stop word removal, special character removal

  #lower case conversion
  doc = doc.lower()

  #Special character removal
  doc = re.sub('\W+', ' ', doc)
  return doc


if __name__ == "__main__":
  main() 
