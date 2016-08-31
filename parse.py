import os
import re
import utils
import sys

import mylib

def main():

  filename = 'does_sun_exposure_cause_skin_cancer.tsv'
  q = 'does sun exposure cause skin cancer'
  q = preprocess_doc(q)
  q_nv = mylib.get_nouns_and_verbs(q)
  print q_nv

  fd = open(filename,'r')
  num_samples = 0
  num_correct = 0

  for line in fd:
    (document, relevance) = line.split('\t')
    relevance = relevance.strip()
    document = preprocess_doc(document)
    nv = mylib.get_nouns_and_verbs(document)
    sys.exit()
    #document = utils.simple_preprocess(document)
    print document
    sim = mylib.jaccard_similarity(q_nv, nv)
    if sim > 0.05:
      pred = "relevant"
    else:
      pred = "irrelevant"
    if pred == relevance:
      num_correct = num_correct + 1
    num_samples = num_samples + 1
    print  "Pred=",pred, "Truth=", relevance
    print "-"*30
  accuracy = num_correct / num_samples
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
