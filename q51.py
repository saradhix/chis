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

  filename = 'women_should_Take_HRT_Post_Menopause_final.txt'

  fd = open(filename,'r')
  num_samples = 0
  num_correct = 0
  num_supp_correct = 0

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
    print  num_samples, document, "Pred=",pred_rel, "Truth=", relevance, "Support=", support, "Pred supp=", pred_supp
    print "-"*30
  accuracy = num_correct / float(num_samples)
  supp_accuracy = num_supp_correct / float(num_samples)
  print accuracy, supp_accuracy

def get_support(document, relevance):

  keywords=['risk']
  if relevance == "irrelevant":
    return "neutral"

  for keyword in keywords:
    if keyword in document:
      return "support"

  return "oppose"



def preprocess_doc(doc):
  #lower case, stemming, stop word removal, special character removal

  #lower case conversion
  doc = doc.lower()

  #Special character removal
  doc = re.sub('\W+', ' ', doc)
  return doc

def classify_doc(document):
  groups = ['hrt']

  synonyms={}

  synonyms['hrt']=['hrt','hormone replacement therapy','estrogen replacement therapy'
          ]
  synonyms['skin']=['skinned','burn','melanoma','melanomas', 'damages','causes',
          'exposure','this','skin']
  synonyms['cancer']=['cancer','melanoma','melanomas','cancerous','carcinoma',
          'health problem','this']
  q = 'does sun exposure cause skin cancer'
  imp_nouns = ['sun', 'skin', 'cancer']
  sim=0
  for word in groups:
    for syn in synonyms[word]:
      if syn in document:
          sim = sim + 1
          print syn
          break
    if sim >=1:
      pred_rel = "relevant"
    else:
      pred_rel = "irrelevant"
    pred_supp = get_support(document, pred_rel)
  return (pred_rel, pred_supp)

if __name__ == "__main__":
  main() 
