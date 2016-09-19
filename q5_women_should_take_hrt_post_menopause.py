import os
import re
import utils
import sys

def main():

  filename = 'women_should_Take_HRT_Post_Menopause_final.txt'

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
  support_words=['shorten','reduction','benefit','help prevent cold','shorter illness','alleviate','neutral','ward off','halve','slash','reduce','prevented','manage','protective','reduc','decrease','stopped','wards off','most useful','lower','most effective','effective agent','fewer cold','less severe','fighting off','take plenty',' mg','strengthens','lessens','dry up','beneficial','fighting','still high','modest','liveli','preventive', 'strengthen','improve','cut their risk','stop cancer','half','may be effective']
  
  oppose_words=['wrong','doubt','questionable','no benefit','zinc','no effect','no difference', 'wont help', 'no evidence','but not','did not','not show', 'doesnt','isnt','does not', 'little','not the best','not help','will not','disagree','unsupported','or not','not worth', 'not necessary','nor is','he same','but only for','has not been','inconclusive','nausea','diarrhea','dangerous','is not an','debunk','similar symptoms','do not recommend','do not prove','not effective','didnt help','alas','didnt','possibly','disappointing','unjustified']

  reverse_words = ['small','unlikely','nor','aren' ]
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
  doc = doc.replace('won t', 'wont')
  doc = doc.replace('doesn t', 'doesnt')
  doc = doc.replace('isn t', 'isnt')
  doc = doc.replace('didn t', 'didnt')
  return doc

def classify_doc(document):
  groups = ['hrt']

  synonyms={}

  synonyms['hrt']=['this','hrt','hormone','progestin','estrogen','dose','menopaus','cancer',' it ', 'antiox','supplement','consumed','mg ', 'reduction','myth']
  synonyms['menopause']=['this', 'menopause','common cold', 'flu', 'illness','sniffles','immun','sick','throat','nose','effect', 'viral','nausea','diabetes','evidence', 'benefit' ]
  q = 'does vitamin C prevent common cold'
  sim=0
  for word in groups:
    for syn in synonyms[word]:
      if syn in document:
          sim = sim + 1
          print "Synonym=",syn
          break
    if sim >=1:
      pred_rel = "relevant"
    else:
      pred_rel = "irrelevant"
    pred_supp = get_support(document, pred_rel)
  return (pred_rel, pred_supp)

if __name__ == "__main__":
  main() 
