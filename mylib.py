import nltk
from nltk.corpus import stopwords


def get_nouns_and_verbs(sentence):
  pos_result = nltk.pos_tag(nltk.word_tokenize(sentence))
  result = []
  #print pos_result
  for t in pos_result:
    if t in stopwords.words("english"):
      continue
    if str(t[1]).startswith('NN'):
      result.append(t[0])
    if str(t[1]).startswith('VB'):
      result.append(t[0])
  return result

def similarity(a, b):
  alist = get_nouns_and_verbs(a)
  blist = get_nouns_and_verbs(b)
  return jaccard_similarity(alist, blist)

def jaccard_similarity(x,y):
  #print "Finding jaccard similarity of "
  #print x
  #print "and"
  #print y
  intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
  union_cardinality = len(set.union(*[set(x), set(y)]))
  #print intersection_cardinality, union_cardinality
  return intersection_cardinality/float(union_cardinality)

def make_bigrams(para):
  text = para.split('.')
  text = [ word.strip() for word in text]
  bigrams = [b for l in text for b in zip(l.split(" ")[:-1], l.split(" ")[1:])]
  return bigrams
def make_trigrams(para):
  text = para.split('.')
  text = [ word.strip() for word in text]
  bigrams = [b for l in text for b in zip(l.split(" ")[:-1], l.split(" ")[1:],
      l.split(" ")[2:])]
  return bigrams
def make_fourgrams(para):
  text = para.split('.')
  text = [ word.strip() for word in text]
  bigrams = [b for l in text for b in zip(l.split(" ")[:-1], l.split(" ")[1:],
      l.split(" ")[2:], l.split(" ")[3:])]
  return bigrams
def make_fivegrams(para):
  text = para.split('.')
  text = [ word.strip() for word in text]
  bigrams = [b for l in text for b in zip(l.split(" ")[:-1], l.split(" ")[1:],
      l.split(" ")[2:], l.split(" ")[3:], l.split(" ")[4:])]
  return bigrams
def make_sixgrams(para):
  text = para.split('.')
  text = [ word.strip() for word in text]
  bigrams = [b for l in text for b in zip(l.split(" ")[:-1], l.split(" ")[1:],
      l.split(" ")[2:], l.split(" ")[3:], l.split(" ")[4:], l.split(" ")[5:])]
  return bigrams
