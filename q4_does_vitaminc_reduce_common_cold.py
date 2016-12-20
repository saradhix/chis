import os
import re
import utils
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy
import csv
import rel_sup
from sklearn.metrics import classification_report, confusion_matrix

# fix random seed for reproducibility
seed = 0 
numpy.random.seed(seed)
support_words=['shorten','reduction','benefit','help prevent cold',
        'shorter illness','alleviate','neutral','ward off','halve','slash',
        'reduce','prevented','manage','protective','reduc','decrease','stopped',
        'wards off','most useful','lower','most effective','effective agent',
        'fewer cold','less severe','fighting off','take plenty',' mg',
        'strengthens','lessens','dry up','beneficial','fighting','still high',
        'modest','liveli','preventive', 'strengthen','improve','cut their risk',
        'stop cancer','half','may be effective']
oppose_words=['wrong','doubt','questionable','no benefit','zinc','no effect',
        'no difference', 'wont help', 'no evidence','but not','did not',
        'not show', 'doesnt','isnt','does not', 'little','not the best',
        'not help','will not','disagree','unsupported','or not','not worth',
        'not necessary','nor is','he same','but only for','has not been',
        'inconclusive','nausea','diarrhea','dangerous','is not an','debunk',
        'similar symptoms','do not recommend','do not prove','not effective',
        'didnt help','alas','didnt','possibly','disappointing','unjustified']
reverse_words = ['small','unlikely','nor','aren' ]

syn1=['this','vitamin','zinc','dose','placebo','grams','intake',
        'fruits and vegetables',' it ', 'antiox','supplement','consumed',
        'mg ', 'reduction','myth']
syn2=['this', 'cold','common cold', 'flu', 'illness','sniffles','immun','sick',
      'throat','nose','effect', 'viral','nausea','diabetes','evidence', 'benefit']
features = support_words + oppose_words + reverse_words + syn1 + syn2
def main():

  train_filename = 'vitaminC_common_cold_final.txt'
  test_filename = 'vitaminC_common_cold_test.txt'
  results_filename = 'results/vitaminC_common_cold.csv'

  fd = open(train_filename,encoding='latin-1')
  num_samples = 0
  num_correct = 0
  num_supp_correct = 0

  X=[]
  y1=[]
  y2=[]

  for line in fd:
    (document, relevance, support) = line.split('\t')
    relevance = relevance.strip()
    support = support.strip()
    document = preprocess_doc(document)

    current_x = make_x(document)

    print( current_x)
    if relevance == "relevant":
      current_y1=1
    else:
      current_y1=0
    if support == "oppose":
      current_y2 = [1,0,0];
    if support == "neutral":
      current_y2 = [0,1,0];
    if support == "support":
      current_y2 = [0,0,1]

    X.append(current_x)
    y1.append(current_y1)
    y2.append(current_y2)
  fd.close()
  X = numpy.array(X)
  y1 = numpy.array(y1)
  y2 = numpy.array(y2)
  print( X.shape)
  print( y1.shape)
  print( y2.shape)

  print( "Number of features=",len(features))
  model = Sequential()
  model.add(Dense(180, input_dim=len(features), init='uniform', activation='relu'))
  model.add(Dense(8, init='uniform', activation='relu'))
  model.add(Dense(1, init='uniform', activation='sigmoid'))
  # Compile model
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  # Fit the model
  model.fit(X, y1, nb_epoch=150, batch_size=10)
  # evaluate the model
  scores = model.evaluate(X, y1)
  print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
  print( "Running predictions for support")

  model2 = Sequential()
  model2.add(Dense(180, input_dim=len(features), init='uniform'))
  model2.add(Activation('tanh'))
  model2.add(Dropout(0.5))
  model2.add(Dense(150, init='uniform'))
  model2.add(Activation('tanh'))
  model2.add(Dropout(0.5))
  model2.add(Dense(3, init='uniform'))
  model2.add(Activation('softmax'))

  #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
  #model2.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
  model2.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

  model2.fit(X, y2, nb_epoch=150, batch_size=10)
  score = model2.evaluate(X, y2, batch_size=10)
  print("%s: %.2f%%" % (model2.metrics_names[1], score[1]*100))

  #Do predictions on test
  X=[]
  y1=[]
  y2=[]
  documents=[]
  fd = open(test_filename,'r')
  for line in fd:
    document = line.strip()
    documents.append(document)
    document = preprocess_doc(document)
    current_x = make_x(document)
    X.append(current_x)

  relevances = model.predict(X)
  supports = model2.predict(X)

  relevances = [int(x+0.5) for x in relevances]
  print( relevances)
  new_supports = [0 for i in supports]
  for (i,support) in enumerate(supports):
    new_supports[i] = vec_to_class(support)
  print( new_supports)
  (true_rel, true_sup) = rel_sup.get_rel_sup('q4_rel.txt', 'q4_sup.txt')
  print( len(true_rel), len(true_sup))
  print( confusion_matrix(true_rel, relevances))
  print( classification_report(true_rel, relevances))

  print( confusion_matrix(true_sup, new_supports))
  print( classification_report(true_sup, new_supports))

  fd = open(results_filename,'w')
  datawriter = csv.writer(fd)
  for (i,document) in enumerate(documents):
    datawriter.writerow([document, label_name_y1(relevances[i]), label_name_y2(new_supports[i])])


def vec_to_class(support):
  support = support.tolist()
  return support.index(max(support))

def label_name_y1(label):
  if label==0:
    return "irrelevant"
  else:
    return "relevant"

def label_name_y2(label):
  if label==0:
    return "oppose"
  elif label==1:
    return "neutral"
  else:
    return "support"


def preprocess_doc(doc):
  #lower case, stemming, stop word removal, special character removal

  #lower case conversion
  doc = doc.lower()

  #Special character removal
  doc = re.sub('\W+', ' ', doc)
  return doc

def make_x(document):
  current_x = [0 for f in features]
  for (i, feature) in enumerate(features):
    if feature in document:
      current_x[i]=1
  return current_x
if __name__ == "__main__":
  main() 
