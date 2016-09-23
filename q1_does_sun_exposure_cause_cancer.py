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
# fix random seed for reproducibility
seed = 0 
numpy.random.seed(seed)

support_words=['because', 'therefore', 'after',
          'since', 'when', 'assuming',
          'so', 'accordingly', 'thus', 'hence',
          'then', 'consequently', 'increase', 'intense','cause','evidence',
'increased','harmful','develops','exposed','exposure','causes','associated','more','common',
'postulated','proven']
oppose_words=['however', 'though',
          'whereas', 'nonetheless', 'yet',
          'despite',]
reverse_words = ['no ',' no ','oppose',' not ', 'does not','least', 'less','nothing','except','decreased','never','although','inverse','weak','lowest' ]

syn1=['sun','sunlight','uv','uva','uvb','sunbathers','sunburns',
         'sunburn','exposure','sunbather','indoor','radiation','outdoors','outside',
          'temperature','light','solarium']
syn2=['skinned','burn','melanoma','melanomas', 'damages','causes',
          'exposure','this','skin']
syn3=['cancer','melanoma','melanomas','cancerous','carcinoma',
          'health problem','this']

features = support_words + oppose_words + reverse_words + syn1 + syn2 + syn3
def main():

  train_filename = 'does_sun_exposure_cause_skin_cancer_final.txt'
  test_filename = 'does_sun_exposure_cause_skin_cancer_test.txt'
  results_filename = 'results/does_sun_exposure_cause_skin_cancer_result.csv'

  fd = open(train_filename,'r')
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

    print current_x
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
  print X.shape
  print y1.shape
  print y2.shape

  print "Number of features=",len(features)
  model = Sequential()
  model.add(Dense(120, input_dim=len(features), init='uniform', activation='relu'))
  model.add(Dense(8, init='uniform', activation='relu'))
  model.add(Dense(1, init='uniform', activation='sigmoid'))
  # Compile model
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  # Fit the model
  model.fit(X, y1, nb_epoch=150, batch_size=10)
  # evaluate the model
  scores = model.evaluate(X, y1)
  print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
  print "Running predictions for support"

  model2 = Sequential()
  model2.add(Dense(150, input_dim=len(features), init='uniform'))
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

  relevances = [int(round(x)) for x in relevances]
  print relevances
  new_supports = [0 for i in supports]
  for (i,support) in enumerate(supports):
    new_supports[i] = vec_to_class(support)
  print new_supports

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
