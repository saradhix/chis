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
import xgboost
# fix random seed for reproducibility
seed = 0
numpy.random.seed(seed)


q1={}
q1['train_file_name']='does_sun_exposure_cause_skin_cancer_final.txt'
q1['test_file_name']='does_sun_exposure_cause_skin_cancer_test.txt'
q1['gold_rel']='q1_rel.txt'
q1['gold_sup']='q1_sup.txt'
q1['features']=['because', 'therefore', 'after', 'since', 'when', 'assuming', 'so', 'accordingly', 'thus', 'hence', 'then', 'consequently', 'increase', 'intense', 'cause', 'evidence', 'increased', 'harmful', 'develops', 'exposed', 'exposure', 'causes', 'associated', 'more', 'common', 'postulated', 'proven', 'however', 'though', 'whereas', 'nonetheless', 'yet', 'despite', 'no ', ' no ', 'oppose', ' not ', 'does not', 'least', 'less', 'nothing', 'except', 'decreased', 'never', 'although', 'inverse', 'weak', 'lowest', 'sun', 'sunlight', 'uv', 'uva', 'uvb', 'sunbathers', 'sunburns', 'sunburn', 'exposure', 'sunbather', 'indoor', 'radiation', 'outdoors', 'outside', 'temperature', 'light', 'solarium', 'skinned', 'burn', 'melanoma', 'melanomas', 'damages', 'causes', 'exposure', 'this', 'skin', 'cancer', 'melanoma', 'melanomas', 'cancerous', 'carcinoma', 'health problem', 'this']
q1['name']='Cancer: Does sun exposure cause skin_cancer'



datasets=[q1]
def main():
  print "Running all tests"
  for dataset in datasets:
    run_experiment(dataset)


def run_experiment(dataset):
  task_name = dataset['name']
  (X_train,y1_train,y2_train) = load_train_data(dataset)
  (X_test, y1_test, y2_test) = load_test_data(dataset)

  model1 = train_y1(X_train, y1_train)
  model2 = train_y2(X_train, y2_train)

  y1_predicted = model1.predict(X_test)
  y2_predicted = model2.predict(X_test)

  print task_name, "Task 1"
  print confusion_matrix(y1_test, y1_predicted)
  print classification_report(y1_test, y1_predicted)
  print task_name, "Task 2"
  print confusion_matrix(y2_test, y2_predicted)
  print classification_report(y2_test, y2_predicted)


def load_train_data(dataset):

  train_file_name = dataset['train_file_name']
  fd = open(train_file_name,'r')

  X_train=[]
  y1_train=[]
  y2_train=[]

  for line in fd:
    (document, relevance, support) = line.split('\t')
    relevance = relevance.strip()
    support = support.strip()
    document = preprocess_doc(document)

    current_x = generate_features(document, dataset['features'])

    if relevance == "relevant":
      current_y1=1
    else:
      current_y1=0
    if support == "oppose":
      current_y2 = 0
    if support == "neutral":
      current_y2 = 1
    if support == "support":
      current_y2 = 2

    X_train.append(current_x)
    y1_train.append(current_y1)
    y2_train.append(current_y2)
  fd.close()
  X_train = numpy.array(X_train)
  y1_train = numpy.array(y1_train)
  y2_train = numpy.array(y2_train)
  return X_train, y1_train, y2_train

def load_test_data(dataset):
  rel=dataset['gold_rel']
  sup=dataset['gold_sup']
  test_file_name = dataset['test_file_name']
  features=dataset['features']
  documents=[]
  X=[]
  fd = open(test_file_name,'r')
  for line in fd:
    document = line.strip()
    documents.append(document)
    document = preprocess_doc(document)
    current_x = generate_features(document,dataset['features'])
    X.append(current_x)

  (rel,sup)=rel_sup.get_rel_sup(rel,sup)
  return X, rel, sup

def train_y1(X_train, y1_train):
  modelx = xgboost.XGBClassifier()
  modelx.fit(X_train, y1_train)
  return modelx

def train_y2(X_train, y2_train):
  modelx = xgboost.XGBClassifier(objective="multi:softmax")
  modelx.fit(X_train, y2_train)
  return modelx

def preprocess_doc(doc):
  #lower case, stemming, stop word removal, special character removal

  #lower case conversion
  doc = doc.lower()

  #Special character removal
  doc = re.sub('\W+', ' ', doc)
  return doc

def generate_features(document, features):
  current_x = [0 for f in features]
  for (i, feature) in enumerate(features):
    if feature in document:
      current_x[i]=1
  return current_x

if __name__ == "__main__":
  main()
