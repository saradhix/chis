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

q2={}
q2['train_file_name']='ecigarettes_final.txt'
q2['test_file_name']='ecigarettes_test.txt'
q2['gold_rel']='q2_rel.txt'
q2['gold_sup']='q2_sup.txt'
q2['features']=['less', 'yet', 'causes', 'inverse', 'except', 'harmless', 'postulated', 'decreased', 'then', 'risk', 'safe', 'ecigarettes', 'cannot', 'effects', 'despite', 'vaping', 'vaper', 'common', 'lung', 'prohibit', 'nicotine', 'exposed', 'concern', 'poison', 'increase', 'safety', 'assuming', 'cause', 'explosion', 'never', 'weak', 'however', 'carcin', 'although', 'does not', 'intense', 'accident', 'cigarette', 'healthier', 'oppose', 'popcorn', 'consequently', 'cancer', 'ecigarette', 'improper', 'carcino', 'least', 'electronic', 'unknown', 'solarium', 'negative', 'irritate', 'accordingly', 'therefore', 'recommend', 'diacetyl', 'more', 'lowest', 'formaldehyde', 'overdose', 'anxiety', 'harms', 'dangerous', 'associated', 'thus', 'proven', 'evidence', 'high', 'increased', 'nonetheless', 'deadly', 'whereas', 'damage', 'adverse', 'aldehyde', 'toxic', 'ecig', 'inhal', 'though', 'potentiali', 'hazard', 'harmful', 'hence', 'develops', 'addict', 'nothing', 'how safe', 'died', 'exposure', 'reduc', 'liquid', 'disease', 'cigarettes', 'serious', 'vapor']
q2['name']='E cigarettes'

q3={}
q3['train_file_name']='MMRVaccineLeadToAutism_final.txt'
q3['test_file_name']='MMRVaccineLeadToAutism_test.txt'
q3['gold_rel']='q3_rel.txt'
q3['gold_sup']='q3_sup.txt'
q3['features']=['exhibit', 'lowest', 'associated', 'less', 'measlesi', 'abnormal', 'spectrum', 'evidence', 'nothing', 'measles vaccination', 'neurotoxic', 'yet', 'autism', 'risk', 'mercury', 'inverse', 'develop', 'disorders', 'nonetheless', 'whereas', 'disease', 'more common', 'least', 'causal', 'increase', 'harmless', 'causing', 'mumps', 'factor', 'thimerosal', 'decreased', 'cause', 'except', 'suppress', 'deficit', 'development', 'rubella', 'immun', 'increased', 'though', 'measles', 'never', 'weak', 'however', 'asd', 'mmr', 'side effect', 'link', 'although', 'despite', 'not', 'true', 'vaccine', 'exposure', 'heightened', 'autistic', 'this', 'vaccin', 'oppose', 'does', 'incidence', 'problem', 'outcome', 'disorder']
q3['name']='Does MMR vaccine lead to autism'

q4={}
q4['train_file_name']='vitaminC_common_cold_final.txt'
q4['test_file_name']='vitaminC_common_cold_test.txt'
q4['gold_rel']='q4_rel.txt'
q4['gold_sup']='q4_sup.txt'
q4['features']=['shorten', 'reduction', 'benefit', 'help prevent cold', 'shorter illness', 'alleviate', 'neutral', 'ward off', 'halve', 'slash', 'reduce', 'prevented', 'manage', 'protective', 'reduc', 'decrease', 'stopped', 'wards off', 'most useful', 'lower', 'most effective', 'effective agent', 'fewer cold', 'less severe', 'fighting off', 'take plenty', ' mg', 'strengthens', 'lessens', 'dry up', 'beneficial', 'fighting', 'still high', 'modest', 'liveli', 'preventive', 'strengthen', 'improve', 'cut their risk', 'stop cancer', 'half', 'may be effective', 'wrong', 'doubt', 'questionable', 'no benefit', 'zinc', 'no effect', 'no difference', 'wont help', 'no evidence', 'but not', 'did not', 'not show', 'doesnt', 'isnt', 'does not', 'little', 'not the best', 'not help', 'will not', 'disagree', 'unsupported', 'or not', 'not worth', 'not necessary', 'nor is', 'he same', 'but only for', 'has not been', 'inconclusive', 'nausea', 'diarrhea', 'dangerous', 'is not an', 'debunk', 'similar symptoms', 'do not recommend', 'do not prove', 'not effective', 'didnt help', 'alas', 'didnt', 'possibly', 'disappointing', 'unjustified', 'small', 'unlikely', 'nor', 'aren', 'this', 'vitamin', 'zinc', 'dose', 'placebo', 'grams', 'intake', 'fruits and vegetables', ' it ', 'antiox', 'supplement', 'consumed', 'mg ', 'reduction', 'myth', 'this', 'cold', 'common cold', 'flu', 'illness', 'sniffles', 'immun', 'sick', 'throat', 'nose', 'effect', 'viral', 'nausea', 'diabetes', 'evidence', 'benefit']
q4['name']='Does Vitamin C reduce common cold'

q5={}
q5['train_file_name']='women_should_Take_HRT_Post_Menopause_final.txt'
q5['test_file_name']='women_should_Take_HRT_Post_Menopause_test.txt'
q5['gold_rel']='q5_rel.txt'
q5['gold_sup']='q5_sup.txt'
q5['features']=['did not find any increase', 'no increase', 'good choice', 'less the increase', 'no evidence', 'flawed', 'counteract', 'benefits', 'not increase', 'alleviate', 'standard', 'good choice', 'neutral', 'ward off', 'slash', 'prevented', 'manage', 'protective', 'reduc', 'decrease', 'wards off', 'most useful', 'lower', 'most effective', 'effective agent', 'fewer cold', 'fighting off', 'take plenty', ' mg', 'strengthens', 'lessens', 'dry up', 'fighting', 'still high', 'liveli', 'preventive', 'strengthen', 'improve', 'cut their risk', 'stop cancer', 'half', 'may be effective', 'outweigh', 'risk', 'small', 'disease', 'alarming', 'deadly', 'not be used', 'stopping', 'inadvisable', 'cause pain', 'bothersome', 'nonhormonal', 'side effects', 'less likely', 'not help', 'no longer', 'more difficult', 'modest increase', 'cancer', 'collapse', 'more health risk', 'ineffective', 'not prevent', 'not beneficial', 'adverse', 'appears', 'not be a good choice', 'unlikely', 'aren', 'none', 'this', 'hrt', 'hormone', 'progestin', 'estrogen', 'dose', 'menopaus', 'cancer', ' it ', 'antiox', 'supplement', 'consumed', 'mg ', 'reduction', 'myth', 'this', 'menopause', 'common cold', 'flu', 'illness', 'sniffles', 'immun', 'sick', 'throat', 'nose', 'effect', 'viral', 'nausea', 'diabetes', 'evidence', 'benefit']
q5['name']='Should women take HRT post menopause'

datasets=[q1,q2,q3,q4,q5]

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
