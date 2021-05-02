import ml.io as io
import ml.nlp as nlp
import ml.classifiers as classifiers
import ml.test as test

n_datapoints = 1000
attribute_pair = ['positive', 'negative']


print('Importing Data')
importdata = io.ImportData()

positive_training = importdata.import_text('./aclImdb/train/pos', n_datapoints)
negative_training = importdata.import_text('./aclImdb/train/neg', n_datapoints)  
labelled_training_data = {'positive': positive_training, 'negative': negative_training}


positive_testing = importdata.import_text('./aclImdb/test/pos', n_datapoints)
negative_testing = importdata.import_text('./aclImdb/test/neg', n_datapoints)  
labelled_testing_data = {'positive': positive_testing, 'negative': negative_testing}


print('Preprocessing Data')
preprocess = nlp.Preprocess()
preprocessed_training_data = preprocess.default_preprocess(labelled_training_data)


print('Feature Extraction')
feature_extraction = nlp.FeatureExtraction()
vocabulary = feature_extraction.build_vocab(preprocessed_training_data)

#max_corpus_frequency = n_datapoints*0.75
#preprocess.add_to_list(preprocess.max_corpus_freq(max_corpus_frequency, vocabulary))
#preprocessed_training_data = preprocess.remove_stopwords_full(preprocessed_training_data)
#vocabulary = feature_extraction.build_vocab(preprocessed_training_data)

feature_set = feature_extraction.frequency_feature_set(preprocessed_training_data, vocabulary)


print('Training')

alpha = 0.01
init_theta = [1,1,1]
print('Learning rate: ',alpha,', Initial theta vector', init_theta)
LogisticRegression = classifiers.LogisticRegression()
theta = LogisticRegression.train_logistic_regression(feature_set, attribute_pair, init_theta, alpha)

print('Trained theta vector: ', theta)


print('Testing')

print('Preprocessing Data')
preprocessed_testing_data = preprocess.default_preprocess(labelled_testing_data)


print('Feature Extraction')
test_feature_set = feature_extraction.frequency_feature_set(preprocessed_testing_data, vocabulary)
test = test.Test()

true_pos, false_pos, true_neg, false_neg = test.test(LogisticRegression.classify, theta, attribute_pair, test_feature_set)
print('False Positive:', false_pos, ', False Negative:', false_neg,', True Positive:', true_pos,', True Negative:', true_neg)

print('Sensitivity: ', round(test.tpr(), 2), ', Specificity:', round(test.tnr(),2) ,', Precision:' , round(test.precision(),2), ', Accuracy:', round(test.accuracy(),2))

print('F1: ', round(test.fb(),2))
