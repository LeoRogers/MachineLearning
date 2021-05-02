# This is a sample script using the ml package to train and test a Logistic
# Regression classifier on a set of IMDB reviews

# Import modules
import ml.io as io
import ml.nlp as nlp
import ml.classifiers as classifiers
import ml.test as test

print('Naive Bayes Classifier')

# Set paramaters
print('Parameters:')
n_datapoints = 500
print(n_datapoints, ' data points')

attribute_pair = ['positive', 'negative']
print('Attribute pair: ', attribute_pair)

max_corpus_proportion = 0.9
max_corpus_frequency = n_datapoints*max_corpus_proportion
print('Remove words more common than ', max_corpus_proportion)

min_corpus_proportion = 0.1
min_corpus_frequency = n_datapoints*min_corpus_proportion
print('Remove words less common than ', min_corpus_proportion)

difference = 0.25
print('Remove words that differ between attributes less than ', difference)


#-------------------------------------------------------------- TRAINING PHASE --------------------------------------------------------------#
print(' ---- TRAINING PHASE ---- ')

#--------------------- IMPORT DATA ------------------------#
print('Importing Data')
importdata = io.ImportData()

# Import training data
positive_training = importdata.import_text('./aclImdb/train/pos', n_datapoints)
negative_training = importdata.import_text('./aclImdb/train/neg', n_datapoints)  
labelled_training_data = {'positive': positive_training, 'negative': negative_training}


#--------------------- PREPROCESSING ----------------------#

print('Preprocessing Data')
preprocess = nlp.Preprocess()
preprocessed_training_data = preprocess.default_preprocess(labelled_training_data)

#--------------------- FEATURE EXTRACTION -----------------#
print('Feature Extraction')

# Extract the vocabulary, which includes word frequencies
feature_extraction = nlp.FeatureExtraction()
vocabulary = feature_extraction.build_vocab(preprocessed_training_data)

# Update stoplist to include words which are too common, too rare, or have frequencies that are too similar accross attribute pair.
preprocess.add_to_list(preprocess.max_corpus_freq(max_corpus_frequency, vocabulary))
preprocess.add_to_list(preprocess.min_corpus_freq(min_corpus_frequency, vocabulary))
preprocess.add_to_list(preprocess.min_attribute_difference(difference, vocabulary, attribute_pair))

# Update the training data and vocabulary with the new stoplist
preprocessed_training_data = preprocess.remove_stopwords_full(preprocessed_training_data)
vocabulary = feature_extraction.build_vocab(preprocessed_training_data)

# Build the feature set
conditional_probabilities = feature_extraction.conditional_probabilities(vocabulary, attribute_pair)

#--------------------- RUN TRAINING --------------------------#
print('Training')
bayes= classifiers.NaiveBayes()
bayes.update_log_likelihood(conditional_probabilities, labelled_training_data, attribute_pair)

#-------------------------------------------------------------- TESTING PHASE --------------------------------------------------------------#
print(' ---- TESTING PHASE ---- ')

#--------------------- IMPORT DATA ------------------------#
print('Importing Data')

# Import testing data
positive_testing = importdata.import_text('./aclImdb/test/pos', n_datapoints)
negative_testing = importdata.import_text('./aclImdb/test/neg', n_datapoints)  
labelled_testing_data = {'positive': positive_testing, 'negative': negative_testing}

#--------------------- PREPROCESSING ----------------------#
print('Preprocessing Data')
preprocessed_testing_data = preprocess.default_preprocess(labelled_testing_data)


#--------------------- RUN TEST --------------------------#
test = test.Test()
true_pos, false_pos, true_neg, false_neg = test.test(bayes.classify, attribute_pair, preprocessed_testing_data)

#------------------- OUTPUT RESULTS -----------------------#
print('False Positive:', false_pos, ', False Negative:', false_neg,', True Positive:', true_pos,', True Negative:', true_neg)
print('Sensitivity: ', round(test.tpr(), 2), ', Specificity:', round(test.tnr(),2) ,', Precision:' , round(test.precision(),2), ', Accuracy:', round(test.accuracy(),2))
print('F1: ', round(test.fb(),2))
