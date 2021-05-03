# This is a sample script using the ml package to train and test a Logistic
# Regression classifier on a set of IMDB reviews

# Import modules
import ml.io as io
import ml.nlp as nlp
import ml.classifiers as classifiers
import ml.test as test



# Set paramaters
print('Parameters:')
n_datapoints = 12500 
print(n_datapoints, ' data points per attribute')

attribute_pair = ['positive', 'negative']
print('Attribute pair: ', attribute_pair)

automatic_stoplist = False
print('Automatically create a stoplist: ', automatic_stoplist)

if automatic_stoplist:
    max_corpus_proportion = 0.95
    max_corpus_frequency = n_datapoints*max_corpus_proportion
    print('Remove words more common than ', max_corpus_proportion)

    min_corpus_proportion = 0.05
    min_corpus_frequency = n_datapoints*min_corpus_proportion
    print('Remove words less common than ', min_corpus_proportion)

    difference = 0.05
    print('Remove words that differ between attributes less than ', difference)

#--------------------------------------------------------------- LOGISTIC REGRESSION ---------------------------------------------------------------#

#------------------------------------------ TRAINING PHASE ------------------------------------------#
print(' -------- LOGISTIC REGRESSION ---------- ')
learning_rate = 0.01 
print('Learning rate: ', learning_rate)
print(' ---- TRAINING PHASE ---- ')

#--------------------- IMPORT DATA 
print('Importing Data')
importdata = io.ImportData()

# Import training data
positive_training = importdata.import_text('./aclImdb/train/pos', n_datapoints)
negative_training = importdata.import_text('./aclImdb/train/neg', n_datapoints)  
labelled_training_data = {'positive': positive_training, 'negative': negative_training}


#--------------------- PREPROCESSING 

print('Preprocessing Data')
preprocess = nlp.Preprocess()
preprocessed_training_data = preprocess.default_preprocess(labelled_training_data)

#--------------------- FEATURE EXTRACTION
print('Feature Extraction')

# Extract the vocabulary, which includes word frequencies
feature_extraction = nlp.FeatureExtraction()
vocabulary = feature_extraction.build_vocab(preprocessed_training_data)

if automatic_stoplist:

    # Update stoplist to include words which are too common, too rare, or have frequencies that are too similar accross attribute pair.
    preprocess.add_to_list(preprocess.max_corpus_freq(max_corpus_frequency, vocabulary))
    preprocess.add_to_list(preprocess.min_corpus_freq(min_corpus_frequency, vocabulary))
    preprocess.add_to_list(preprocess.min_attribute_difference(difference, vocabulary, attribute_pair))

    # Update the training data and vocabulary with the new stoplist
    preprocessed_training_data = preprocess.remove_stopwords_full(preprocessed_training_data)
    vocabulary = feature_extraction.build_vocab(preprocessed_training_data)

# Build the feature set
feature_set = feature_extraction.frequency_feature_set(preprocessed_training_data, vocabulary)

#--------------------- RUN TRAINING 
print('Training')
LogisticRegression = classifiers.LogisticRegression()
theta = LogisticRegression.train(feature_set, attribute_pair, learning_rate)

print('Trained theta vector: ', theta)

#------------------------------------------ TESTING PHASE ------------------------------------------#
print(' ---- TESTING PHASE ---- ')

#--------------------- IMPORT DATA 
print('Importing Data')

# Import testing data
positive_testing = importdata.import_text('./aclImdb/test/pos', n_datapoints)
negative_testing = importdata.import_text('./aclImdb/test/neg', n_datapoints)  
labelled_testing_data = {'positive': positive_testing, 'negative': negative_testing}

#--------------------- PREPROCESSING 
print('Preprocessing Data')
preprocessed_testing_data = preprocess.default_preprocess(labelled_testing_data)

#--------------------- FEATURE EXTRACTION 
print('Feature Extraction')
test_feature_set = feature_extraction.frequency_feature_set(preprocessed_testing_data, vocabulary)

#--------------------- RUN TEST 
lr_test = test.Test()
lr_true_pos, lr_false_pos, lr_true_neg, lr_false_neg = lr_test.test(LogisticRegression.classify, attribute_pair, test_feature_set)

#------------------- OUTPUT RESULTS
print('False Positive:', lr_false_pos, ', False Negative:', lr_false_neg,', True Positive:', lr_true_pos,', True Negative:', lr_true_neg)
print('Sensitivity: ', round(lr_test.tpr(), 2), ', Specificity:', round(lr_test.tnr(),2) ,', Precision:' , round(lr_test.precision(),2), ', Accuracy:', round(lr_test.accuracy(),2))
print('F1: ', round(lr_test.fb(),2))


#--------------------------------------------------------------- NAIVE BAYES ---------------------------------------------------------------#

print(' -------- NAIVE BAYES ---------- ')

#------------------------------------------ TRAINING PHASE ------------------------------------------#
#--------------------- FEATURE EXTRACTION
conditional_probabilities = feature_extraction.conditional_probabilities(vocabulary, attribute_pair)

#--------------------- RUN TRAINING
bayes = classifiers.NaiveBayes()
bayes.update_log_likelihood(conditional_probabilities, labelled_training_data, attribute_pair)

#------------------------------------------ TESTING PHASE ------------------------------------------#
#--------------------- RUN TEST 
b_test = test.Test()
b_true_pos, b_false_pos, b_true_neg, b_false_neg = b_test.test(bayes.classify, attribute_pair, preprocessed_testing_data)

#------------------- OUTPUT RESULTS
print('False Positive:', b_false_pos, ', False Negative:', b_false_neg,', True Positive:', b_true_pos,', True Negative:', b_true_neg)
print('Sensitivity: ', round(b_test.tpr(), 2), ', Specificity:', round(b_test.tnr(),2) ,', Precision:' , round(b_test.precision(),2), ', Accuracy:', round(b_test.accuracy(),2))
print('F1: ', round(b_test.fb(),2))

