"""A module with classes specific to Natural Language Processing

Preprocess: A class for preprocessing text
	Attributes: 
		punctuation_list: An incomplete list of punctuation symbols
		stoplist: A list of common words 
		suffixes: An incomplete list of English language suffixes
	Methods:
		__init__
		add_to_list
		create_stoplist
		remove_stopwords_full
		default_preprocess
		remove_punctuation
		remove_stopwords
		stemming
		tokenize

FeatureExtraction: A class for extracting features for use in NLP classification
	Attributes:
		None
	Methods:
		__init__
		build_vocab
		assign_freqs
		extract_feature_set
		extract_frequency
		normalise_feature_set
"""

import copy
class Preprocess:
    """Class for preprocessing text"""
    
    def __init__(self):
        """Initialises the preprocessing class with three lists strings commonly used in preprocessing.

        Stopword list taken from https://www.textfixer.com/tutorials/common-english-words.txt, except with the suffixes in self.suffixes removed
        """
        
        self.punctuation_list = [';', '-', '/', '\\', '"', "'", '.', ',', '?', '!', '(', ')', ':', '<br />', '<br >']
        self.stoplist = ['a', 'able', 'about', 'acros', 'aft', 'all', 'almost', 'also', 'am', 'among', 'an', 'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'but', 'by', 'can', 'cannot', 'could', 'dear', 'did', 'do', 'doe', 'eith', 'else', 'ev', 'every', 'for', 'from', 'get', 'got', 'had', 'has', 'have', 'he', 'h', 'her', 'him', 'his', 'how', 'howev', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'just', 'least', 'let', 'like', 'likely', 'may', 'me', 'might', 'most', 'must', 'my', 'neith', 'no', 'nor', 'not', 'of', 'off', 'often', 'on', 'only', 'or', 'oth', 'our', 'own', 'rath', 'said', 'say', 'say', 'she', 'should', 'since', 'so', 'some', 'than', 'that', 'the', 'their', 'them', 'then', 'there', 'these', 'they', 'thi', 'tis', 'to', 'too', 'twa', 'us', 'want', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'would', 'yet', 'you', 'your']
        self.suffixes = ['ed ', 'ing ', 'er ', 'ers ', 'ings ', 's ']
        return

    def add_to_list(self, new_list, addtolist = 'stoplist'):
        """Extends a specified instance attribute with a user-defined list."""
        if addtolist == 'stoplist':
            self.stoplist += new_list
        elif addtolist == 'punctuation':
            self.punctuation_list += new_list
        elif addtolist == 'suffixes':
            self.suffixes += new_list
        return
    
    def max_corpus_freq(self, max_corpus_frequency, vocabulary):
        """Returns a list of words that appear more frequently than max_corpus_frequency"""
        new_list = []
        for word in vocabulary:
            if vocabulary[word]['total'] > max_corpus_frequency:
                if word not in self.stoplist:
                    new_list.append(word)
        return new_list

    def min_corpus_freq(self, min_corpus_frequency, vocabulary):
        """Returns a list of words that appear less than min_corpus_frequency"""
        new_list = []
        for word in vocabulary:
            if vocabulary[word]['total'] < min_corpus_frequency:
                if word not in self.stoplist:
                    new_list.append(word)
        return new_list

    def min_attribute_difference(self, difference, vocabulary, attribute_pair):
        """Returns a list of words for which difference in the frequency between the two attributes as a proportion of the total frequency is less than the variable 'difference'"""
        new_list = []
        for word in vocabulary:
            if abs(vocabulary[word][attribute_pair[0]] - vocabulary[word][attribute_pair[1]])/vocabulary[word]['total'] < difference:
                if word not in self.stoplist:
                    new_list.append(word)
        return new_list

    def remove_stopwords_full(self, data):
        """Removes all instances of the stop words from the full dataset"""
        for attribute in data:
            dataset = []
            for datapoint in data[attribute]:
                new_datapoint = self.remove_stopwords(datapoint) 
                dataset.append(new_datapoint)
            data.update({attribute: dataset})
        return data
    
    def default_preprocess(self, attributed_data):
        """Preprocesses a attributed data set using the recommended preprocessing functions

        For each element in the data set, this function lowers the case, removes the puncutation, removes suffixes, removes stopwords and tokenizes.
        A dictionary of attributed preprocessed data is returned."""
        preprocessed_training_data = {}
        
        for attribute in attributed_data:
            dataset = copy.copy(attributed_data[attribute])
            preprocessed_dataset = []
            for data in dataset:
                preprocessed_data = data.lower()
                preprocessed_data = self.remove_punctuation(preprocessed_data)
                preprocessed_data = self.stemming(preprocessed_data)
                preprocessed_data = self.tokenize(preprocessed_data)
                preprocessed_data = self.remove_stopwords(preprocessed_data)
                
                preprocessed_dataset.append(preprocessed_data)
            
            preprocessed_training_data.update({attribute: preprocessed_dataset})
        return preprocessed_training_data

    
    def remove_punctuation(self, data):
        """Takes a string and removes punctuation"""

        for punct in self.punctuation_list:
            if punct in data:
                data = data.replace(punct, '')
        return data
    
    def remove_stopwords(self, data):
        """Removes any instances of a stopword from a list"""

        d = [x for x in data if x not in self.stoplist]
            
        return d
    
    def stemming(self, data):
        """Removes suffixes from words in a string"""

        for suff in self.suffixes:
            if suff in data and len(data) > 3:
                data = data.replace(suff, ' ')
        return data
    
    def tokenize(self, data):
        """Turns a string into a list of words"""

        data = eval('["'+data.replace(' ', '", "')+'"]')
        return data

class FeatureExtraction:
    """A class for extracting features used in NLP classification"""
    def __init__(self):
        return
    
    def build_vocab(self, dataset):
        """Takes a tokenized attributed dataset and returns a attributed vocabulary"""
        vocab = {}
        attribute_list = []
        for attribute in dataset:
            attribute_list.append(attribute)
        attribute_list.append('total')
        for attribute in dataset:
            
            for text in dataset[attribute]:
                for word in text:
                    if word not in vocab:                     
                        vocab.update({word: {l:0 for l in attribute_list} })
                        vocab[word]['total'] += 1
                        vocab[word][attribute] += 1
                    else:
                        
                        vocab[word][attribute] += 1
                        vocab[word]['total'] += 1
        return vocab
    def assign_freqs(self, data, vocab):
        """Takes an unattributed preprocessed datapoint and a attributed vocabulary and computes the frequency feature.

        The frequency feature is a dictionary relating each attribute to the sum of the frequencies for that attribute of each unique word in the datapoint"""
        attribute_list = list(vocab.items())[0][1]
        attribute_list = list(attribute_list.keys())
        attribute_list.remove('total')
        
        used_words = []
        freqs = {l:0 for l in attribute_list}
        for word in data:
            if word in vocab:
                if word not in used_words:
                    used_words.append(word)
                    for attribute in attribute_list:
                        freqs[attribute]+= vocab[word][attribute]
        
        return freqs
    
    def _extract_frequency_feature(self, datapoint, vocabulary):
        """Returns the feature vector starting with 1 followed by the frequency features for each attribute."""
        freqs = self.assign_freqs(datapoint, vocabulary)
        feature = [1]+[f for f in freqs.values()]
        return feature
    
    def frequency_feature_set(self, dataset, vocabulary):
        """Returns the 'feature set' relating to the word frequencies in each attribute.

        The 'feature set' is a dictionary in which the attribute of the attributed data is given as a key, returning a list of feature vectors for each attributed datapoint
        """
        
        feature_set = {}
        for attribute in dataset:
            feature_array = []
            for datapoint in dataset[attribute]:
                feature = self._extract_frequency_feature(datapoint, vocabulary)
                feature_array.append(feature)
            feature_set.update({attribute: feature_array})
        return feature_set
    
    def normalise_feature_set(self, feature_set):
        """Normalises the feature set.

        This currently makes the results significantly worse.
        """
        normalised_feature_set = {}
        for attribute in feature_set:
            feature_array = feature_set[attribute]
            norm_feature = [[1 for i in range(len(feature_set)+1)] for feature in feature_array]
            for i in range(1, len(feature_set)+1):
                av = sum(feature_array[:][i])/len(feature_array)
                mx = max(feature_array[:][i])
                mn = min(feature_array[:][i])
                for j, feature in enumerate(feature_array):
                    
                    norm_feature[j][i] = (feature[i] - mn)/(mx - mn)

            normalised_feature_set.update({attribute: norm_feature})
        return normalised_feature_set

    def conditional_probabilities(self, vocabulary, attribute_list):
        """Returns a vocabulary dictionary but with conditional probabilities instead of frequencies

        Uses Laplacian smoothing.
        """
        v = len(vocabulary)
        conditional_probability_vocab = {}
        for word in vocabulary:
            conditional_probability = {}
            for attribute in attribute_list:
                conditional_probability.update( { attribute: (vocabulary[word][attribute]+1)/(vocabulary[word]['total'] + v) } )
                
            conditional_probability_vocab.update({word: conditional_probability})
            
        return conditional_probability_vocab
            
            

