from __future__ import annotations

import re

import nltk
from nltk import word_tokenize
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer 
from nltk.stem import SnowballStemmer 
from nltk.stem import PorterStemmer 
from nltk.corpus import wordnet

import numpy as np
import pandas as pd
from IPython.display import display

downloaded = nltk.download('stopwords')
downloaded = nltk.download('averaged_perceptron_tagger')
downloaded = nltk.download('punkt')
downloaded = nltk.download('wordnet')

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stop_words
from sklearn.feature_selection import mutual_info_classif

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import logging

class NLTKNormalizer(BaseEstimator, TransformerMixin):
	""" Class to normalize tokens using NLTK Stemmers and Lemmatizers """
	def __init__(self, 
				 normalizer:str|None = None, # the normalizer to use None, 'PorterStemmer', 'SnowballStemmer', 'WordNetLemmatizer'
				 lowercase:bool = True, # whether to lowercase the tokens before normalizing
				 stop_word_list:str|None = None, # the stop word list to use - None, 'sklearn', 'nltk', 'both'
				 extra_stop_words:list = [] # extra stop words to add to the stop word list
				 ):
		self.normalizer_type = normalizer
		if self.normalizer_type == 'PorterStemmer':
			self.normalizer = PorterStemmer()
		elif self.normalizer_type == 'SnowballStemmer':
			self.normalizer = SnowballStemmer('english')
		elif self.normalizer_type == 'WordNetLemmatizer':
			self.normalizer = WordNetLemmatizer()
		elif self.normalizer_type is None:
			self.normalizer = None
		else:
			logging.warning(f'Unknown normalizer_type - ignoring, no normalizer will be used')
			self.normalizer_type = None
			self.normalizer = None

		self.lowercase = lowercase

		self.stop_word_lists = get_stopword_lists()

		self.stop_word_list = stop_word_list
		if self.stop_word_list in ['sklearn', 'nltk', 'both']:
			self.stop_words = self.stop_word_lists[self.stop_word_list]
		else:
			self.stop_word_list = None
			self.stop_words = []

		self.extra_stop_words = extra_stop_words

		if len(extra_stop_words) > 0:
			self.stop_words += extra_stop_words

	def fit(self, X, y=None):
		return self

	def transform(self, 
				  X):
		return [self.normalize(tokens) for tokens in X]

	def normalize(self,
				  tokens: list[str] # the list of tokens to normalize
				  ) -> list[str]: # the normalized tokens

		# if using a normalizer then iterate through tokens and return the normalized tokens ...
		if self.normalizer_type == 'PorterStemmer':
			tokens = [self.normalizer.stem(t) for t in tokens]
		elif self.normalizer_type == 'SnowballStemmer':
			tokens = [self.normalizer.stem(t) for t in tokens]
		elif self.normalizer_type == 'WordNetLemmatizer':
			# NLTK's lemmatiser needs parts of speech, otherwise assumes everything is a noun
			pos_tokens = nltk.pos_tag(tokens)
			lemmatised_tokens = []
			for token in pos_tokens:
				# NLTK's lemmatiser needs specific values for pos tags - this rewrites them ...
				# default to noun
				tag = wordnet.NOUN
				if token[1].startswith('J'):
					tag = wordnet.ADJ
				elif token[1].startswith('V'):
					tag = wordnet.VERB
				elif token[1].startswith('R'):
					tag = wordnet.ADV
				lemmatised_tokens.append(self.lemmatize(token[0],tag))
			tokens = lemmatised_tokens
		else:
			# no normaliser so just return tokens
			tokens = tokens

		# lowercase the tokens if required
		if self.lowercase:
			tokens = [t.lower() for t in tokens]
		
		# removing stop words if required
		if self.stop_words is not None:
			tokens = [t for t in tokens if t not in self.stop_words]
		
		return tokens

class NLTKTokenizer(BaseEstimator, TransformerMixin):
	""" Class to tokenize text using NLTK tokenizers """
	def __init__(self, tokenizer:str = 'word_tokenize'):
		if tokenizer in ['sklearn', 'word_tokenize', 'wordpunct', 'nopunct']:
			self.tokenizer = tokenizer
		else:
			logging.warning(f'Unknown tokenizer - defaulting to sklearn')
			self.tokenizer = 'sklearn'

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		tokens = []
		for text in X:
			# add more tokenizers here if needed ...
			if self.tokenizer == 'sklearn':
				tokenizer = RegexpTokenizer(r"(?u)\b\w\w+\b") # this is copied straight from sklearn source
				tokens.append(tokenizer.tokenize(text))
			elif self.tokenizer == 'wordpunct':
				tokens.append(wordpunct_tokenize(text))
			elif self.tokenizer == 'nopunct':
				tokenizer = RegexpTokenizer(r'\w+')
				tokens.append(tokenizer.tokenize(text))
			else:
				tokens.append(word_tokenize(text))
		return tokens
	
def pass_tokens(tokens):
	return tokens

def get_stopword_lists():
	nltk_stop_words = nltk.corpus.stopwords.words('english')

	stop_word_lists = {
		'sklearn': list(sklearn_stop_words),
		'nltk': nltk_stop_words,
		'both': list(set(nltk_stop_words).union(set(sklearn_stop_words))),
	}
	return stop_word_lists


def get_preview(docs, targets, target_names, doc_id, max_len=0):
	""" Get a nice preview of a document """
	preview = ''
	if max_len < 1:
		preview += 'Label\n'
		preview += '=====\n'
	else:
		preview += str(doc_id)
		preview += '\t'
	preview += target_names[targets[doc_id]]
	if max_len < 1:
		preview += '\n\nFull Text\n'
		preview += '=========\n'
		if docs[doc_id].strip() == '':
			preview += 'No text in this document.'
		else:
			preview += docs[doc_id]
		preview += '\n'
	else:
		excerpt = get_excerpt(docs[doc_id], max_len)
		preview += '\t' + excerpt
	return preview

# regular expression to combine whitespace
_RE_COMBINE_WHITESPACE = re.compile(r"\s+")

def get_excerpt(text, max_len):
	""" Get an excerpt of a document """
	excerpt = _RE_COMBINE_WHITESPACE.sub(' ',text[0:max_len])
	if max_len < len(text):
		excerpt += '...'
	return excerpt.strip()


def nb_binary_display_features(pipeline, label_names, features_to_show=20):
	vect = pipeline.named_steps['vectorizer']
	clf = pipeline.named_steps['classifier']
	feature_names = vect.get_feature_names_out()
	logodds=clf.feature_log_prob_[1]-clf.feature_log_prob_[0]

	# if selector in pipeline
	if 'selector' in pipeline.named_steps:
		# Get SelectKBest feature scores
		features = pipeline.named_steps['selector']
		# get top k feature indices
		cols = features.get_support(indices=True)
		# get corresponding feature scores
		# populate array of length feature_names with nulls
		# top_k_feature_scores = [None] * len(feature_names)
		# for i in feature_names:
		# 	if i in cols:
		# 		top_k_feature_scores[i] = features.scores_[i]
		top_k_feature_scores = [features.scores_[i] for i in cols if i in cols]
		feature_names = [feature_names[i] for i in cols]

	df = pd.DataFrame({
		'Feature': feature_names,
		'Log-Odds': logodds,
	})

	if 'selector' in pipeline.named_steps:
		# if scoring func is mi 
		if pipeline.named_steps['selector'].score_func == mutual_info_classif:
			score_column_name = 'MI Score'
		else:
			score_column_name = 'Feature Score'

		df[score_column_name] = top_k_feature_scores

	if 'selector' in pipeline.named_steps:
		print('Top features by information gain')
		print('================================')
		sorted_df = df.sort_values([score_column_name], ascending=False).head(features_to_show)
		display(sorted_df)

	print("Features most indicative of", label_names[0])
	print('============================' + '='*len(label_names[0]))

	sorted_df = df.sort_values('Log-Odds', ascending=True).head(features_to_show)
	display(sorted_df)

	print("Features most indicative of", label_names[1])
	print('============================' + '='*len(label_names[1]))

	sorted_df = df.sort_values('Log-Odds', ascending=False).head(features_to_show)
	display(sorted_df)

def get_feature_frequencies(pipeline, text):
	preprocessor = Pipeline(pipeline.steps[:-1])
	frequency = preprocessor.transform([text]).toarray()[0].T
	feature_names = preprocessor.named_steps['vectorizer'].get_feature_names_out()
	
	if 'selector' in pipeline.named_steps:
		cols = pipeline.named_steps['selector'].get_support(indices=True)
		feature_names = [feature_names[i] for i in cols]

	df = pd.DataFrame(frequency, index=feature_names, columns=['frequency'])
	df = df[df['frequency'] > 0].sort_values('frequency', ascending=False)
	if len(df) < 1:
		return 'No features extracted from this document.'
	else:
		return df


class TokenVectorizer(BaseEstimator, TransformerMixin):
	def __init__(self, vectorizer_type='tfidf', 
            min_df = 0.0, 
            max_df = 1.0, 
            max_features = None,
            ngram_range = (1, 1),
            encoding = 'utf-8', 
            decode_error = 'ignore'
			  ):
		self.vectorizer_type = vectorizer_type
		self.vectorizer = None
		self.tokenizer = pass_tokens
		self.lowercase = False
		self.stop_words = None
		self.token_pattern = None
		self.min_df = min_df
		self.max_df = max_df
		self.max_features = max_features
		self.ngram_range = ngram_range
		self.encoding = encoding
		self.decode_error = decode_error

	def fit(self, X, y=None):
		if self.vectorizer_type == 'tfidf':
			self.vectorizer = TfidfVectorizer(tokenizer=self.tokenizer, lowercase=self.lowercase, stop_words=self.stop_words, token_pattern=self.token_pattern, min_df=self.min_df, max_df=self.max_df, max_features=self.max_features, ngram_range=self.ngram_range, encoding=self.encoding, decode_error=self.decode_error)
		elif self.vectorizer_type == 'count':
			self.vectorizer = CountVectorizer(tokenizer=self.tokenizer, lowercase=self.lowercase, stop_words=self.stop_words, token_pattern=self.token_pattern, min_df=self.min_df, max_df=self.max_df, max_features=self.max_features, ngram_range=self.ngram_range, encoding=self.encoding, decode_error=self.decode_error)
		else:
			raise ValueError("Invalid vectorizer_type. Use 'tfidf' or 'count'.")
		return self.vectorizer.fit(X, y)

	def transform(self, X):
		return self.vectorizer.transform(X)
	
	def get_feature_names_out(self, input_features=None):
		return self.vectorizer.get_feature_names_out(input_features)
	

__all__ = ['NLTKNormalizer', 'NLTKTokenizer', 'TokenVectorizer', 'pass_tokens', 'get_stopword_lists', 'get_excerpt', 'get_preview', 'nb_binary_display_features', 'get_feature_frequencies']
