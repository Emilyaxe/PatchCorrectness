# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Python implementation of BLEU and smooth-BLEU.

This module provides a Python implementation of BLEU and smooth-BLEU.
Smooth BLEU is computed following the method outlined in the paper:
Chin-Yew Lin, Franz Josef Och. ORANGE: a method for evaluating automatic
evaluation metrics for machine translation. COLING 2004.
"""

import collections
import math
import numpy as np
import codecs
from nltk import word_tokenize
from rouge import *
from meteor import *
import sys

reload(sys)

sys.setdefaultencoding('utf8')
def _get_ngrams(segment, max_order):
  """Extracts all n-grams upto a given maximum order from an input segment.

  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.

  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
  ngram_counts = collections.Counter()
  for order in range(1, max_order + 1):
    for i in range(0, len(segment) - order + 1):
      ngram = tuple(segment[i:i+order])
      ngram_counts[ngram] += 1
  return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
  """Computes BLEU score of translated segments against one or more references.

  Args:
    reference_corpus: list of lists of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    smooth: Whether or not to apply Lin et al. 2004 smoothing.

  Returns:
    3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
    precisions and brevity penalty.
  """
  matches_by_order = [0] * max_order
  possible_matches_by_order = [0] * max_order
  reference_length = 0
  translation_length = 0
  for (references, translation) in zip(reference_corpus,
                                       translation_corpus):
    reference_length += min(len(r) for r in references)
    translation_length += len(translation)
    #print(translation_length, reference_length)
    if(len(translation) != len(references[0])):
      print(references, translation)
    merged_ref_ngram_counts = collections.Counter()
    for reference in references:
      merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
    translation_ngram_counts = _get_ngrams(translation, max_order)
    overlap = translation_ngram_counts & merged_ref_ngram_counts
    for ngram in overlap:
      matches_by_order[len(ngram)-1] += overlap[ngram]
    for order in range(1, max_order+1):
      possible_matches = len(translation) - order + 1
      if possible_matches > 0:
        possible_matches_by_order[order-1] += possible_matches

  precisions = [0] * max_order
  for i in range(0, max_order):
    if smooth:
      precisions[i] = ((matches_by_order[i] + 1.) /
                       (possible_matches_by_order[i] + 1.))
    else:
      if possible_matches_by_order[i] > 0:
        precisions[i] = (float(matches_by_order[i]) /
                         possible_matches_by_order[i])
      else:
        precisions[i] = 0.0

  if min(precisions) > 0:
    p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
    geo_mean = math.exp(p_log_sum)
  else:
    geo_mean = 0

  ratio = float(translation_length) / reference_length
  if ratio > 1.0:
    bp = 1.
  else:
    bp = math.exp(1 - 1. / ratio)

  bleu = geo_mean * bp

  return (bleu, precisions, bp, ratio, translation_length, reference_length)
def is_number(s):
    try:
        int(s) # for int, long and float
        return True
    except ValueError:
        return False

def tokenize(code):
    fans = []
    ans = word_tokenize(code.strip().lower())
    for x in ans:
        if "/" in x:
            lst = x.split("/")
            for w in lst:
                if w != "":
                    fans.append(w)
        elif "-" in x:
            lst = x.split("-")
            if len(lst) == 2 and is_number(lst[0]) and is_number(lst[1]):
                fans.append(lst[0])
                fans.append("-")
                fans.append(lst[1])
        else:
            fans.append(x)
    return fans
def prepare_results(metric, p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)

def rouges(pre, ref):
	for aggregator in ['Avg', 'Best', 'Individual']:
	    print('Evaluation with {}'.format(aggregator))
	    apply_avg = aggregator == 'Avg'
	    apply_best = aggregator == 'Best'

	    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
	                           max_n=4,
	                           limit_length=True,
	                           length_limit=100,
	                           length_limit_type='words',
	                           apply_avg=apply_avg,
	                           apply_best=apply_best,
	                           alpha=0.5, # Default F1_score
	                           weight_factor=1.2,
	                           stemming=True)

	    all_hypothesis = pre
	    all_references = ref

	    scores = evaluator.get_scores(all_hypothesis, all_references)

	    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
	        if not apply_avg and not apply_best: # value is a type of list as we evaluate each summary vs each reference
	            for hypothesis_id, results_per_ref in enumerate(results):
	                nb_references = len(results_per_ref['p'])
	                for reference_id in range(nb_references):
	                    print('\tHypothesis #{} & Reference #{}: '.format(hypothesis_id, reference_id))
	                    print('\t' + prepare_results(results_per_ref['p'][reference_id], results_per_ref['r'][reference_id], results_per_ref['f'][reference_id]))
	            print()
	        else:
	            print(prepare_results(metric, results['p'], results['r'], results['f']))
	    print()

if __name__ == "__main__":
  f = codecs.open("nl.txt", "r", encoding='utf-8')
  lines = f.readlines()
  f = codecs.open("outval.txt", "r", encoding='utf-8')
  lines1 = f.readlines()
  tmp = []
  for x in lines:
    tmp.append([word_tokenize(x.strip().lower())])
  lines1 = [word_tokenize(x.strip().lower()) for x in lines1]
  print(compute_bleu(tmp, lines1))
  print(len(tmp), len(lines1))
  ref = {}
  ge = {}
  for i, x in enumerate(lines1):
  	ge[i] = [" ".join(x)] 
  for i, x in enumerate(tmp):
  	if i >= len(lines1):
  		continue
  	ref[i] = [" ".join(x[0])] 
  r = Rouge()
  m = Meteor()
  print(r.compute_score(ref, ge))#rouges(lines1, tmp[:len(lines1)])
  print(m.compute_score(ref, ge)[0])
