# 
# Evaluate the performance of the markov and baseline models on test data.
# 
# Usage:
#   eval.py training_file devtest_file
#
# Compiled against Python 2.7
# Author: Stephen Bahr (sbahr@bu.edu)

import math
import sys

import markov
import common

class Ratio:
  def __init__(self):
    self.numerator = 0
    self.denominator = 0

  def observe(self, predicate):
    self.denominator += 1
    if predicate:
      self.numerator += 1

  def value(self):
    if not self.denominator:
      return float('nan')
    else:
      return float(self.numerator) / self.denominator

  def __str__(self):
    if not self.denominator:
      return '%21s' % 'NaN'
    return '%5d/%5d = %6.2f%%' % (self.numerator, self.denominator,
        100.0 * self.numerator / self.denominator)


def compute_score(predictions, known_words):
  '''Compute the score for a set of predictions.

  Args:
    predictions: a list of predicted-part-of-speech, word, true-pos triples.
    known_words: a set of words that appear in the training data.

  Returns:
    number of correct predictions, total number of predictions
    Does not count sentence-end tokens.
  '''
  accuracy = Ratio()
  unknown_accuracy = Ratio()
  for word, pos, true_pos in predictions:
    if pos == '<s>':
      continue
    accuracy.observe(true_pos == pos)
    if word not in known_words:
      unknown_accuracy.observe(true_pos == pos)
  return unknown_accuracy, accuracy


def main():
  if len(sys.argv) != 3:
    print 'Usage: %s training_filename test_filename' % sys.argv[0]
    return 1
  train_filename, test_filename = sys.argv[1:]

  training_data = hw5_common.read_part_of_speech_file(train_filename)
  known_words = set(word for pos, word in training_data)
  print >> sys.stderr, 'Training baseline model'
  baseline_model = hw5.BaselineModel(training_data)
  print >> sys.stderr, 'Evaluating baseline model'
  baseline_unknown_accuracy, baseline_accuracy = compute_score(
      hw5_common.get_predictions(
        test_filename, baseline_model.predict_sentence), known_words)

  print >> sys.stderr, 'Training hmm model'
  hmm_model = hw5.HiddenMarkovModel.train(training_data)
  print >> sys.stderr, 'Evaluating hmm model'
  hmm_unknown_accuracy, hmm_accuracy = compute_score(
      hw5_common.get_predictions(
        test_filename, hmm_model.predict_sentence), known_words)

  print '%s Baseline accuracy' % baseline_accuracy
  print '%s Baseline accuracy on unknown words' % baseline_unknown_accuracy
  print '%s HMM accuracy' % hmm_accuracy
  print '%s HMM accuracy on unknown words' % hmm_unknown_accuracy

  print baseline_accuracy.value() * 50

  print 'Score for Part III: %d/50' % (
      math.ceil(max(baseline_accuracy.value(), hmm_accuracy.value()) * 50))

  print 'Score for Part IV-unknown words: %d/20' % (
      max(0, math.ceil((hmm_unknown_accuracy.value() - 0.6) * 50)))


if __name__ == '__main__':
  main()
