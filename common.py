# 
# Library of functions shared between markov.py and eval.py
#
# Compiled against Python 2.7
# Author: Stephen Bahr (sbahr@bu.edu)

def read_part_of_speech_file(filename):
  '''Read a part-of-speech file and return a list of (pos, word) pairs.'''
  with open(filename) as pos_file:
    return [line.split() for line in pos_file]


def get_predictions(test_filename, predict_sentence):
  '''Given an HMM, compute predictions for each word in the test data.'''
  sentence = []
  true_poses = []
  for true_pos, word in (read_part_of_speech_file(test_filename)[1:]):
    if word != '<s>':
      sentence.append(word)
      true_poses.append(true_pos)
    else:
      predictions = predict_sentence(sentence)
      for word, pos, true_pos in zip(sentence, predictions, true_poses):
        yield word, pos, true_pos
      yield ('<s>', '<s>', '<s>')
      sentence = []
      true_poses = []
