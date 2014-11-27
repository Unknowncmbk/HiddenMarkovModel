# 
# Train and predict using a Hidden Markov Model part-of-speech tagger.
# 
# Usage:
#   markov.py training_file test_file
#
# Compiled against Python 2.7
# Author: Stephen Bahr (sbahr@bu.edu)

import optparse
import collections
import math

import common

# Smoothing methods
NO_SMOOTHING = 'None'  # Return 0 for the probability of unseen events
ADD_ONE_SMOOTHING = 'AddOne'  # Add a count of 1 for every possible event.
# *** Add additional smoothing methods here ***

# Unknown word handling methods
PREDICT_ZERO = 'None'  # Return 0 for the probability of unseen words

# If p is the most common part of speech in the training data,
# Pr(unknown word | p) = 1; Pr(unknown word | <anything else>) = 0
PREDICT_MOST_COMMON_PART_OF_SPEECH = 'MostCommonPos'
# *** Add additional unknown-word-handling methods here ***


class BaselineModel:
  '''A baseline part-of-speech tagger.

  Fields:
    dictionary: map from a word to the most common part-of-speech for that word.
    default: the most common overall part of speech.
  '''
  def __init__(self, training_data):
    '''Train a baseline most-common-part-of-speech classifier.

    Args:
      training_data: a list of pos, word pairs:
    '''

    set_input = {}
    set_output = {}
    for tup in training_data:
      pos = tup[0]
      word = tup[1]

      if pos in set_input:
        set_input[pos] = set_input[pos] + 1
      else:
        set_input[pos] = 1

      if word in set_output:
        set_output[word] = set_output[word] + [pos]
      else:
        set_output = [pos]

    self.dictionary = {}
    for o in set_output:
      freq = {}
      for POS in list(set(set_output[o])):
        freq[POS] = set_output[o].count(POS)
      self.dictionary[o] = max(freq, key=freq.get)

    self.default = max(set_input, key=set_input.get)

  def predict_sentence(self, sentence):
    return [self.dictionary.get(word, self.default) for word in sentence]


class HiddenMarkovModel:
  def __init__(self, order, emission, transition, parts_of_speech, known):
    # Order 0 -> unigram model, order 1 -> bigram, order 2 -> trigram, etc.
    self.order = order
    # Emission probabilities, a map from (pos, word) to Pr(word|pos)
    self.emission = emission
    # Transition probabilities
    # For a bigram model, a map from (pos0, pos1) to Pr(pos1|pos0)
    self.transition = transition
    # A set of parts of speech known by the model
    self.parts_of_speech = parts_of_speech
    # A set of words known by the model
    self.known_words = known

  def predict_sentence(self, sentence):
    return self.find_best_path(self.compute_lattice(sentence))

  def compute_lattice(self, sentence):
    """Compute the Viterbi lattice for an example sentence.

    Args:
      sentence: a list of words, not including the <s> tokens on either end.

    Returns:
      FOR ORDER 1 Markov models:
      lattice: [{pos: (score, prev_pos)}]
        That is, lattice[i][pos] = (score, prev_pos) where score is the
        log probability of the most likely pos/word sequence ending in word i
        having part-of-speech pos, and prev_pos is the part-of-speech of word i-1
        in that sequence.

        i=0 is the <s> token before the sentence
        i=1 is the first word of the sentence.
        len(lattice) = len(sentence) + 2.

      FOR ORDER 2 Markov models: ??? (extra credit)
    """
    sent = ['<s>'] + sentence + ['<s>']
    lattice = []
    # Construct the length of the lattice
    for i in range(0, len(sentence) + 2):
      if i == 0 or i == len(sentence) + 1:
        d = {'<s>': (self.emission[('<s>', '<s>')], None)}
        lattice.append(d)
      else:
        lattice.append({})

    prev = None
    prev_pos = None
    most_common_pos = None
    #Populate the lattice
    for x in range(0, len(sent)):

      word = sent[x]

      if word in self.known_words or word == '<s>':

        for pos in self.parts_of_speech:
          if (pos, word) in self.emission:
            probability = self.emission[(pos, word)]

            if (prev_pos, pos) in self.transition:
              probability = self.transition[(prev_pos, pos)] + probability

            if prev is not None:
              probability = prev[0] + probability

            lattice[x][pos] = (probability, prev_pos)

        prev_pos = max(lattice[x], key=lambda k:lattice[x][k][0])
        prev = lattice[x][prev_pos]
      else:
        prev = (0.0, prev_pos)

        for pos in self.parts_of_speech:
          lattice[x][pos] = ( float('-inf'), prev_pos )
          
        if most_common_pos is None:
          counter = {}
          for key, val in self.emission.keys():
            if key in counter:
              counter[key] = counter[key] + 1
            else:
              counter[key] = 1
          most_common_pos = max(counter, key=counter.get)
        prev_pos = most_common_pos

    return lattice


  @staticmethod
  def train(training_data,
      smoothing=NO_SMOOTHING,
      unknown_handling=PREDICT_MOST_COMMON_PART_OF_SPEECH,
      order=1):
    '''Train a hidden-Markov-model part-of-speech tagger.

    Args:
      training_data: A list of pairs of a word and a part-of-speech.
      smoothing: The method to use for smoothing probabilities.
         Must be one of the _SMOOTHING constants above.
      unknown_handling: The method to use for handling unknown words.
         Must be one of the PREDICT_ constants above.
      order: The Markov order; the number of previous parts of speech to
        condition on in the transition probabilities.  A bigram model is order 1.

    Returns:
      A HiddenMarkovModel instance.
    '''

    parts_of_speech = HiddenMarkovModel.get_unique_pos(training_data)
    known = HiddenMarkovModel.get_unique_known_words(training_data)
    emission = HiddenMarkovModel.get_emissions(training_data, smoothing, known, parts_of_speech)
    transition = HiddenMarkovModel.get_transitions(training_data, smoothing, known, parts_of_speech)

    return HiddenMarkovModel(order, emission, transition, parts_of_speech, known)

  @staticmethod
  def find_best_path(lattice):
    """Return the best path backwards through a complete Viterbi lattice.

    Args:
      FOR ORDER 1 MARKOV MODELS (bigram):
        lattice: [{pos: (score, prev_pos)}].  See compute_lattice for details.

    Returns:
      FOR ORDER 1 MARKOV MODELS (bigram):
        A list of parts of speech.  Does not include the <s> tokens surrounding
        the sentence, so the length of the return value is 2 less than the length
        of the lattice.
    """
    order = []
    for x in range(0, len(lattice)):
      if x > 0 and x < len(lattice) - 1:
        pos = None
        maximum = None
        for key in lattice[x]:
          if maximum is None or lattice[x][key] > maximum:
            maximum = lattice[x][key]
            pos = key
        order.append(pos)

    return order

  @staticmethod
  def get_unique_pos(training_data):
    parts_of_speech = list()

    for tup in training_data:
      if not (tup[0] == '<s>'):
        parts_of_speech.append(tup[0])
    return parts_of_speech

  @staticmethod
  def get_unique_known_words(training_data):
    known = list()

    for tup in training_data:
      if not (tup[0] == '<s>'):
        known.append(tup[1])
    return known

  """ Emission probabilities, a map from (pos, word) to Pr(word|pos) """
  @staticmethod
  def get_emissions(training_data, smoothing, known_words, parts_of_speech):
    if smoothing == ADD_ONE_SMOOTHING:
      a = set(known_words)
      b = set(parts_of_speech)
      for known_word in a:
        for pos in b:
          tup = (pos, known_word)
          training_data = (tup, ) + training_data

    emissions = {}
    pos_count = {}

    # Count the occurences of (pos, word) in training_data
    for x in range(0, len(training_data)):
      tup = (training_data[x][0], training_data[x][1])
      if tup in emissions:
        emissions[tup] = emissions[tup] + 1
      else:
        emissions[tup] = 1

      # Count how many pos in data
      if tup[0] in pos_count:
        pos_count[tup[0]] = pos_count[tup[0]] + 1
      else:
        pos_count[tup[0]] = 1

    if smoothing == NO_SMOOTHING:
      for tup in emissions:
        occurences = float(emissions[tup])
        emissions[tup] = math.log(occurences / pos_count[tup[0]])
    else:
      for tup in emissions:
        occurences = float(emissions[tup])
        emissions[tup] = math.log((occurences) / pos_count[tup[0]])
    return emissions

  """ For a bigram model, a map from (pos0, pos1) to Pr(pos1|pos0)"""
  @staticmethod
  def get_transitions(training_data, smoothing, known_words, parts_of_speech):
    transitions = {}
    pos_count = {}

    # Count the occurences of (pos, word) in training_data
    for x in range(1, len(training_data)):
      tup = (training_data[x-1][0], training_data[x][0])
      if tup in transitions:
        transitions[tup] = transitions[tup] + 1
      else:
        transitions[tup] = 1

      # Count how many pos in data
      if tup[0] in pos_count:
        pos_count[tup[0]] = pos_count[tup[0]] + 1
      else:
        pos_count[tup[0]] = 1
    
    if smoothing == NO_SMOOTHING:
      for tup in transitions:
        occurences = float(transitions[tup])
        transitions[tup] = math.log(occurences / pos_count[tup[0]])
    else:
      for x in pos_count:
        for y in pos_count:
          #print (x,y)
          if (x, y) in transitions:
            occurences = transitions[(x, y)]
          else:
            occurences = 0

          transitions[(x, y)] = math.log(float(occurences + 1) / (pos_count[x] + len(pos_count)))

    return transitions


def main():
  parser = optparse.OptionParser()
  parser.add_option('-s', '--smoothing', choices=(NO_SMOOTHING,
    ADD_ONE_SMOOTHING), default=NO_SMOOTHING)
  parser.add_option('-o', '--order', default=1, type=int)
  parser.add_option('-u', '--unknown',
      choices=(PREDICT_ZERO, PREDICT_MOST_COMMON_PART_OF_SPEECH,),
      default=PREDICT_ZERO)
  options, args = parser.parse_args()
  train_filename, test_filename = args
  training_data = hw5_common.read_part_of_speech_file(train_filename)
  if options.order == 0:
    model = BaselineModel(training_data)
  else:
    model = HiddenMarkovModel.train(
        training_data, options.smoothing, options.unknown, options.order)
  predictions = hw5_common.get_predictions(
      test_filename, model.predict_sentence)
  for word, prediction, true_pos in predictions:
    print word, prediction, true_pos

if __name__ == '__main__':
  main()
