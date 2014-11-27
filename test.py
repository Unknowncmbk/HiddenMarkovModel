# 
# Provides test cases for various test cases.
#
# Compiled against Python 2.7
# Author: Stephen Bahr (sbahr@bu.edu)

import copy
import math
import unittest
import markov

class Test(unittest.TestCase):
  '''Unit tests for assignment 5'''

  _TRAINING_DATA = (
      ('<s>', '<s>'),
      ('N', 'spot'),
      ('V', 'runs'),
      ('<s>', '<s>'),
      ('N', 'spot'),
      ('V', 'runs'),
      ('<s>', '<s>'),
      ('N', 'spot'),
      ('N', 'runs'),
      ('<s>', '<s>'),
      ('V', 'runs'),
      ('V', 'spot'),
      ('<s>', '<s>'),
      ('V', 'runs'),
      ('N', 'spot'),
      ('<s>', '<s>'))

  _TRANSITION_PROBABILITIES = {
      # Transitions from <s>
      ('<s>', 'N'): math.log(0.6),
      ('<s>', 'V'): math.log(0.4),
      # Transitions from N
      ('N', '<s>'): math.log(0.4),
      ('N', 'N'): math.log(0.2),
      ('N', 'V'): math.log(0.4),
      # Transitions from V
      ('V', '<s>'): math.log(0.6),
      ('V', 'N'): math.log(0.2),
      ('V', 'V'): math.log(0.2),
  }

  _EMISSION_PROBABILITIES = {
      # Emission probabilities from <s>
      ('<s>', '<s>'): math.log(1),
      # Emission probabilities from N
      ('N', 'runs'): math.log(0.2),
      ('N', 'spot'): math.log(0.8),
      # Emission probabilities from V
      ('V', 'spot'): math.log(0.2),
      ('V', 'runs'): math.log(0.8),
  }

  _MODEL = hw5.HiddenMarkovModel(1, copy.deepcopy(_EMISSION_PROBABILITIES),
      copy.deepcopy(_TRANSITION_PROBABILITIES), ['N', 'V', '<s>'],
      set(['runs', 'spot']))

  _LATTICE = [
      {'<s>': (math.log(1), None)},
      {'N': (math.log(0.48), '<s>'), 'V': (math.log(0.08), '<s>')},
      {'N': (math.log(0.0192), 'N'), 'V': (math.log(0.1536), 'N')},
      {'<s>': (math.log(0.09216), 'V')}]

  def test_train(self):
    model = hw5.HiddenMarkovModel.train(self._TRAINING_DATA,
        hw5.NO_SMOOTHING,
        hw5.PREDICT_MOST_COMMON_PART_OF_SPEECH,
        order=1)
    self.assertEqual(1, model.order)
    for (p0, p1), log_expected in self._TRANSITION_PROBABILITIES.iteritems():
      found = math.exp(model.transition.get((p0, p1), -50))
      expected = math.exp(log_expected)
      self.assertAlmostEqual(expected, found, places=2, msg=(
        'Pr(%s=>%s): should be %s, is %s' % (p0, p1, expected, found)))

    for (p, w), log_expected in self._EMISSION_PROBABILITIES.iteritems():
      found = math.exp(model.emission.get((p, w), -50))
      expected = math.exp(log_expected)
      self.assertAlmostEqual(expected, found, places=2, msg=(
        'Pr(%s|%s): should be %s, is %s' % (w, p, expected, found)))

  def test_compute_lattice(self):
    lattice = self._MODEL.compute_lattice(['spot', 'runs'])
    self.assertEqual(len(lattice), len(self._LATTICE))
    for expected, found in zip(self._LATTICE, lattice):
      for pos in expected:
        log_expected_score, expected_previous = expected[pos]
        found_score, found_previous = found.get(pos, (0, None))
        expected_score = math.exp(log_expected_score)
        self.assertEqual(expected_previous, found_previous)
        self.assertAlmostEqual(expected_score, math.exp(found_score), places=4)

  def test_find_best_path(self):
    self.assertEqual(['N', 'V'],
        hw5.HiddenMarkovModel.find_best_path(self._LATTICE))

  def test_baseline(self):
    model= hw5.BaselineModel(
        self._TRAINING_DATA + (('N', 'spot'), ('N', 'spot')))
    self.assertEqual('N', model.dictionary['spot'])
    self.assertEqual('V', model.dictionary['runs'])
    self.assertEqual('N', model.default)

  def test_add_one_smoothing(self):
    model = hw5.HiddenMarkovModel.train(
        (('N', 'spot'),
         ('V', 'runs'),
         ('N', 'spot'),
         ('N', 'spot')),
        hw5.ADD_ONE_SMOOTHING,
        hw5.PREDICT_MOST_COMMON_PART_OF_SPEECH,
        order=1)
    self.assertAlmostEqual(0.8, math.exp(model.emission['N', 'spot']), places=2)
    self.assertAlmostEqual(0.67, math.exp(model.emission['V', 'runs']),
      places=2)
    self.assertAlmostEqual(0.2, math.exp(model.emission['N', 'runs']), places=2)
    self.assertAlmostEqual(0.33, math.exp(model.emission['V', 'spot']),
      places=2)
    self.assertAlmostEqual(0.5, math.exp(model.transition['N', 'N']), places=2)
    self.assertAlmostEqual(0.5, math.exp(model.transition['N', 'V']), places=2)
    self.assertAlmostEqual(0.33, math.exp(model.transition['V', 'V']), places=2)
    self.assertAlmostEqual(0.67, math.exp(model.transition['V', 'N']), places=2)

  def test_hmm_train_has_default_parameters(self):
    # Make sure that later parameters to HMM.train are optional
    model = hw5.HiddenMarkovModel.train(self._TRAINING_DATA)

if __name__ == '__main__':
  unittest.main()
