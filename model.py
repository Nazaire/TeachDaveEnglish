import numpy as np
from sklearn.linear_model import LinearRegression, SGDClassifier
from vectorizer import get_sentence_vector
import datetime
import itertools
# from data import get_sets, get_subsets
# from respond import respond
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import datasets
import random
from weightings import WEIGHTINGS, weight
from pprint import pprint
import pickle



# Mutate a child slightly
def mutate(child):
	
	# Mutate n tags
	for i in range(1):
		tag = random.choice(child.keys()) 

		child[tag] *= random.uniform(0.90, 1.10)


	return child

childid = itertools.count(0)

class Child:
	def __init__(self, weightings):
		self.id = "N%d" % next(childid)
		self.weightings = weightings;

	def make_child(self):
		return Child(mutate(self.weightings.copy()));

	def __str__(self):
		return "#%s: %s" % (self.id, self.weightings)

class ResearchModel:
	def __init__(self):
		self.Ymap = {}
		self.sets = []

	def mapSet(self, setToMap):
		X, Y = zip(*setToMap)
		return map(lambda x: get_sentence_vector(x), X), map(lambda y: self.Ymap[y], Y)

	def getRandomSubsets(self):
		# Return equal but random sets

		setA, setB = [], []
		for s in self.sets:
			random.shuffle(s)

			a, b = np.array_split(s, 2)

			setA += a.tolist()
			setB += b.tolist()

		return setA, setB

	def scoreWeightings(self, weightings, setA, setB):

		# Set weightings
		for k in weightings:
			WEIGHTINGS[k] = weightings[k]

		# Map X, Y for training
		X, Y = self.mapSet(setA)

		# Build a SVM model off setA
		model = SGDClassifier();
		model.fit(X, Y);

		# Map X, Y for testing
		X, Y = self.mapSet(setB)

		# Test model on setB
		predictions = model.predict(X)
		score = np.mean(predictions == Y)

		return score, model, weightings


	def step(self, parents):
		children = [];

		# Keep parents in next generation
		for parent in parents:
			children.append(parent);

		# Mutate parents to create children
		while len(children) < 20:
			for parent in parents:
				children.append(parent.make_child())

		# Compute average score over n different subsets
		scores = {}
		for i in range(10):
			# Get random subsets
			setA, setB = self.getRandomSubsets()

			for child in children:
				results = self.scoreWeightings(child.weightings, setA, setB)

				if not child.id in scores:
					scores[child.id] = []

				scores[child.id].append(results[0])

		evaluated = []
		for child in children:
			avg_score = np.mean(scores[child.id])

			evaluated.append([avg_score, child])

		# Evaluate each child
		# for child in children:

		# 	scores = []

		# 	# Get average score from 5 shuffles
		# 	for i in range(3):

		# 		results = self.scoreWeightings(child.weightings, setA, setB)

		# 		scores.append(results[0])

		# 	score = np.mean(scores)

		# 	evaluated.append([score, child])

		evaluated.sort(key=lambda x: x[0], reverse=True)

		# For debug: Print the score of the best parent
		parent_performance = None
		for each in evaluated:
			if each[1].id == parents[0].id:
				parent_performance = each

		# Return top n performing children
		return evaluated[:5], parent_performance

	def train(self, X, Y, n_epochs=200):

		if n_epochs > 0:
			with open("model.csv", "w") as logfile:
				logfile.write("Started training at %s\n" % datetime.datetime.now())


		# set mappings for Y values

		i = 0
		for y in Y:
			if not y in self.Ymap:
				self.Ymap[y] = i
				i += 1

		# Split training set into an even distribution of Y
		zipped = zip(X, Y)
		for t in zipped:
			i = self.Ymap[t[1]]

			if i >= len(self.sets):
				self.sets.append([t])
			else:
				self.sets[i].append(t)

		for s in self.sets:
			assert len(s) >= 2, "Training set requires at least two values of X for each value of Y"

		# Initialize single parent as default WEIGHTINGS
		parents = [Child(WEIGHTINGS)]

		top_survivor = [0.0, None, {}]

		epoch = 1

		# Start training
		for i in range(n_epochs):

			# Perform a single step
			survivors, parent_performance = self.step(parents)

			print "Epoch #%d: %f%% -- #%s\t\t#%s = %f" % (epoch, survivors[0][0] * 100, survivors[0][1].id, parent_performance[1].id, parent_performance[0])

			with open("model.csv", "a") as logfile:
				logfile.write("%d, %f, %f, %f\n" % ( epoch, survivors[0][0], survivors[1][0], survivors[2][0]))

			# Keep top_survivor
			if top_survivor[0] < survivors[0][0]:
				top_survivor =  survivors[0]

			parents = map(lambda s: s[1], survivors);

			epoch += 1

		if top_survivor[1] is not None:
			self.weightings = top_survivor[1].weightings

	def load(self, filename):
		with open(filename, "rb") as file:
			self.weightings = pickle.load(file)

	def save(self, filename):
		with open(filename, "wb") as file:
			pickle.dump(self.weightings, file)

	def test(self, X, Y):
		scores = []

		for i in range(500):
			setA, setB = self.getRandomSubsets()

			results = self.scoreWeightings(self.weightings, setA, setB)

			scores.append(results[0])

		return np.mean(scores)





# # ResearchModel()
# ResearchModel().train([
# 	"what is the weather",
# 	"tell me what the time is",
# 	"what date is it",
# 	"what is it like outside",
# 	"what is the time",
# 	"what time is it",
# 	"what is the date"], [
# 	"WEATHER",
# 	"TIME",
# 	"DATE",
# 	"WEATHER",
# 	"TIME",
# 	"TIME",
# 	"DATE"])