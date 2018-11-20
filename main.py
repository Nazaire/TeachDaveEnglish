import codecs
from model import ResearchModel
from weightings import WEIGHTINGS
from vectorizer import get_sentence_vector
from sklearn.linear_model import SGDClassifier
import pickle
from slackclient import SlackClient
from nltk import word_tokenize, pos_tag
import nltk

slack_client = SlackClient("xoxb-434093319717-435238043348-stZ9oBvjWIOikQsIQkC45uS6")
slack_client.rtm_connect()

grammar = """
    ENTITY:	{<JJ>*<NN.*>+} # Chunk sequences of proper nouns 
""" 
cp = nltk.RegexpParser(grammar)

def main():
	#TRAIN
	# with codecs.open("train.csv", 'r', encoding="utf-8-sig") as file:
	# 	lines = [ x.strip() for x in file.readlines() ]
		
	# 	train = map( lambda s: s.split(","), lines)

	# 	Y, X = zip(*train)

	# 	model = ResearchModel()
	# 	model.train(X, Y, n_epochs=2000)
	# 	model.save("weightings.pickle")

	#TEST
	# with codecs.open("validate.csv", 'r', encoding="utf-8-sig") as file:
	# 	lines = [ x.strip() for x in file.readlines() ]
		
	# 	test = map( lambda s: s.split(","), lines)

	# 	Y, X = zip(*test)

	# 	# This is messy but whatevs
	# 	model = ResearchModel()
	# 	model.train(X, Y, n_epochs=0) # This isn't actually training im only initializing values

	# 	# Set the models WEIGHTINGS
	# 	# model.load("weightings.pickle")
	# 	model.weightings = WEIGHTINGS;
	# 	accuracy = model.test(X,Y)

	# 	print("ACCURACY = %f" % accuracy)
	# 	print(model.weightings)

	#SLACK
	with codecs.open("all.csv", 'r', encoding="utf-8-sig") as file:
		lines = [ x.strip() for x in file.readlines() ]
		
		data = map( lambda s: s.split(","), lines)

		Y, X = zip(*data)

		# Load weghtings
		with open("weightings.pickle", "rb") as file:
			WEIGHTINGS = pickle.load(file)

		print(WEIGHTINGS)
		# Make Ymap
		Ymap = {}
		i = 0
		for y in Y:
			if not y in Ymap:
				Ymap[y] = i
				i += 1

		# Convert
		X, Y = map(lambda x: get_sentence_vector(x), X), map(lambda y: Ymap[y], Y)


		model = SGDClassifier();
		model.fit(X, Y);

		print("Model ready")


		while True:
			events = slack_client.rtm_read()

			for event in events:
				if ('channel' in event and
					'text' in event and
					event['user'] == 'UCRPZ9R4K' and
					event.get('type') == 'message'):

					channel = event['channel']
					text = event['text'].replace("\n", "").replace("?", "")
					input_vector = get_sentence_vector(text.lower())

					prediction = model.predict([input_vector])

					label = None
					for k in Ymap:
						if Ymap[k] == prediction[0]:
							label = k

					tokens = pos_tag(word_tokenize(text))
					chunks = cp.parse(tokens)

					# Extract entities
					meta = [" ".join([token[0] for token in ch.leaves()]) for ch in filter(lambda x: x.label() == "ENTITY", chunks.subtrees())]

					# Remove empty meta
					meta = [m for m in meta if len(m) > 0]

					print("Predicted intent: %s, entities: %s" % (label, meta))

					slack_client.api_call(
						'chat.postMessage',
						channel=channel,
						text="I predicted the intent to be: %s with the entities: %s\n" % (label, ", ".join(meta)),
						as_user='true:'
                    )











main()
