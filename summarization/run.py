import pickle
from summarizer import Summarizer
network_testing_data = pickle.load(open("./network_testing.pickle", "rb"))
print(len(network_testing_data))