import pickle
all_data = pickle.load(open("./training_0.pickle", "rb"))
test_data = all_data[:1000]
pickle.dump(test_data, open("./network_testing.pickle", "wb"))
