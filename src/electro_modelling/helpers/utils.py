import pickle


def save_pickle(data_dict, filepath):
    with open(filepath, 'wb') as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filepath):
    with open(filepath, 'rb') as handle:
        data = pickle.load(handle)
    return data
