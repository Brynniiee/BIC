import pickle
with open('picklewithsplit.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data)
    data = pickle.load(f)
    print(type(data))
    print(data.keys() if isinstance(data, dict) else "Not a dict")