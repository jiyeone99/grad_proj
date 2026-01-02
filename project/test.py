import pickle

fitres_object = "Test"

# FitRes 객체를 직렬화하고 다시 역직렬화
serialized_fitres = pickle.dumps(fitres_object)
print(serialized_fitres)
deserialized_fitres = pickle.loads(serialized_fitres)
print(deserialized_fitres)
