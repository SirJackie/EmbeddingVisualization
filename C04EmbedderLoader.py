from gensim.models import KeyedVectors

print("Loading Model, Wait for 10s...")
model = KeyedVectors.load_word2vec_format("./Models/glove-wiki-gigaword-50.txt")

pass

print(model["tower"])
print(model.most_similar("hitler", topn=10))  # adolf stalin nazi ... mussolini goebbels

vec = model["hitler"] - model["germany"] + model["italy"]
print(model.similar_by_vector(vec, topn=10))  # mussolini 墨索里尼

vec = model["hitler"] - model["germany"] + model["japan"]
print(model.similar_by_vector(vec, topn=10))  # hirohito 裕仁

vec = model["hitler"] - model["germany"] + model["russia"]
print(model.similar_by_vector(vec, topn=10))  # stalin 斯大林

vec = model["hitler"] - model["germany"] + model["china"]
print(model.similar_by_vector(vec, topn=10))  # mao

vec = model["merkel"] - model["germany"] + model["china"]
print(model.similar_by_vector(vec, topn=10))  # hu jintao wen jiabao

vec = model["daughter"] - model["woman"] + model["man"]
print(model.similar_by_vector(vec, topn=10))  # son
