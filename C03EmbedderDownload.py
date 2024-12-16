import gensim.downloader as api
model = api.load("glove-wiki-gigaword-50", return_path=True)
print(f"Model path: {model}")
