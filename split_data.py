# This py is used to split the sentences into words, won't remove stop words
def split_poetry(file):
    all_data = open(file, "r", encoding="utf-8").read()
    all_data_split = " ".join(all_data)
    with open("./data/split.txt", "w", encoding="utf-8") as f:
        f.write(all_data_split)


# split_poetry('./data/poetry_7.txt')
