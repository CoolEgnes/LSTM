from torch.utils.data import DataLoader
import split_data
import word_embedding
import load_data
import LSTM_model
import torch
import numpy as np


# generate the poetry, given the first word, predict the left
def generate_poetry():
    result = ""
    word_index = np.random.randint(0, word_size, 1)[0]
    result += index_2_word[word_index]

    h_0 = torch.zeros(size=(2, 1, hidden_num), dtype=torch.float32)
    c_0 = torch.zeros(size=(2, 1, hidden_num), dtype=torch.float32)

    for i in range(31):   # generate the left 31 words
        word_embedding = torch.tensor(w1[word_index][None][None])  # the index of the first word
        pre = model(word_embedding, (h_0, c_0))
        word_index = int(torch.argmax(pre))
        result += index_2_word[word_index]

    print(result)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    original_file = "./data/poetry_7.txt"
    split_file = "./data/split.txt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64

    all_data, (w1, word_2_index, index_2_word) = word_embedding.word_embedding(split_file, original_file)
    # print("w1.shape in main: ", w1.shape)  # [word_size, embedding_num]
    dataset = load_data.MyDataset(all_data, w1, word_2_index)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    hidden_num = 128
    lr = 0.001
    epochs = 100

    word_size, embedding_num = w1.shape

    model = LSTM_model.Mymodel(embedding_num, hidden_num, word_size)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for batch_index, (xs_embedding, ys_index) in enumerate(dataloader):
        xs_embedding = xs_embedding.to(device)
        print("xs_embedding.shape: ", xs_embedding.shape)
        ys_index = ys_index.to(device)

        pre = model(xs_embedding)
        loss = model.cross_entropy(pre, ys_index.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 10 == 0:
            print(f"loss:{loss:.3f}")
            generate_poetry()



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
