# This py is used to build the LSTM model
import torch
import torch.nn as nn


device = "cuda" if torch.cuda.is_available() else "cpu"


class Mymodel(nn.Module):
    def __init__(self, embedding_num, hidden_num, word_size):
        super().__init__()

        self.embedding_num = embedding_num
        self.hidden_num = hidden_num
        self.word_size = word_size

        self.lstm = nn.LSTM(input_size=embedding_num, hidden_size=hidden_num, batch_first=True, num_layers=2, bidirectional=False)
        self.flatten = nn.Flatten(0,1)
        self.linear = nn.Linear(hidden_num,  word_size)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, xs_embedding, h_0=None, c_0=None):
        # print("xs_embedding.shape in LSTM: ", xs_embedding.shape)  # torch.Size([batch_size, seq_len, embedding_dim])
        xs_embedding = xs_embedding.to(device)
        if h_0 == None or c_0 == None:
            h_0 = torch.zeros(size=(2, xs_embedding.shape[0], self.hidden_num), dtype=torch.float32)
            c_0 = torch.zeros(size=(2, xs_embedding.shape[0], self.hidden_num), dtype=torch.float32)
        hidden, (h_0,c_0) = self.lstm(xs_embedding)
        flatten_hidden = self.flatten(hidden)
        pre = self.linear(flatten_hidden)

        return pre
