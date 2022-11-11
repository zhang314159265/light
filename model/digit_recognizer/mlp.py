from light.utils import import_torch_light
import os
import pandas as pd

import_torch_light()

path = "~/TrainingData/digit-recognizer/" + (
    "train-tiny.csv"
    if os.environ.get("USE_TINY_DS", "0") == "1"
    else "train.csv"
)
path = os.path.expanduser(path)

class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(784, 100)
        self.linear2 = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x

def gen_batch(X, y, batch_size):
    off = 0
    while off < len(X):
        yield X[off: off + batch_size], y[off: off + batch_size]
        off += batch_size

def train(model, X, y, nepoch=5, lr=0.01, batch_size=100):
    optim = torch.optim.SGD(model.parameters(), lr=lr)
    ce = torch.nn.CrossEntropyLoss()
    for epoch_id in range(nepoch):
        print(f"Epoch {epoch_id}/{nepoch}")
        tot_loss = 0
        nbatch = 0
        for batch_X, batch_y in gen_batch(X, y, batch_size):
            loss = ce(model(batch_X), batch_y)
            optim.zero_grad()
            loss = loss.mean()
            tot_loss += loss.item()
            nbatch += 1
            loss.backward()
            optim.step()
        print(f"  Avg loss {tot_loss / nbatch}")

@torch.no_grad()
def test_model(model, X, y, batch_size=100):
    tot = 0
    tot_correct = 0
    for batch_X, batch_y in gen_batch(X, y, batch_size):
        pred = model(batch_X)
        pred_cls = pred.max(dim=1).indices
        tot += len(batch_X)
        tot_correct += (pred_cls == batch_y).sum()

    print(f"Accuracy: {tot_correct} / {tot} = {tot_correct / tot}")

def main():
    whole_df = pd.read_csv(path)
    ntraining_examples = int(len(whole_df) * 0.7)
    train_df = whole_df.iloc[:ntraining_examples]
    test_df = whole_df.iloc[ntraining_examples:]

    train_X = torch.LongTensor(train_df.loc[:, train_df.columns!="label"].to_numpy()) / 255
    train_y = torch.LongTensor(train_df.loc[:, "label"].to_numpy())

    test_X = torch.LongTensor(test_df.loc[:, test_df.columns!="label"].to_numpy()) / 255
    test_y = torch.LongTensor(test_df.loc[:, "label"].to_numpy())

    model = Classifier()
    train(model, train_X, train_y)
    test_model(model, test_X, test_y)
    print("bye")

if __name__ == "__main__":
    main()
