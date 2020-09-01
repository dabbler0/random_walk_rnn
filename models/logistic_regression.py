import torch

# Now we have all these examples; train logsitic regression
class LogisticRegression(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(in_size, out_size)

    def forward(self, x):
        return self.linear(x)
