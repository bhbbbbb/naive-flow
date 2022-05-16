from torch import nn

class FooModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)
        return
    
    def forward(self, x):
        return self.layer(x)
    