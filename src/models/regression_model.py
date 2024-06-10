from torch.nn import Linear, Module, MSELoss
from torch import sigmoid
from transformers.file_utils import ModelOutput


class RegressionModel(Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(RegressionModel, self).__init__()
        self.hidden = Linear(n_feature, n_hidden)  # hidden layer
        self.predict = Linear(n_hidden, n_output)  # output layer

    def forward(self, inputs=None, labels=None) -> ModelOutput:
        if inputs is None:
            raise
        hidden_output = sigmoid(self.hidden(inputs))  # activation function for hidden layer
        logits = self.predict(hidden_output)  # linear output
        loss = None
        if labels is not None:
            loss_fct = MSELoss()
            loss = loss_fct(logits.squeeze(), labels.squeeze())

        return ModelOutput(logits=logits, loss=loss)
