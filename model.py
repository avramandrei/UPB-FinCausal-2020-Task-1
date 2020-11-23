import torch
import torch.nn as nn
import torch.nn.functional as F


class LangModelWithDense(nn.Module):
    def __init__(self, lang_model, input_size, hidden_size, fine_tune):
        super(LangModelWithDense, self).__init__()
        self.lang_model = lang_model

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(0.1)
        self.linear2 = nn.Linear(hidden_size, 1)

        self.fine_tune = fine_tune

    def forward(self, x, mask):
        if self.fine_tune:
            embeddings = self.lang_model(x, attention_mask=mask)[0][:, 0, :]

        else:
            with torch.no_grad():
                self.lang_model.eval()
                embeddings = self.lang_model(x, attention_mask=mask)[0][:, 0, :]

        output = self.dropout1(F.gelu(self.linear1(embeddings)))

        output = self.linear2(output)

        return output