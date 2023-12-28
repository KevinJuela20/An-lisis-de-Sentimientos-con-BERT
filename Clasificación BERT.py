from transformers import BertModel, BertTokenizer
import torch
from torch import nn
from textwrap import wrap

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MAX_LEN = 200
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

class BERTSentimentModule(nn.Module):
    def __init__(self, n_classes):
        super(BERTSentimentModule, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.linear = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, cls_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        drop_output = self.drop(cls_output)
        output = self.linear(drop_output)
        return output

model = BERTSentimentModule(2)
model_path = "./modeloBertEntrenado.pth"

# Carga el modelo en la CPU
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.to(device)
model.eval()

def classifySentiment(review_text):
    encoding_review = tokenizer.encode_plus(
        review_text,
        max_length=MAX_LEN,
        truncation=True,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding_review['input_ids'].to(device)
    attention_mask = encoding_review['attention_mask'].to(device)
    output = model(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)
    print("\n".join(wrap(review_text)))
    if prediction.item() == 1:
        print('Sentimiento predicho: Positivo')
    else:
        print('Sentimiento predicho: Negativo')

review_text = "this dance is so weird"
classifySentiment(review_text)
