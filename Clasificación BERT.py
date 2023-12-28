# Script para probar el nuevo modelo ingresando solo el comentario

from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch import nn
import torch
from textwrap import wrap

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

MAX_LEN = 200
# TOKENIZACIÓN: convierte la palabras en números para ingresar los comentarios
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

# Modelo BERT, agregando una capa para clasificar sentimeintos
class BERTSentimentModule(nn.Module):
    def __init__(self, n_classes):
        super(BERTSentimentModule, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.linear = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, cls_output = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask
        )
        drop_output = self.drop(cls_output)
        output = self.linear(drop_output)
        return output

model = BERTSentimentModule(2)

# Carga los parámetros guardados en el modelo
model_path = "./modeloBertEntrenado.pth"
model.load_state_dict(torch.load(model_path))

# No olvides poner el modelo en modo de evaluación si no vas a entrenar más
model.eval()

def classifySentiment(review_text):
    encoding_review = tokenizer.encode_plus(
        review_text,
        max_length = MAX_LEN,
        truncation = True,
        add_special_tokens = True,
        return_token_type_ids = False,
        pad_to_max_length = True,
        return_attention_mask = True,
        return_tensors = 'pt'
        )
    
    input_ids = encoding_review['input_ids'].to(device)
    attention_mask = encoding_review['attention_mask'].to(device)
    output = model(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)
    print("\n".join(wrap(review_text)))
    if prediction:
        print('Sentimiento predicho: * * * * *')
    else:
        print('Sentimiento predicho: *')


review_text = "avengers infinity war at least had the good taste to abstain from jeremy renner no such luck in endgame"

classifySentiment(review_text)