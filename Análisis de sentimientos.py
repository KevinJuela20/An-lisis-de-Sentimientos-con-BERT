#Librerías
from transformers import BertModel, AdamW, BertTokenizer,  get_linear_schedule_with_warmup
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from textwrap import wrap

#Ruta de dataset
ruta = './Datasets limpios/'


# Inicialización
RANDOM_SEED = 42
MAX_LEN = 200
BATCH_SIZE = 16
DATASET_1_PATH = ruta + 'dataset_coments_1.csv'
DATASET_2_PATH = ruta + 'dataset_coments_2.csv'
DATASET_3_PATH = ruta + 'dataset_coments_3.csv'
DATASET_4_PATH = ruta + 'dataset_coments_merge.csv'
NCLASSES = 2

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Cargar los dataset
df = pd.read_csv(DATASET_1_PATH)
df = df[:1000]
# df2 = pd.read_csv(DATASET_2_PATH)
# df3 = pd.read_csv(DATASET_3_PATH)
# df4 = pd.read_csv(DATASET_4_PATH)

# Reajustar dataset cambiando Positive a 1 y Negative a 0
df['label'] = (df['sentiment']=='Positive').astype(int)
df.drop('sentiment', axis=1, inplace=True)
df.head()

# TOKENIZACIÓN: convierte la palabras en números para ingresar los comentarios
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

# CREACIÓN DATASET: comentarios en números con tokens y encodings (formato de entrada de Bert)
class IMDBDataset(Dataset):
    def __init__(self,originalText,labels,tokenizer,max_len):
        self.originalText = originalText
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.originalText)
        
    def __getitem__(self, item):
        originalText = str(self.originalText[item])
        label = self.labels[item]
        encoding = tokenizer.encode_plus(
            originalText,
            max_length = self.max_len,
            truncation = True,
            add_special_tokens = True,
            return_token_type_ids = False,
            padding = True,
            return_attention_mask = True,
            return_tensors = 'pt'
            )
        

        return {
            'originalText': originalText,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        } 

#División del dataset para Traing y Testing
def data_loader(df, tokenizer, max_len, batch_size):
    dataset = IMDBDataset(
        originalText = df.originalText.to_numpy(),
        labels = df.label.to_numpy(),
        tokenizer = tokenizer,
        max_len = MAX_LEN
    )

    return DataLoader(dataset, batch_size = BATCH_SIZE, num_workers = 1)

# Dataset dividido 
df_train, df_test = train_test_split(df, test_size = 0.2, random_state=RANDOM_SEED)

train_data_loader = data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

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
            attention_mask = attention_mask,
            return_dict=False
        )
        drop_output = self.drop(cls_output)
        output = self.linear(drop_output)
        return output

# Construcción del modelo
model = BERTSentimentModule(NCLASSES)
model = model.to(device)

# Datos para el entrenamiento
EPOCHS = 5
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps = 0,
    num_training_steps = total_steps
)
loss_fn = nn.CrossEntropyLoss().to(device)

#Métodos para el entrenamiento y evaluación del modelo 
def train_model(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    for batch in data_loader:
        print(1)
        input_ids = batch['input_ids'].to(device)
        print(2)
        attention_mask = batch['attention_mask'].to(device)
        print(3)
        labels = batch['label'].to(device)
        print(4)
        outputs = model(input_ids = input_ids, attention_mask = attention_mask)
        print(5)
        _, preds = torch.max(outputs, dim=1)
        print(6)
        loss = loss_fn(outputs, labels)
        print(7)
        correct_predictions += torch.sum(preds == labels)
        print(8)
        losses.append(loss.item())
        print(9)
        loss.backward()
        print(10)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        print(11)
        optimizer.step()
        print(12)
        scheduler.step()
        print(13)
        optimizer.zero_grad()
        print(14)
    return correct_predictions.double()/n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids = input_ids, attention_mask = attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
    return correct_predictions.double()/n_examples, np.mean(losses)

# Entrenamiento del modelo
for epoch in range(EPOCHS):
    print('Epoch {} de {}'.format(epoch+1, EPOCHS))
    print('------------------')
    train_acc, train_loss = train_model(
        model, train_data_loader, loss_fn, optimizer, device, scheduler, len(df_train)
    )
    test_acc, test_loss = eval_model(
        model, test_data_loader, loss_fn, device, len(df_test)
    )
    print('Entrenamiento: Loss: {}, accuracy: {}'.format(train_loss, train_acc))
    print('Validación: Loss: {}, accuracy: {}'.format(test_loss, test_acc))
    print('')


# Guardar el modelo
model_path = "./modeloBertEntrenado.pth"
torch.save(model.state_dict(), model_path)