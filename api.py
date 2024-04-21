from fastapi import FastAPI
from cassandra.cluster import Cluster
from datetime import datetime
import torch
import re
from transformers import BertTokenizerFast
import torch.nn as nn
import uuid

class BERT_SpamClassifier(nn.Module):
    def __init__(self, bert):
        super(BERT_SpamClassifier, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(768, 512)  
        self.fc2 = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.pooler_output
        x = self.fc1(pooled_output)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

app = FastAPI()
print("====================================================================================================")
# Load tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model_path = "my_spam_classifier_model.pth"
if not torch.cuda.is_available():
    model = torch.load(model_path, map_location=torch.device('cpu'))
else:
    model = torch.load(model_path)

# Set the model to evaluation mode
model.eval()

# Connect to Cassandra
cluster = Cluster()
session = cluster.connect()
session.set_keyspace('textclassifier')

# Create an index on the 'text' column
session.execute("CREATE INDEX IF NOT EXISTS text_index ON text_data (text)")

@app.post("/classify/")
async def classify_text(text: str, username: str):
    start_time = datetime.now()

    # Preprocess text
    text = clean_text(text)

    # Check if the text exists in the database
    query = f"SELECT label, probability FROM text_data WHERE text='{text}'"
    rows = session.execute(query)
    for row in rows:
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()
        return {"text": text,
                "label": "spam" if row.label == 1 else "ham",
                "probability": row.probability,
                "response_time": float(response_time),
                "source": "database"}

    # Tokenize text
    inputs = tokenizer.encode_plus(text, return_tensors="pt", truncation=True, padding=True)

    # Move inputs to device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Predict label
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    predicted_label = torch.argmax(outputs, dim=1).item()

    # Compute response time
    end_time = datetime.now()
    response_time = (end_time - start_time).total_seconds()

    # Store text, predicted label, username, and timestamp in the database
    probability = max(outputs.softmax(dim=1).tolist()[0])
    store_text_in_db(text, predicted_label, username, probability)

    return {"text": text,
            "label": "spam" if predicted_label == 1 else "ham",
            "probability": probability,
            "response_time": float(response_time),
            "source": "model"}

def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)  # remove special characters
    text = text.lower()  # lowercase
    return text

def store_text_in_db(text, label, username, probability):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    id = uuid.uuid1()  # Generate a unique ID
    query = f"INSERT INTO text_data (id, text, probability, label, username, timestamp) VALUES ({id}, '{text}', {probability}, {label}, '{username}', '{timestamp}')"
    session.execute(query)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
