from transformers import BertTokenizer, BertForSequenceClassification

MODEL_NAME = "yiyanghkust/finbert-tone"
SAVE_DIR = "./finbert_model"  # Change to your preferred local directory

# Download and save locally
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)

tokenizer.save_pretrained(SAVE_DIR)
model.save_pretrained(SAVE_DIR)
