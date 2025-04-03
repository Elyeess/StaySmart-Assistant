from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# Charger le modèle et le tokenizer
model_name = "distilbert-base-uncased"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    exit()

# Fonction pour obtenir les embeddings
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings

def compute_similarity(desc, comment):
    emb1 = get_embeddings(desc)
    emb2 = get_embeddings(comment)
    similarity = torch.mm(emb1, emb2.transpose(0, 1))
    return float(similarity[0][0])