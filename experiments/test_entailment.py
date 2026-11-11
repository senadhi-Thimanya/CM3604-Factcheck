import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(model_name)

model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

print("Model labels:", model.config.id2label)

premise = "The human body requires vitamin D for normal bone health."
hypothesis = "Vitamin D is important for maintaining healthy bones."

inputs = tokenizer(
    premise,
    hypothesis,
    return_tensors="pt",
    truncation=True
)

device = next(model.parameters()).device
inputs = {key: value.to(device) for key, value in inputs.items()}

with torch.inference_mode():
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=-1)[0]

predicted_index = probabilities.argmax().item()

print("\nPrediction results:")

for index, probability in enumerate(probabilities):
    print(
        model.config.id2label[index],
        round(probability.item(), 4)
    )

print("\nPredicted class:", model.config.id2label[predicted_index])
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")