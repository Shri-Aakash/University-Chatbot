import random
import json
import torch
from model import NeuralNet
from nltkUtils import tokenize,stem,bagOfWords


with open('intents.json','r') as f:
    intents=json.load(f)

FILE='model.pth'
data=torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["modelState"]

model=NeuralNet(input_size,hidden_size,output_size)
model.load_state_dict(model_state)
model.eval()

botName="Om"
print("Let's chat! type 'quit' to exit")
while True:
    sentence=input("User: ")
    if sentence=='quit':
        break
    sentence=tokenize(sentence)
    x=bagOfWords(sentence,all_words)
    x=x.reshape(1,x.shape[0])
    x=torch.from_numpy(x)

    output=model(x)
    _,pred=torch.max(output,dim=1)

    tag = tags[pred.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][pred.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{botName}: {random.choice(intent['responses'])}")

    else:
        print(f"{botName}: I do not understand...")
