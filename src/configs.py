import torch.nn as nn
import time
import json
import os

class config_object:
    NUM_HIDDEN_ROLES = 3
    NUM_HIDDEN_CONCEPTS = 3
    NUM_LAYERS = 4
    NUM_EPOCHS = 5000
    TEST_INTERVAL = 1
    BATCH_SIZE = 1
    MODE = "relaxed"  # "relaxed" or "strict"
    ACTIVATION_FUNCTION = nn.Identity()
    TRANSITIVE_CLOSURE = True
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4


    

    def __dict__(self):
        return {
            "NUM_HIDDEN_ROLES": self.NUM_HIDDEN_ROLES,
            "NUM_HIDDEN_CONCEPTS": self.NUM_HIDDEN_CONCEPTS,
            "NUM_LAYERS": self.NUM_LAYERS,
            "NUM_EPOCHS": self.NUM_EPOCHS,
            "TEST_INTERVAL": self.TEST_INTERVAL,
            "BATCH_SIZE": self.BATCH_SIZE,
            "MODE": self.MODE,
            "ACTIVATION_FUNCTION": str(self.ACTIVATION_FUNCTION),
            "TRANSITIVE_CLOSURE": self.TRANSITIVE_CLOSURE,
            "LEARNING_RATE": self.LEARNING_RATE,
            "WEIGHT_DECAY": self.WEIGHT_DECAY,
        }
    
    def to_json(self):
        return json.dumps(self.__dict__(), indent=4)

def generate_name(config, index):
    a= "ID" if config.ACTIVATION_FUNCTION==nn.Identity() else "SIG"
    tr= "TR" if config.TRANSITIVE_CLOSURE else "NOTR"
    tp = config.MODE
    return f"{tp}_{a}_{tr}_HC{config.NUM_HIDDEN_CONCEPTS}_HR{config.NUM_HIDDEN_ROLES}_L{config.NUM_LAYERS}_LR{config.LEARNING_RATE}_WD{config.WEIGHT_DECAY}_IDX{index}/"

if __name__=="__main__":
    c=config_object()
    print(c.to_json())