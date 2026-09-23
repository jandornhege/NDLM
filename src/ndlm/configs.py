import torch.nn as nn
import time
import json
import os


def _activation_to_name(activation):
    if isinstance(activation, nn.Identity):
        return "identity"
    if isinstance(activation, nn.Sigmoid):
        return "sigmoid"
    raise ValueError(f"Unsupported activation type: {type(activation).__name__}")


def _activation_from_name(name):
    normalized = str(name).strip().lower()
    if normalized in {"identity", "nn.identity()", "identity()"}:
        return nn.Identity()
    if normalized in {"sigmoid", "nn.sigmoid()", "sigmoid()"}:
        return nn.Sigmoid()
    raise ValueError(f"Unsupported activation name in config JSON: {name}")


def _is_identity_activation(activation):
    return isinstance(activation, nn.Identity)

class config_object:
    def __init__(self):

        self.NUM_HIDDEN_ROLES = 3
        self.NUM_HIDDEN_CONCEPTS = 3
        self.NUM_LAYERS = 4
        self.NUM_EPOCHS = 5000
        self.TEST_INTERVAL = 10
        self.BATCH_SIZE = 1
        self.MODE = "strict"  # "relaxed" or "strict"
        self.ACTIVATION_FUNCTION = nn.Identity()
        self.TRANSITIVE_CLOSURE = False
        self.LEARNING_RATE = 1e-4
        self.WEIGHT_DECAY = 1e-4
        self.RESIDUAL = False
        self.INPUT_RESIDUAL = False
        self.INITIAL_FFN = False
        # Memory budget for Strict_Layer's role-role min-max attention.
        # Larger values compute more of the reduction in one vectorized
        # step (faster); set lower (e.g. near the per-k tensor size) to
        # fall back toward the original per-k iteration; None means
        # always compute in a single step regardless of memory use.
        # 512 MiB comfortably covers a single step up to ~180 objects at
        # 10 hidden roles, while leaving headroom for the CUDA context
        # and other tensors when training on a fractional GPU shard.
        self.STRICT_RR_CHUNK_BYTES = 256 * 2**20

        # self.NLM_RESIDUAL = True
        # self.NLM_EXCLUDE_SELF = True

    

    def to_dict(self):
        return {
            "NUM_HIDDEN_ROLES": self.NUM_HIDDEN_ROLES,
            "NUM_HIDDEN_CONCEPTS": self.NUM_HIDDEN_CONCEPTS,
            "NUM_LAYERS": self.NUM_LAYERS,
            "NUM_EPOCHS": self.NUM_EPOCHS,
            "TEST_INTERVAL": self.TEST_INTERVAL,
            "BATCH_SIZE": self.BATCH_SIZE,
            "MODE": self.MODE,
            "ACTIVATION_FUNCTION": _activation_to_name(self.ACTIVATION_FUNCTION),
            "TRANSITIVE_CLOSURE": self.TRANSITIVE_CLOSURE,
            "LEARNING_RATE": self.LEARNING_RATE,
            "WEIGHT_DECAY": self.WEIGHT_DECAY,
            "RESIDUAL": self.RESIDUAL,
            "INPUT_RESIDUAL": self.INPUT_RESIDUAL,
            "INITIAL_FFN": self.INITIAL_FFN,
            "STRICT_RR_CHUNK_BYTES": self.STRICT_RR_CHUNK_BYTES,
        }
    
    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)
    def save_to_file(self, file_path):
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
def config_from_nlm_args(args):
    config = config_object()

    config.NUM_LAYERS = getattr(args, "ndlm_depth", 4)
    config.NUM_HIDDEN_CONCEPTS = getattr(args, "ndlm_hidden_concepts", 10)
    config.NUM_HIDDEN_ROLES = getattr(args, "ndlm_hidden_roles", 10)

    config.TRANSITIVE_CLOSURE = getattr(args, "ndlm_transitive_closure", True)

    config.MODE = getattr(args, "ndlm_mode", "relaxed")
    config.ACTIVATION_FUNCTION = nn.Identity() if getattr(args, "ndlm_activation_function", "identity") == "identity" else nn.Sigmoid()
    config.RESIDUAL = getattr(args, "ndlm_residual", False)
    config.INPUT_RESIDUAL = getattr(args, "ndlm_input_residual", False)
    config.INITIAL_FFN = getattr(args, "ndlm_initial_ffn", False)

    return config

def config_from_json_file(file_path):
    with open(file_path, 'r') as f:
        config_dict = json.load(f)
    config = config_object()
    for key, value in config_dict.items():
        if hasattr(config, key):
            if key == "ACTIVATION_FUNCTION":
                value = _activation_from_name(value)
            setattr(config, key, value)
    return config

def generate_name(config, index):
    a= "ID" if _is_identity_activation(config.ACTIVATION_FUNCTION) else "SIG"
    tr= "TR" if config.TRANSITIVE_CLOSURE else "NOTR"
    tp = config.MODE
    return f"{tp}_{a}_{tr}_HC{config.NUM_HIDDEN_CONCEPTS}_HR{config.NUM_HIDDEN_ROLES}_L{config.NUM_LAYERS}_LR{config.LEARNING_RATE}_WD{config.WEIGHT_DECAY}_IDX{index}/"

if __name__=="__main__":
    c=config_object()
    print(c.to_json())