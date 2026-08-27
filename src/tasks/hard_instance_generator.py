import torch
import json 
import ndlm.configs as Config
import time
import argparse

def generate_cycle_problem(identity=False):
    in_concepts= torch.zeros((0,12))  # (num_concepts, num_objects)
    if identity:
        in_roles= torch.zeros((2,12,12))  # list of (num_roles, num_objects, num_objects)
    else:
        in_roles= torch.zeros((1,12,12))  # list of (num_roles, num_objects, num_objects)
    out_concepts= torch.zeros((1,12))  # (num_concepts, num_objects)
    out_roles= torch.zeros((0,12,12))  # list of (num_roles, num_objects, num_objects)
    #6 cycle
    for i in range(6):
        in_roles[0,i,(i+1)%6]=1
        in_roles[0,(i+1)%6,i]=1
    if identity:
        for i in range(12):
            in_roles[1,i,i]=1
    
    for i in range(3):
        in_roles[0,6+i,(6+(i+1)%3)]=1
        in_roles[0,(6+(i+1)%3),6+i]=1

        in_roles[0,9+i,(9+(i+2)%3)]=1
        in_roles[0,(9+(i+2)%3),9+i]=1
    
    for i in range(6):
        out_concepts[0,i]=1
    # for i in range(6,12):
    #     out_concepts[1,i]=1
    return [(in_concepts, in_roles, out_concepts, out_roles)]

def generate_star_problem():
    in_concepts= torch.zeros((0,9))  # (num_concepts, num_objects)
    in_roles= torch.zeros((1,9,9))  # list of (num_roles, num_objects, num_objects)
    out_concepts= torch.zeros((2,9))  # (num_concepts, num_objects)
    out_roles= torch.zeros((0,9,9))  # list of (num_roles, num_objects, num_objects)
    
    in_roles[0,0,1]=1
    in_roles[0,1,0]=1
    in_roles[0,0,2]=1
    in_roles[0,2,0]=1
    in_roles[0,0,3]=1
    in_roles[0,3,0]=1
    in_roles[0,0,4]=1
    in_roles[0,4,0]=1

    in_roles[0,5,6]=1
    in_roles[0,6,5]=1
    in_roles[0,5,7]=1
    in_roles[0,7,5]=1
    in_roles[0,5,8]=1
    in_roles[0,8,5]=1

    for i in range(1,5):
        out_concepts[0,i]=1
    
    return [(in_concepts, in_roles, out_concepts, out_roles)]

def generate_line_problem(length=5):
    in_concepts= torch.zeros((2,2*length))  # (num_concepts, num_objects)
    in_roles= torch.zeros((1,2*length,2*length))  # list of (num_roles, num_objects, num_objects)
    out_concepts= torch.zeros((2,2*length))  # (num_concepts, num_objects)
    out_roles= torch.zeros((0,2*length,2*length))  # list of (num_roles, num_objects, num_objects)
    
    for i in range(length-1):
        in_roles[0,i,i+1]=1
        in_roles[0,i+1,i]=1

        in_roles[0,length+i,length+i+1]=1
        in_roles[0,length+i+1,length+i]=1

    in_concepts[0,0]=1
    in_concepts[1,length]=1

    for i in range(length):
        out_concepts[0,i]=1
    for i in range(length,2*length):
        out_concepts[1,i]=1
    return [(in_concepts, in_roles, out_concepts, out_roles)]


def get_hard_graphs_dataset(args):
    if args.graph_problem_type == "cycle":
        return {"cycle": (generate_cycle_problem(), (generate_cycle_problem()))}
    elif args.graph_problem_type == "line":
        line_problems = {}
        if not hasattr(args, "line_min_length"):
            args.line_min_length = 2
        if not hasattr(args, "line_max_length"):
            args.line_max_length = 33
        for length in range(args.line_min_length, args.line_max_length + 1):  # Generate line problems of lengths 2 to line_length
            line_problems[f"line_{length}"] = (generate_line_problem(length=length), generate_line_problem(length=length))
        return line_problems
    elif args.graph_problem_type == "star":
        return {"star": (generate_star_problem(), generate_star_problem())}
    else:
        raise ValueError(f"Unknown graph problem type: {args.graph_problem_type}")

def generate_xor_problem():
    in_concepts= torch.zeros((2,4))  # (num_concepts, num_objects)
    in_roles= torch.zeros((0,4,4))  # list of (num_roles, num_objects, num_objects)
    out_concepts= torch.zeros((1,4))  # (num_concepts, num_objects)
    out_roles= torch.zeros((0,4,4))  # list of (num_roles, num_objects, num_objects)
    
    in_concepts[0,0]=0
    in_concepts[1,0]=0
    in_concepts[0,1]=1
    in_concepts[1,1]=0
    in_concepts[0,2]=0
    in_concepts[1,2]=1
    in_concepts[0,3]=1
    in_concepts[1,3]=1

    out_concepts[0,1]=1
    out_concepts[0,2]=1

    return [(in_concepts, in_roles, out_concepts, out_roles)]

configurations = [
    {"name":"NDLMs_id", "tc": False, "activation": "identity", "strict": True},
    {"name": "NDLMs_sig","tc": False, "activation": "sigmoid", "strict": True},
    {"name": "NDLMs+_id","tc": True, "activation": "identity", "strict": True},
    {"name": "NDLMs+_sig","tc": True, "activation": "sigmoid", "strict": True},
    {"name":"NDLMr_id", "tc": False, "activation": "identity", "strict": False},
    {"name": "NDLMr_sig","tc": False, "activation": "sigmoid", "strict": False},
    {"name": "NDLMr+_id","tc": True, "activation": "identity", "strict": False},
    {"name": "NDLMr+_sig","tc": True, "activation": "sigmoid", "strict": False}
    ]



def get_config(selected_config, name):
    config_settings = configurations[selected_config]

    print("Generation complete!")
    config = Config.config_object()
    config.NUM_EPOCHS = 1000
    config.NUM_LAYERS = 3
    config.NUM_HIDDEN_CONCEPTS = 10
    config.NUM_HIDDEN_ROLES = 10
    config.TRANSITIVE_CLOSURE = config_settings["tc"]
    config.SKIP_CONNECTIONS = False
    config.RAGG_NORM = False
    config.RRA_NORM = False
    config.CRA_NORM = False

    config.ACTIVATION_FUNCTION = (
            torch.nn.Sigmoid() if config_settings["activation"] == "sigmoid" else torch.nn.Identity()
    )
    config.LAYER_TYPE = "single_step_minmax" if config_settings["strict"] else "single_step"
    config.EXP_NAME = f"{config_settings['name']}_{config_settings['activation']}_tc_{config_settings['tc']}_{time.time()}"

    config.EXP_PATH = f"ffn_experiments/HARD_FINAL/{name}/{selected_config}/{config.EXP_NAME}/"
    config.TEST_INTERVAL = 1
    config.LEARNING_RATE = 0.001
    config.write_config()
    return config
# LAYERS = 6
# EPOCHS = 1000
# def get_config(CONFIG_TO_USE, NAME, length=None, LAYERS=LAYERS, ACTIVATION_FUNCTION=torch.nn.Sigmoid()):
#     C0 = config.config_object()
#     C0.NUM_EPOCHS = EPOCHS
#     C0.NUM_LAYERS = LAYERS
#     C0.BATCH_SIZE = 1
#     C0.NUM_HIDDEN_CONCEPTS = 10
#     C0.NUM_HIDDEN_ROLES = 10
#     C0.SKIP_CONNECTIONS = False
#     C0.RAGG_NORM = False
#     C0.RRA_NORM = False
#     C0.CRA_NORM = False
#     C0.TEST_INTERVAL = 5
#     C0.LEARNING_RATE = 0.001
#     if CONFIG_TO_USE == 0:
#         C0.TRANSITIVE_CLOSURE = False
#         C0.ACTIVATION_FUNCTION = ACTIVATION_FUNCTION
#         C0.LAYER_TYPE = "single_step_minmax"
            
#         C0.EXP_NAME = f"{NAME}_0_noTR_strict_sig"+str(int(time.time()))
#         if length is not None:
#             C0.EXP_NAME = f"{NAME}_0ID_{length}_{C0.NUM_LAYERS}_noTR_sig"+str(int(time.time()))
#             C0.EXP_PATH = f"ffn_experiments/HARD_FINAL/{NAME}/{C0.NUM_LAYERS}/{CONFIG_TO_USE}/{C0.EXP_NAME}/"
#         else:
#             C0.EXP_PATH = f"ffn_experiments/HARD_FINAL/{NAME}/{C0.EXP_NAME}/"

#     elif CONFIG_TO_USE == 1:
#         C0.TRANSITIVE_CLOSURE = True
       
#         C0.ACTIVATION_FUNCTION = ACTIVATION_FUNCTION
#         C0.LAYER_TYPE = "single_step_minmax"
#         C0.EXP_NAME = f"{NAME}_1_TR_strict_sig"+str(int(time.time()))
#         if length is not None:
#             C0.EXP_NAME = f"{NAME}_1ID_{length}_{C0.NUM_LAYERS}_TR(minmax)_sig"+str(int(time.time()))
#             C0.EXP_PATH = f"ffn_experiments/HARD_FINAL/{NAME}/{C0.NUM_LAYERS}/{CONFIG_TO_USE}/{C0.EXP_NAME}/"
#         else:
#             C0.EXP_PATH = f"ffn_experiments/HARD_FINAL/{NAME}/{C0.EXP_NAME}/"

       

#     elif CONFIG_TO_USE == 2:
       
#         C0.TRANSITIVE_CLOSURE = False
     
#         C0.ACTIVATION_FUNCTION = ACTIVATION_FUNCTION
#         C0.LAYER_TYPE = "single_step"
#         C0.EXP_NAME = f"{NAME}_2_noTR_sig"+str(int(time.time()))
#         if length is not None:
#             C0.EXP_NAME = f"{NAME}_2ID_{length}_{C0.NUM_LAYERS}_noTR_sig"+str(int(time.time()))
#             C0.EXP_PATH = f"ffn_experiments/HARD_FINAL/{NAME}/{C0.NUM_LAYERS}/{CONFIG_TO_USE}/{C0.EXP_NAME}/"
#         else:
#             C0.EXP_PATH = f"ffn_experiments/HARD_FINAL/{NAME}/{C0.EXP_NAME}/"
       
#     elif CONFIG_TO_USE == 3:
     
#         C0.TRANSITIVE_CLOSURE = True
      
#         C0.ACTIVATION_FUNCTION = ACTIVATION_FUNCTION 
#         C0.LAYER_TYPE = "single_step"
#         C0.EXP_NAME = f"{NAME}_3_TR_relaxed"+str(int(time.time()))
#         if length is not None:
#             C0.EXP_NAME = f"{NAME}_3Sig_{length}_{C0.NUM_LAYERS}_TR(minmax)_sig"+str(int(time.time()))
#             C0.EXP_PATH = f"ffn_experiments/HARD_FINAL/{NAME}/{C0.NUM_LAYERS}/{CONFIG_TO_USE}/{C0.EXP_NAME}/"
#         else:
#             C0.EXP_PATH = f"ffn_experiments/HARD_FINAL/{NAME}/{C0.EXP_NAME}/"
    
#     C0.write_config()
    
#     return C0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hard line instance generation for a single k value.")
    # parser.add_argument("--k", type=int, required=True, help="Line length k (must be >= 2).")
    # parser.add_argument("--l", type=int, default=LAYERS, help="Number of layers (must be >= 1).")
    parser.add_argument("--c", type=int, default=-1, help="Configuration to use (must be >= 0).")
    args = parser.parse_args()

    # if args.k < 2:
    #     raise ValueError("k must be >= 2")
    # if args.l < 1:
    #     raise ValueError("l must be >= 1")

    # LAYERS = args.l

    selected_config = args.c
    if selected_config < 0:
        to_solve=list(range(len(configurations)))
    else:
        to_solve=[selected_config]

    for i in to_solve:
        data=[generate_star_problem()]
        co = get_config(i, name="STAR")
        acc_history, test_miss, test_acc, time_stats = ffn_main.main(data, data, co)
            
    # co = get_config(args.c, "LINE_CONVERGENCE2", args.k, LAYERS)
    # data=[generate_line_problem(args.k)]
    # acc_history, test_miss, test_acc, time_stats = ffn_main.main(data, data, co)