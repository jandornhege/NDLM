import ffn_main_general_data as ffn_main
import maze_generator
import config as Config
import time
import torch
import argparse

configurations = [
        {"name":"NDLMr_id", "tc": False, "activation": "identity"},
        {"name": "NDLMr_sig","tc": False, "activation": "sigmoid"},
        {"name": "NDLMr+_id","tc": True, "activation": "identity"},
        {"name": "NDLMr+_sig","tc": True, "activation": "sigmoid"}
        ]
if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--configuration", type=int, default=0, help="Index of the configuration to use")
        args = parser.parse_args()
        maze_config = configurations[args.configuration]
        # train, test = maze_generator.train_test_dataset_split(dataset, test_ratio=0.2)
        config = Config.config_object()
        config.NUM_EPOCHS = 100
        config.NUM_LAYERS = 4
        config.NUM_HIDDEN_CONCEPTS = 10
        config.NUM_HIDDEN_ROLES = 10
        config.TRANSITIVE_CLOSURE = maze_config["tc"]
        config.SKIP_CONNECTIONS = False
        config.RAGG_NORM = False
        config.RRA_NORM = False
        config.CRA_NORM = False

        config.ACTIVATION_FUNCTION = (
                torch.nn.Sigmoid() if maze_config["activation"] == "sigmoid" else torch.nn.Identity()
        )
        config.LAYER_TYPE = "single_step_minmax"
        config.EXP_NAME = f"{maze_config['name']}_{maze_config['activation']}_tc_{maze_config['tc']}_{time.time()}"

        config.EXP_PATH = f"ffn_experiments/MAZE_EXPERIMENTS/"+config.EXP_NAME+"/"
        config.TEST_INTERVAL = 1
        config.LEARNING_RATE = 0.001
        config.write_config()

        train = maze_generator.load_concept_role_data(f"maze_dataset/maze_size_5_num_100_path_length_8_1.pt")
        test = []
        for i in range(2,17):
                test.append(maze_generator.load_concept_role_data(f"maze_dataset/maze_size_5_num_100_path_length_{i}_0.pt"))
        for i in range(2,17):
                test.append(maze_generator.load_concept_role_data(f"maze_dataset/maze_size_6_num_100_path_length_{i}_0.pt"))
        print(len(train), len(test))
        # print(f"Starting experiment with configuration: {config.EXP_NAME}")
        acc_history, test_miss, test_acc, time_stats = ffn_main.main([train], test, config)
        
                # print(f"Final Test Accuracy: {test_acc}, Final Test Misclassifications: {test_miss}, Time Stats: {time_stats}")