import torch
import json
import os

def load_problem(file_path, file_name):
    data = json.load(open(os.path.join(file_path, file_name), 'r'))
    train_data = data['train']
    train_list = []
    for train in train_data:
        train_list.append([train["input"][0], train["output"][0]])
    test_data = data['test']
    test_list = []
    for test in test_data:
        test_list.append([test["input"][0], test["output"][0]])
    
    return train_list, test_list


#matrix containing 1 in diagonal above main diagonal
conn_matrix= lambda size, padding: [[1 if j-i==1 else 0 for j in range(size)]+[0]*(padding) for i in range(size)]+[[0]*(size+padding) for _ in range(padding)]
inv_conn_matrix= lambda size, padding: [[1 if j-i==-1 else 0 for j in range(size)]+[0]*(padding) for i in range(size)]+[[0]*(size+padding) for _ in range(padding)]
id_matrix= lambda size: [[1 if j==i else 0 for j in range(size)] for i in range(size)]

color_concept_generator= lambda image_length, num_colors, color_i: [0]*image_length + [1 if j==color_i else 0 for j in range(num_colors)]


def construct_single_image_encoding(image_array, solution_array, num_colors, config=None, max_length=None):
    data=[]
    for t in range(len(image_array)):
        target_concept= [1 if i==t else 0 for i in range(len(image_array))]+[0]*num_colors
        cell_concept = [1]*len(image_array)+[0]*num_colors 
        color_concept = color_concept = [0]*len(image_array)+[1]*num_colors


        at_role = [[0]*len(image_array) + [1 if i==image_array[j] else 0 for i in range(num_colors)] for j in range(len(image_array))] + [[0]* (len(image_array)+num_colors) for _ in range(num_colors)] 
        at_role = torch.tensor(at_role, dtype=torch.float32).transpose(0,1).tolist()
        conn_role = conn_matrix(len(image_array), num_colors)
        id_role = id_matrix(len(image_array)+num_colors)
        
        concepts_arrays=[]
        # if config.target_concept:
        concepts_arrays.append(target_concept)
        # if config.cell_concept:
        concepts_arrays.append(cell_concept)
        # if config.color_concept:
        concepts_arrays.append(color_concept)
        # if config.color_i_concept:
        for i in range(num_colors):
            concepts_arrays.append(color_concept_generator(len(image_array), num_colors, i))
        concepts= torch.tensor(concepts_arrays, dtype=torch.float32) # (num_concepts, num_objects)
        
        roles_array= []
        # if config.at_role:
        roles_array.append(at_role)
        # if config.conn_role:
        roles_array.append(conn_role)
        # if config.id_role:
        roles_array.append(id_role)
        roles= torch.tensor(roles_array, dtype=torch.float32)  # list of (num_roles, num_objects, num_objects)
        # roles= torch.tensor([at_role, conn_role, id_role], dtype=torch.float32)  # list of (num_roles, num_objects, num_objects)

        target_concept= [0]* (len(image_array)+num_colors)
        target_concept[len(image_array)+solution_array[t]]=1

        #if max_length is given, pad the concepts and roles with zeros to reach max_length
        if max_length is not None:
            padding_needed = max(0, max_length - len(image_array))
            if padding_needed > 0:
                concepts = torch.cat([concepts, torch.zeros((concepts.shape[0], padding_needed))], dim=1)
                roles = torch.cat([roles, torch.zeros((roles.shape[0], padding_needed, roles.shape[2]))], dim=1)
                roles = torch.cat([roles, torch.zeros((roles.shape[0], roles.shape[1], padding_needed))], dim=2)
                target_concept += [0]*padding_needed
                target_roles = torch.zeros((0,roles.shape[1], roles.shape[2]), dtype=torch.float32)
            else:
                target_roles = torch.zeros((0,roles.shape[1], roles.shape[2]), dtype=torch.float32)
        target_concept = torch.tensor(target_concept, dtype=torch.float32)
        target_concept = target_concept.unsqueeze(0) # num_out_concepts, num_objects
        data.append( (concepts, roles, target_concept, target_roles))
                    #torch.zeros(0,len(image_array)+num_colors, len(image_array)+num_colors, dtype=torch.float32)) )
    return data


def construct_problem_encoding(problem_data, num_colors, config=None, max_length=None):
    dataset = []
    for image_array, solution_array in problem_data:
        image_encoding = construct_single_image_encoding(image_array, solution_array, num_colors, config, max_length=max_length)
        dataset.extend(image_encoding)
    return dataset

def create_dataset_from_file(file_path, file_name, config =None , max_length=None):
    train_list, test_list = load_problem(file_path, file_name)
    num_colors = 10  # Example number of colors
    train_dataset = construct_problem_encoding(train_list, num_colors, config, max_length=max_length)
    test_dataset = construct_problem_encoding(test_list, num_colors, config=config, max_length=max_length)
    return train_dataset, test_dataset

def create_dataset_from_dir(dir_path, config = None):
    # figure out maximum length of arrays in dir
    max_length=0
    for name in os.listdir(dir_path):
        if name.endswith(".json"):
            train_list, test_list = load_problem(dir_path , name)
            for image_array, solution_array in train_list+test_list:
                max_length = max(max_length, len(image_array))

    # now generate the datasets including the padding
    test_datasets = []
    train_datasets = []
    for name in os.listdir(dir_path):
        if name.endswith(".json"):
            train_dataset, test_dataset = create_dataset_from_file(dir_path , name, config, max_length=max_length)
            train_datasets+= train_dataset
            test_datasets+=test_dataset
            # train_datasets.append(train_dataset)
            # test_datasets.append(test_dataset)
    return train_datasets, test_datasets

if __name__ == "__main__":
    # Example usage
    import config
    c=config.get_exp_configs()[0]
    link = "1D-ARC-main/dataset/1d_denoising_mc/"
    name = "1d_denoising_mc_0.json" 
    # train, test = create_dataset_from_dir(link, c)
    train, test = create_dataset_from_file(link, name, c) 
    tr_concepts= torch.stack([item[0] for item in test])  # (num_batch, num_concepts, num_objects)
    tr_roles= torch.stack([item[1] for item in test])      # (num_batch, num_roles, num_objects, num_objects)
    tr_targets= torch.stack([item[2] for item in test])
    print(tr_concepts.shape, tr_roles.shape, tr_targets.shape)
    # num_colors = 10  # Example number of colors
    # train_dataset = construct_problem_encoding(train_list, num_colors, config)
    # test_dataset = construct_problem_encoding(test_list, num_colors, config)
    # train_dataset, test_dataset = create_dataset_from_dir("1D-ARC-main/dataset/1d_move_1p/", config)
    # print(f"Loaded {len(train_dataset)} training samples and {len(test_dataset)} testing samples.")