"""
Generate mazes using maze-dataset library and convert them to graph with adjacency matrix.
"""

import numpy as np
from maze_dataset import MazeDataset, MazeDatasetConfig
from maze_dataset.generation import LatticeMazeGenerators
from maze_dataset.maze import LatticeMaze, SolvedMaze
import torch

def generate_mazes(n_mazes=5, grid_size=10):
    cfg = MazeDatasetConfig(
        name="graph_mazes",
        grid_n=grid_size,
        n_mazes=n_mazes,
        maze_ctor=LatticeMazeGenerators.gen_dfs,
        maze_ctor_kwargs=dict(do_forks=True),
    )
    
    dataset = MazeDataset.from_config(cfg)
    return dataset


def maze_to_adjacency_matrix(maze: LatticeMaze):
    n_rows, n_cols = maze.grid_shape
    n_nodes = n_rows * n_cols
    
    adjacency_matrix = np.zeros((n_nodes, n_nodes), dtype=np.int8)
    
    def coord_to_index(row, col):
        return row * n_cols + col
    
    connection_list = maze.connection_list
    
    for row in range(n_rows):
        for col in range(n_cols):
            current_idx = coord_to_index(row, col)
            
            # Downward connection
            if connection_list[0, row, col] and row + 1 < n_rows:
                neighbor_idx = coord_to_index(row + 1, col)
                adjacency_matrix[current_idx, neighbor_idx] = 1
                adjacency_matrix[neighbor_idx, current_idx] = 1
            
            # Rightward connection
            if connection_list[1, row, col] and col + 1 < n_cols:
                neighbor_idx = coord_to_index(row, col + 1)
                adjacency_matrix[current_idx, neighbor_idx] = 1
                adjacency_matrix[neighbor_idx, current_idx] = 1
    
    return adjacency_matrix


def get_node_coordinates(maze: LatticeMaze):
    """Get all node coordinates mapping."""
    n_rows, n_cols = maze.grid_shape
    return {row * n_cols + col: (row, col) 
            for row in range(n_rows) 
            for col in range(n_cols)}

def extract_solution_path(solved_maze: SolvedMaze):
    """Extract solution path as list of node indices."""
    n_cols = solved_maze.grid_shape[1]
    path_indices = []
    
    for pos in solved_maze.solution:
        idx = pos[0] * n_cols + pos[1]
        path_indices.append(idx)
    
    return path_indices


def extract_start_goal(solved_maze: SolvedMaze):
    """Extract start and goal node indices."""
    n_cols = solved_maze.grid_shape[1]
    start_idx = solved_maze.start_pos[0] * n_cols + solved_maze.start_pos[1]
    goal_idx = solved_maze.end_pos[0] * n_cols + solved_maze.end_pos[1]
    return start_idx, goal_idx

def maze_to_concept_role_representation(adjacency_matrix, start_idx, goal_idx, solution_path):
    #concepts: start, goal (target), 
    #roles: conn, id
    #target:
    concepts= torch.tensor([[1 if i==start_idx else 0 for i in range(len(adjacency_matrix))],
                            [1 if i==goal_idx else 0 for i in range(len(adjacency_matrix))]], dtype=torch.float32) # (num_concepts, num_objects)
    conn_role = torch.tensor(adjacency_matrix, dtype=torch.float32)  # (num_objects, num_objects)
    id_role = torch.eye(len(adjacency_matrix), dtype=torch.float32)  # (num_objects, num_objects)
    roles= torch.stack([conn_role, id_role], dim=0)  # (num_roles, num_objects, num_objects)
    target_roles = torch.zeros_like(conn_role).unsqueeze(0)  # (1, num_objects, num_objects)
    for i in range(len(solution_path)-1):
        target_roles[0, solution_path[i], solution_path[i+1]] = 1
    return concepts, roles,  torch.zeros((0 , len(adjacency_matrix))), target_roles

def maze_to_concept_role_representation_single_step(adjacency_matrix,goal_idx, solution_path, step_index=0, padding=0):
    #concepts: start, goal (target), 
    #roles: conn, id
    #target:
    concepts= torch.tensor([[1 if i==solution_path[step_index] else 0 for i in range(len(adjacency_matrix))]+[0]*padding,
                            [1 if i==goal_idx else 0 for i in range(len(adjacency_matrix))]+[0]*padding], dtype=torch.float32) # (num_concepts, num_objects)
    conn_role = torch.tensor(adjacency_matrix, dtype=torch.float32)  # (num_objects, num_objects)
    conn_role = torch.nn.functional.pad(conn_role, (0, padding, 0, padding), value=0)  # Pad to (num_objects+padding, num_objects+padding)
    id_role = torch.eye(len(adjacency_matrix), dtype=torch.float32)  # (num_objects, num_objects)
    id_role = torch.nn.functional.pad(id_role, (0, padding, 0, padding), value=0)  # Pad to (num_objects+padding, num_objects+padding)
    roles= torch.stack([conn_role, id_role], dim=0)  # (num_roles, num_objects, num_objects)
    target_concepts = torch.zeros((1, len(adjacency_matrix)+padding), dtype=torch.float32)  # Next position of the player
    target_concepts[0, solution_path[step_index+1]] = 1.0  # target is the next step in the solution path
    target_roles = torch.zeros((0, len(adjacency_matrix)+padding, len(adjacency_matrix)+padding), dtype=torch.float32)  # No target roles in this example
    return concepts, roles, target_concepts, target_roles

def log_dataset(mazes, path):
    with open(path, "w") as f:
        for i, maze in enumerate(mazes, 1):
            f.write(f"Maze {i}:\n")
            f.write(maze_to_adjacency_matrix(maze).tolist().__str__())


def generate_dataset(n_mazes=100, grid_size=3, padding=0, fixed_length_path=None):
    """Generate mazes and convert to graphs."""
    dataset = generate_mazes(n_mazes=n_mazes, grid_size=grid_size)
    concept_role_data = []
    for i, maze in enumerate(dataset, 1):
        
        solution_path = extract_solution_path(maze)
        if fixed_length_path is not None:
            # If the solution path is shorter than the fixed length, skip this maze
            if len(solution_path)<fixed_length_path:
                continue
            else:
                solution_path=solution_path[len(solution_path)-fixed_length_path:]
        # Convert to adjacency matrix
        adj_matrix = maze_to_adjacency_matrix(maze)
        start, goal = extract_start_goal(maze)
        
        for i in range(len(solution_path)-1):
            step_concept_role = maze_to_concept_role_representation_single_step(adj_matrix, goal, solution_path, step_index=i, padding=padding)
            concept_role_data.append(step_concept_role)
    print("length:", fixed_length_path,"num_mazes:", len(concept_role_data)/fixed_length_path)
    return concept_role_data

def save_concept_role_data(concept_role_data, file_path):
    torch.save(concept_role_data, file_path)

def load_concept_role_data(file_path):
    return torch.load(file_path)

def train_test_dataset_split(dataset, test_ratio=0.2):
    """Split dataset into train and test sets."""
    n_total = len(dataset)
    n_test = int(n_total * test_ratio)
    n_train = n_total - n_test
    
    train_set = dataset[:n_train]
    test_set = dataset[n_train:]
    
    return train_set, test_set

if __name__ == "__main__":
    gridsize=6
    goal_grid_size=6
    num_mazes=100
    max_path_length=10
    padding = 0
    for pl in range(10, 11):
        print(f"Generating dataset with path length: {pl}")
        path_length = pl
        ds = generate_dataset(n_mazes=num_mazes, grid_size=gridsize, padding=0, fixed_length_path=path_length)
        
        save_concept_role_data(ds, f"maze_dataset/maze_size_{gridsize}_num_{num_mazes}_path_length_{path_length}_0.pt")
    # ds = load_concept_role_data(f"maze_dataset/maze_size_{gridsize}_num_{num_mazes}_1.pt")
    # res = {}
    # print(len(ds))
    # for i, (concepts, roles, target_concepts, target_roles) in enumerate(ds):
    #     roles_tuple = tuple([tuple(role.tolist()) for role in roles[0]])
    #     if roles_tuple in res:
    #         res[roles_tuple]+=1
    #     else:
    #         res[roles_tuple]=1
    # print(res.values())
    # print(len(res))
    # a=[0]*gridsize**2
    # for k, v in res.items():
    #     a[v-1]+=1
    # print(a)
    # scaled_a = [count * (i) for i, count in enumerate(a)]
    # print(sum(scaled_a)/len(a))
    # adj=ds[0][1][0]
    # visualize adjacency matrix as maze
    # res=[["" for i in range(4)] for j in range(4)]
    # # add right arrows
    # for x in range(3):
    #     for y in range(4):
    #         idx=y*4+x
    #         #going left:
    #         if adj[idx, idx+1]==1:
    #             res[x][y]+="<-"
    
    # # add down arrows
    # for x in range(4):
    #     for y in  range(3):
    #         idx=y*4+x
    #         #going down:
    #         if adj[idx, idx+4]==1:
    #             res[x][y]+="^"
    # for row in res:
    #     print(" ".join([f"{cell:4}" for cell in row]))