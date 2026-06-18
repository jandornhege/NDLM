import torch
import torch.nn as nn
import torch.optim as optim


class transitiveClosure(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, roles):
        B, K, N, _ = roles.shape

        # Add identity (reflexive closure)
        I = torch.eye(N, device=roles.device)
        I = I.expand(B, K, N, N)
        X = I+roles
        for _ in range(N):
            X = X @ (I+roles) 
        return self.sigmoid(X)

class transitiveClosure_sum(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, roles):
        B, K, N, _ = roles.shape

        # Add identity (reflexive closure)
        I = torch.eye(N, device=roles.device)
        I = I.expand(B, K, N, N)
        Out = torch.zeros_like(roles)
        X = I
        for _ in range(N):
            X = X @ (roles)
            Out+=X 
        return Out

class transitiveClosureMinMax(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, roles):
        B, K, N, _ = roles.shape

        # Add identity (reflexive closure)
        I = torch.eye(N, device=roles.device)
        T = I.expand(B, K, N, N)
        
        for _ in range(N):
            # Compute: T_new[i,j] = max_p(min(T[i,p], roles[p,j]))
            T_prev = T
            T = torch.full_like(T_prev, -torch.inf)

            # memory-efficient max-min composition over intermediate node p
            for p in range(N):
                left = T_prev[:, :, :, p].unsqueeze(-1)   # (B, K, N, 1)
                right = roles[:, :, p, :].unsqueeze(-2)   # (B, K, 1, N)
                T =torch.maximum(T, torch.minimum(left, right))
        return T

class transitiveClosureMinMax_agg(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, roles):
        B, K, N, _ = roles.shape

        # Add identity (reflexive closure)
        I = torch.eye(N, device=roles.device)
        T = I.expand(B, K, N, N)
        aggregation = torch.eye(N, device=roles.device).expand(B, K, N, N)  
        for _ in range(N):
            # Compute: T_new[i,j] = max_p(min(T[i,p], roles[p,j]))
            T_prev = T
            T = torch.full_like(T_prev, -torch.inf)

            # memory-efficient max-min composition over intermediate node p
            for p in range(N):
                left = T_prev[:, :, :, p].unsqueeze(-1)   # (B, K, N, 1)
                right = roles[:, :, p, :].unsqueeze(-2)   # (B, K, 1, N)
                T = torch.maximum(T, torch.minimum(left, right))
            aggregation = torch.maximum(aggregation, T)
        return aggregation


class CRAttention(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, concepts, roles):
        concepts_exp = concepts.unsqueeze(2).unsqueeze(-1)
        roles_exp = roles.unsqueeze(1)  # insert concept dimension
        scores = torch.matmul(roles_exp, concepts_exp)
        scores = scores.squeeze(-1)
        b, c, r, o = scores.shape
        return scores.reshape(b, c * r, o)

class CRMinMaxAttention(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, concepts, roles):
        concepts_exp = concepts[:, :, None, None, :]  # (b, c, 1, 1, k)
        roles_exp = roles[:, None, :, :, :]           # (b, 1, r, n, k)
        max_scores = (concepts_exp * roles_exp).max(dim=-1).values  # max over k
        min_scores = (concepts_exp * roles_exp).min(dim=-1).values  # min over k
        scores = torch.cat([max_scores, min_scores], dim=2)
        b, c, r, o = scores.shape
        return scores.reshape(b, c * r, o)
    
class RRAttention(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, roles):
        # roles: (b, num_roles, num_objects, num_objects)
        roles_T = roles.transpose(2,3)
        roles_j = roles.unsqueeze(2)
        roles_i = roles.unsqueeze(1)
        scores = torch.matmul(roles_j, roles_i)
        b, r, _, o, _ = scores.shape
        scores = scores.reshape(b, r * r, o, o)
        scores = torch.cat([scores, roles_T], dim=1)
        return scores

class RRMinMaxAttention(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, roles):
        # roles: (b, num_roles, num_objects, num_objects)
        roles_T = roles.transpose(2, 3)

        b, r, o, k = roles.shape

        #memory efficient computation
        max_scores = torch.full((b, r, r, o, o), -torch.inf, device=roles.device)
        min_scores = torch.full((b, r, r, o, o), torch.inf, device=roles.device)

        for kk in range(k):
            left  = roles[:, :, :, kk]        # [b, r, o]
            right = roles[:, :, kk, :]        # [b, r, o]

            prod = left[:, :, None, :, None] * right[:, None, :, None, :]  # [b,r,r,o,o]

            max_scores = torch.maximum(max_scores, prod)
            min_scores = torch.minimum(min_scores, prod)

            
        # combine min and max aggregation results
        scores = torch.cat([max_scores, min_scores], dim=1) 

        #reshape to (b, 2*r*r, o, o)
        b, r1, r2, o, _ = scores.shape
        scores = scores.reshape(b, 2 * r1 * r2, o, o)  # [b, 2*r*r, o, o]
        # append transposed roles
        scores = torch.cat([scores, roles_T], dim=1)  # final shape: [b, 2*r*r + r, o, o]

        return scores

class RoleAggregation(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, roles):
        return roles.sum(dim=2)  # sum over object dimension

class RoleMinMaxAggregation(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, roles):
        return torch.cat([roles.max(dim=2)[0], roles.min(dim=2)[0]], dim=1)  # max and min over object dimension

class ConceptExtension(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, concepts):
        concepts_unsqueezed = concepts.unsqueeze(3)  # shape (b, num_concepts, num_objects, 1)
        return concepts_unsqueezed.repeat(1, 1, 1, concepts.size(2))

# Extend along first and second role dimension
class ConceptExtensionDouble(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, concepts):
        concepts_unsqueezed = concepts.unsqueeze(3)  # shape (b, num_concepts, num_objects, 1)
        concepts_unsqueezed2 = concepts.unsqueeze(2)  # shape (b, num_concepts, num_objects, 1)

        ex1 =  concepts_unsqueezed.repeat(1, 1, 1, concepts.size(2))  # shape (b, num_concepts, num_objects, num_objects)
        ex2 = concepts_unsqueezed2.repeat(1, 1, concepts.size(2), 1)  # shape (b, num_concepts, num_objects, num_objects)
        return torch.cat([ex1, ex2], dim=1)  # shape (b, num_concepts*2, num_objects, num_objects)

class RFFN(nn.Module):
    def __init__(self, in_roles, out_roles, activation_function):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_roles, out_roles))
        self.in_roles = in_roles
        self.out_roles = out_roles
        
        self.bias = nn.Parameter(torch.Tensor(1, out_roles))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        self.activation_function = activation_function

    def forward(self, x):
        if x.shape[1] != self.in_roles:
            print("Expected input role size:", self.in_roles, "but got:", x.shape[1])
            raise ValueError("Input role size does not match the initialized size.")
        x_t = x.transpose(1, 2)
        out = torch.matmul(x_t.transpose(2, 3), self.weight) + self.bias
        out = out.transpose(2, 3).transpose(1, 2)
        out = self.activation_function(out)
        return out


class CFFN(nn.Module):
    def __init__(self, in_concepts, out_concepts, activation_function):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_concepts, in_concepts))
        self.bias = nn.Parameter(torch.Tensor(out_concepts))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        self.activation_function = activation_function
        self.in_concepts = in_concepts
        self.out_concepts = out_concepts

    def forward(self, x):
        if x.shape[1]!=self.in_concepts:
            raise ValueError("Input concept size does not match the initialized size.") 
        # x: (b, num_concepts, num_objects)
        x_t = x.transpose(1, 2)
        out = torch.matmul(x_t, self.weight.t()) + self.bias
        out = self.activation_function(out)
        return out.transpose(1, 2)
    
       
class MultiLayerNDLM(nn.Module):
    def __init__(self, in_concepts, in_roles, out_concepts, out_roles, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        for i in range(self.config.NUM_LAYERS):
            _in_concepts = in_concepts if i == 0 else self.config.NUM_HIDDEN_CONCEPTS
            _in_roles = in_roles if i == 0 else self.config.NUM_HIDDEN_ROLES
            _out_concepts = out_concepts if i == self.config.NUM_LAYERS - 1 else self.config.NUM_HIDDEN_CONCEPTS
            _out_roles = out_roles if i == self.config.NUM_LAYERS - 1 else self.config.NUM_HIDDEN_ROLES
            # don't apply activation function on last layer (addapt if multilayer MLPs are used in the future)
            _activation_function = self.config.ACTIVATION_FUNCTION if i < self.config.NUM_LAYERS - 1 else nn.Identity()

            # print("Layer", i, "in_concepts", _in_concepts, "in_roles", _in_roles, "out_concepts", _out_concepts, "out_roles", _out_roles)
            if config.MODE == "relaxed":
                self.layers.append(Relaxed_Layer(_in_concepts, _in_roles, _out_concepts, _out_roles, _activation_function, config))
            elif config.MODE == "strict":
                self.layers.append(Strict_Layer(_in_concepts, _in_roles, _out_concepts, _out_roles, _activation_function, config))
            else:
                raise ValueError("Invalid MODE in config. Expected 'relaxed' or 'strict', got: {}".format(config.MODE))
            
    def forward(self, concepts, roles):
        out_concepts = concepts
        out_roles = roles
        # apply layers sequentially
        for layer in self.layers:
            out_concepts, out_roles = layer(out_concepts, out_roles)
        return out_concepts, out_roles

class Relaxed_Layer(nn.Module):
    def __init__(self, in_concepts, in_roles, out_concepts, out_roles, activation_function, config):
        super().__init__()
        self.config = config 
        self.in_concepts = in_concepts
        self.out_concepts = out_concepts
        self.out_roles = out_roles
        self.in_roles = in_roles
    
        #Initialize Stage 1 operations
        self.CRA = CRAttention()
        self.RRA = RRAttention()
        self.TC = transitiveClosure()
        self.RAgg = RoleAggregation()
        self.CExt = ConceptExtensionDouble()

        # determine size of intermediate concepts and roles after Stage 1 operations 
        if self.config.TRANSITIVE_CLOSURE:
            self.total_intermediate_r = in_roles*3 + in_roles*in_roles+ in_concepts*2
        else:
            self.total_intermediate_r = in_roles*2 + in_roles*in_roles+ in_concepts*2
        self.total_intermediate_c = in_concepts + in_roles * in_concepts +  in_roles
        # print("Total intermediate size: roles:", self.total_intermediate_c,", concepts:" ,self.total_intermediate_r)

        # Initialize FFN layers for concepts and roles
        self.C_ffn = CFFN(self.total_intermediate_c, out_concepts, activation_function)
        self.R_ffn = RFFN(self.total_intermediate_r, out_roles, activation_function)

    def forward(self, concepts, roles):
        if concepts.size(1)!= self.in_concepts or roles.size(1)!= self.in_roles:
            raise ValueError("Input concept or role size does not match the initialized size.")
        # concepts: (batch, num_concepts, num_objects)
        # roles: (batch, num_roles, num_objects, num_objects)
        cra_output_concepts = self.CRA(concepts, roles)  # (batch, num_concepts * num_roles, num_objects)
    
        rra_output_roles = self.RRA(roles)  # (batch, num_roles * num_roles, num_objects, num_objects)
        c_roles = self.CExt(concepts)  # (batch, num_concepts, num_objects, num_objects)
        r_concepts = self.RAgg(roles)  # (batch, num_roles, num_objects)
    
        # compute and concat TC if enabled
        if self.config.TRANSITIVE_CLOSURE:
            roles_tc = self.TC(roles)
            roles = torch.cat([roles, roles_tc], dim=1)
        
        cat_concepts = torch.cat([concepts, cra_output_concepts, r_concepts], dim=1)  # (batch, in_concepts + in_concepts * in_roles, num_objects)
        cat_roles = torch.cat([roles, rra_output_roles, c_roles], dim=1)  # (batch, in_roles + in_roles * (in_roles + 1), num_objects, num_objects) 
        
        updated_concepts = self.C_ffn(cat_concepts)  # (batch, out_concepts, num_objects)
        updated_roles = self.R_ffn(cat_roles)  # (batch, out_roles, num_objects, num_objects)
        
        return updated_concepts, updated_roles

class Strict_Layer(nn.Module):
    def __init__(self, in_concepts, in_roles, out_concepts, out_roles, activation_function, config):
        super().__init__()
        self.config = config 
        self.in_concepts = in_concepts
        self.out_concepts = out_concepts
        self.out_roles = out_roles
        self.in_roles = in_roles
    
        # Initialize Stage 1 operations with min-max variants
        self.CRA = CRMinMaxAttention()
        self.RRA = RRMinMaxAttention()
        self.RAgg = RoleMinMaxAggregation()
        self.CExt = ConceptExtensionDouble()
        self.TC = transitiveClosureMinMax_agg()
        
        # determine size of intermediate concepts and roles after Stage 1 operations 
        if self.config.TRANSITIVE_CLOSURE:
            self.total_intermediate_r = in_roles*3 + in_roles*in_roles*2+ in_concepts*2
        else:
            self.total_intermediate_r = in_roles*2 + in_roles*in_roles*2 + in_concepts*2
        self.total_intermediate_c = in_concepts + in_roles * in_concepts*2 +  in_roles*2
        # print("Total intermediate size: roles:", self.total_intermediate_c,", concepts:" ,self.total_intermediate_r)
        
        self.R_ffn = RFFN(self.total_intermediate_r, out_roles, activation_function)
        self.C_ffn = CFFN(self.total_intermediate_c, out_concepts, activation_function)

    def forward(self, concepts, roles):
        if concepts.size(1)!= self.in_concepts or roles.size(1)!= self.in_roles:
            raise ValueError("Input concept or role size does not match the initialized size.")
        # concepts: (batch, num_concepts, num_objects)
        # roles: (batch, num_roles, num_objects, num_objects)

        CRA = self.CRA(concepts, roles)  # (batch, num_concepts * num_roles, num_objects)
        RRA = self.RRA(roles)  # (batch, num_roles * num_roles, num_objects, num_objects)
        EXT = self.CExt(concepts)  # (batch, num_concepts, num_objects, num_objects)
        AGG = self.RAgg(roles)  # (batch, num_roles, num_objects)
    
        # compute and concat TC if enabled
        if self.config.TRANSITIVE_CLOSURE:
            roles_tc = self.TC(roles)
            roles = torch.cat([roles, roles_tc], dim=1)
        
        cat_concepts = torch.cat([concepts, CRA, AGG], dim=1)  # (batch, in_concepts + in_concepts * in_roles, num_objects)
        cat_roles = torch.cat([roles, RRA, EXT], dim=1)  # (batch, in_roles + in_roles * (in_roles + 1), num_objects, num_objects) 
        
        
        updated_concepts = self.C_ffn(cat_concepts)  # (batch, out_concepts, num_objects)
        updated_roles = self.R_ffn(cat_roles)  # (batch, out_roles, num_objects, num_objects)
        
        return updated_concepts, updated_roles