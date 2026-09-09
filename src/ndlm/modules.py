import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.checkpoint import checkpoint

import ndlm.configs

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
    """Role-role min-max composition: scores[i,j,o1,o2] = max/min_k roles[i,o1,k]*roles[j,k,o2].

    The reduction over the intermediate-object dimension `k` can be done
    in one fully vectorized step (fastest, needs a (b,r,r,o,o,chunk)
    tensor with chunk=num_objects) or by iterating over `k` in smaller
    chunks (the original behavior, chunk=1) to bound peak memory.
    `chunk_bytes` picks the largest chunk that keeps that intermediate
    tensor under the given byte budget; pass `chunk_bytes=None` to
    always process all of `k` in a single step regardless of size, or a
    small value to fall back to the original per-`k` iteration.
    """

    def __init__(self, chunk_bytes=256 * 2**20):
        super().__init__()
        if chunk_bytes is not None and chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be positive or None")
        self.chunk_bytes = chunk_bytes

    def _chunk_size(self, b, r, o, element_size):
        if o == 0:
            return 1
        if self.chunk_bytes is None:
            return o
        bytes_per_k = b * r * r * o * o * element_size
        if bytes_per_k <= 0:
            return o
        return max(1, min(o, self.chunk_bytes // bytes_per_k))

    @staticmethod
    def _chunk_extrema(roles, start, end):
        left = roles[:, :, None, :, None, start:end]
        right = roles[:, :, start:end, :].transpose(2, 3)[:, None, :, None, :, :]
        prod = left * right
        return prod.max(dim=-1).values, prod.min(dim=-1).values

    def forward(self, roles):
        # roles: (b, num_roles, num_objects, num_objects)
        b, r, o, k = roles.shape
        chunk_size = self._chunk_size(b, r, o, roles.element_size())

        max_scores = torch.full((b, r, r, o, o), -torch.inf, device=roles.device, dtype=roles.dtype)
        min_scores = torch.full((b, r, r, o, o), torch.inf, device=roles.device, dtype=roles.dtype)

        for start in range(0, k, chunk_size):
            end = min(start + chunk_size, k)
            if self.training and roles.requires_grad:
                chunk_max, chunk_min = checkpoint(
                    lambda input_roles, chunk_start=start, chunk_end=end: self._chunk_extrema(
                        input_roles, chunk_start, chunk_end
                    ),
                    roles,
                    use_reentrant=True,
                )
            else:
                chunk_max, chunk_min = self._chunk_extrema(roles, start, end)

            max_scores = torch.maximum(max_scores, chunk_max)
            min_scores = torch.minimum(min_scores, chunk_min)

        # combine min and max aggregation results
        scores = torch.cat([max_scores, min_scores], dim=1)

        #reshape to (b, 2*r*r, o, o)
        b, r1, r2, o, _ = scores.shape

        scores = scores.reshape(b, 2 * r * r, o, o)  # [b, 2*r*r, o, o]
        # append transposed roles
        roles_T = roles.transpose(2, 3)
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
        self.initial_ffn = getattr(self.config, "INITIAL_FFN", False)
        if self.initial_ffn:
            self.initial_concept_ffn = CFFN(
                in_concepts,
                self.config.NUM_HIDDEN_CONCEPTS,
                self.config.ACTIVATION_FUNCTION,
            )
            self.initial_role_ffn = RFFN(
                in_roles,
                self.config.NUM_HIDDEN_ROLES,
                self.config.ACTIVATION_FUNCTION,
            )
        self.layers = nn.ModuleList()
        for i in range(self.config.NUM_LAYERS):
            _in_concepts = (
                self.config.NUM_HIDDEN_CONCEPTS
                if self.initial_ffn or i > 0
                else in_concepts
            )
            _in_roles = (
                self.config.NUM_HIDDEN_ROLES if self.initial_ffn or i > 0 else in_roles
            )
            _out_concepts = out_concepts if i == self.config.NUM_LAYERS - 1 else self.config.NUM_HIDDEN_CONCEPTS
            _out_roles = out_roles if i == self.config.NUM_LAYERS - 1 else self.config.NUM_HIDDEN_ROLES
            # don't apply activation function on last layer (addapt if multilayer MLPs are used in the future)
            _activation_function = self.config.ACTIVATION_FUNCTION if i < self.config.NUM_LAYERS - 1 else nn.Identity()
            if self.config.INPUT_RESIDUAL:
                if i >= 1:
                    _in_concepts += in_concepts
                    _in_roles += in_roles
                           
            if self.config.RESIDUAL:
                if i == 1:
                    _in_concepts += in_concepts
                    _in_roles += in_roles
                elif i > 1:
                    _in_concepts += self.config.NUM_HIDDEN_CONCEPTS
                    _in_roles += self.config.NUM_HIDDEN_ROLES

            # print("Layer", i, "in_concepts", _in_concepts, "in_roles", _in_roles, "out_concepts", _out_concepts, "out_roles", _out_roles)
            if config.MODE == "relaxed":
                self.layers.append(Relaxed_Layer(_in_concepts, _in_roles, _out_concepts, _out_roles, _activation_function, config))
            elif config.MODE == "strict":
                self.layers.append(Strict_Layer(_in_concepts, _in_roles, _out_concepts, _out_roles, _activation_function, config))
            else:
                raise ValueError("Invalid MODE in config. Expected 'relaxed' or 'strict', got: {}".format(config.MODE))
            
    def forward(self, concepts, roles):
        in_concepts = concepts
        in_roles = roles
        if self.initial_ffn:
            in_concepts = self.initial_concept_ffn(in_concepts)
            in_roles = self.initial_role_ffn(in_roles)
        # apply layers sequentially
        for i, layer in enumerate(self.layers):
            previous_concepts_intermediate = in_concepts
            previous_roles_intermediate = in_roles
            
            # Concatenates input concepts and roles to the input of the next layer
            if i>0 and self.config.INPUT_RESIDUAL:
                in_concepts = torch.cat([in_concepts, concepts], dim=1)
                in_roles = torch.cat([in_roles, roles], dim=1)
            # Concatenates the output of the previous layer to the input of the next layer
            if i>0 and self.config.RESIDUAL:
                in_concepts = torch.cat([in_concepts, previous_concepts], dim=1)
                in_roles = torch.cat([in_roles, previous_roles], dim=1)
            
            #stores output of the previous layer to be used in the next layer
            previous_concepts = previous_concepts_intermediate
            previous_roles = previous_roles_intermediate
            in_concepts, in_roles = layer(in_concepts, in_roles)
        return in_concepts, in_roles

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
        self.RRA = RRMinMaxAttention(
            chunk_bytes=getattr(config, "STRICT_RR_CHUNK_BYTES", 256 * 2**20)
        )
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
    




class NLM_adapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.input_concept_projection = CFFN(
            config.IN_CONCEPTS,
            config.NUM_HIDDEN_CONCEPTS,
            config.ACTIVATION_FUNCTION,
        )
        self.input_role_projection = RFFN(
            config.IN_ROLES + 1,
            config.NUM_HIDDEN_ROLES,
            config.ACTIVATION_FUNCTION,
        )

        model_config = copy.copy(config)
        model_config.INITIAL_FFN = False
        
        self.model = MultiLayerNDLM(
            config.NUM_HIDDEN_CONCEPTS,
            config.NUM_HIDDEN_ROLES,
            config.OUT_CONCEPTS,
            config.OUT_ROLES,
            model_config
        )
        
    def forward(self, C,R):
        if C is None:
            C = torch.zeros(R.size(0), R.size(1), 0, device=R.device)
        if R is None:
            R = torch.zeros(C.size(0), C.size(1), C.size(1), 0, device=C.device)
        C = C.permute(0, 2, 1)        # [B, C, N]
        R = R.permute(0, 3, 1, 2)     # [B, R, N, N]
        #add identity role to R
        R= torch.cat([R, torch.eye(R.size(2), device=R.device).unsqueeze(0).unsqueeze(0).expand(R.size(0), 1, R.size(2), R.size(2))], dim=1)

        C = self.input_concept_projection(C)
        R = self.input_role_projection(R)

        out_C, out_R = self.model(C,R)
        
        if self.config.OUT_ROLES>=1:
            out_R = out_R.permute(0, 2, 3, 1)   # [B, N, N, R]
            return out_R
        elif self.config.OUT_CONCEPTS>=1:
            out_C = out_C.permute(0, 2, 1)      # [B, N, C]
            return out_C
        else:
            raise ValueError("At least one of OUT_CONCEPTS or OUT_ROLES must be >=1 in the config.")
