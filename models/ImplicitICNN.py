import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvexQuadratic(nn.Module):
    '''Convex Quadratic Layer'''
    __constants__ = ['in_features', 'out_features', 'quadratic_decomposed', 'weight', 'bias']

    def __init__(self, in_features, out_features, bias=True, rank=1):
        super(ConvexQuadratic, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        self.quadratic_decomposed = nn.Parameter(torch.Tensor(
            torch.randn(in_features, rank, out_features, dtype=torch.float32)
        ))
        self.weight = nn.Parameter(torch.Tensor(
            torch.randn(out_features, in_features, dtype=torch.float32)
        ))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features, dtype=torch.float32))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        quad = ((input.matmul(self.quadratic_decomposed.transpose(1,0)).transpose(1, 0)) ** 2).sum(dim=1)
        linear = torch.nn.functional.linear(input, self.weight, self.bias)
        return quad + linear    

class DenseICNN(nn.Module):
    '''Fully Conncted ICNN with input-quadratic skip connections.'''
    def __init__(
        self, dim, 
        hidden_layer_sizes=[32, 32, 32],
        rank=1, activation='celu',
        strong_convexity=1e-6,
        batch_size=20000,
        weights_init_std=0.1,
    ):
        super(DenseICNN, self).__init__()
        
        self.dim = dim
        self.strong_convexity = strong_convexity
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.rank = rank
        self.batch_size = batch_size
        
        self.quadratic_layers = nn.ModuleList([
            ConvexQuadratic(dim, out_features, rank=rank, bias=True)
            for out_features in hidden_layer_sizes
        ])
        
        sizes = zip(hidden_layer_sizes[:-1], hidden_layer_sizes[1:])
        self.convex_layers = nn.ModuleList([
            nn.Linear(in_features, out_features, bias=False)
            for (in_features, out_features) in sizes
        ])
        
        # self.final_layer = nn.Linear(hidden_layer_sizes[-1], 1, bias=False)
        self.final_layer = nn.Linear(hidden_layer_sizes[-1], 1, bias=False).to(dtype=torch.float32)
        for layer in self.convex_layers:
            layer.to(dtype=torch.float32)
        for layer in self.quadratic_layers:
            layer.to(dtype=torch.float32)
            
        self._init_weights(weights_init_std)
        
    def _init_weights(self, std):
        for p in self.parameters():
            p.data = (torch.randn(p.shape, dtype=torch.float32) * std).to(p)   

    def forward(self, input):
        '''Evaluation of the discriminator value. Preserves the computational graph.'''
        input = input.float()
        output = self.quadratic_layers[0](input)
        for quadratic_layer, convex_layer in zip(self.quadratic_layers[1:], self.convex_layers):
            output = convex_layer(output) + quadratic_layer(input)
            if self.activation == 'celu':
                output = torch.celu(output)
            elif self.activation == 'softplus':
                output = F.softplus(output)
            elif self.activation == 'tanh':
                output = torch.tanh(output)
            elif self.activation == 'relu':
                output = F.relu(output)
            else:
                raise Exception('Activation is not specified or unknown.')
        
        return self.final_layer(output) + .5 * self.strong_convexity * (input ** 2).sum(dim=1).reshape(-1, 1)
    
    def push(self, input, create_graph=True, retain_graph=True):
        '''
        Pushes input by using the gradient of the network. By default preserves the computational graph.
        Apply to small batches.
        '''
        assert len(input) <= self.batch_size
        output = torch.autograd.grad(
            outputs=self.forward(input), inputs=input,
            create_graph=create_graph, retain_graph=retain_graph,
            only_inputs=True,
            grad_outputs=torch.ones_like(input[:, :1], requires_grad=False)
        )[0]
        return output
    
    def push_nograd(self, input):
        '''
        Pushes input by using the gradient of the network. Does not preserve the computational graph.
        Use for pushing large batches (the function uses minibatches).
        '''
        output = torch.zeros_like(input, requires_grad=False)
        for i in range(0, len(input), self.batch_size):
            input_batch = input[i:i+self.batch_size]
            output.data[i:i+self.batch_size] = self.push(
                input[i:i+self.batch_size],
                create_graph=False, retain_graph=False
            ).data
        return output  
    
    def convexify(self):
        for layer in self.convex_layers:
            if (isinstance(layer, nn.Linear)):
                layer.weight.data.clamp_(0)
        self.final_layer.weight.data.clamp_(0)
        
    # def convexify(self):
    #     for layer in self.convex_layers:
    #         if isinstance(layer, nn.Linear):
    #             # Clamp only the weights except the last input dimension
    #             layer.weight.data[:, :-1].clamp_(0)
    #     self.final_layer.weight.data.clamp_(0)

    def get_stepsize(self, eps: float = 1e-8, safety: float = 1.0) -> float:
        return float(safety / (float(self.Lgrad.item()) + eps))
    
    def update_Lgrad(self, x: torch.Tensor, n_power_iter: int = 20, eps: float = 1e-12, take_max_over_batch: bool = True) -> torch.Tensor:
        """
        Estimate the Lipschitz constant of ∇g via power iteration.
        """
        x = x.detach().requires_grad_(True)
        
        # define g function
        def g_func(x_in):
            return self.forward(x_in).squeeze(-1)
        
        v = torch.randn_like(x)
        v = v / (v.norm(dim=1, keepdim=True) + eps)
        
        for _ in range(n_power_iter):
            y = g_func(x)
            g = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
            gv = (g * v).sum()
            Hv = torch.autograd.grad(gv, x, retain_graph=True)[0]
            Hv_norm = Hv.norm(dim=1, keepdim=True) + eps
            v = Hv / Hv_norm
        
        # final estimate
        y = g_func(x)
        g = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
        gv = (g * v).sum()
        Hv = torch.autograd.grad(gv, x, retain_graph=False)[0]
        sigma = Hv.norm(dim=1)
        est = sigma.max() if take_max_over_batch else sigma.mean()
        
        # store in buffer
        if not hasattr(self, 'Lgrad'):
            self.register_buffer('Lgrad', torch.tensor(0.0, dtype=x.dtype, device=x.device))
        self.Lgrad.copy_(est.detach())
        return est.detach()


class ImplicitICNN(nn.Module):
    def __init__(self, dim, 
        hidden_layer_sizes=[32, 32, 32],
        rank=1, activation='celu',
        strong_convexity=1e-6,
        batch_size=20000,
        weights_init_std=0.1,
        ):
        super().__init__()
        self.g_net = DenseICNN(dim=dim, hidden_layer_sizes=hidden_layer_sizes,
                                rank=rank, activation=activation,
                                strong_convexity=strong_convexity,
                                batch_size=batch_size,
                                weights_init_std=weights_init_std)

    def grad_g(self, x):
        # return self.g_net.push(x) 
        with torch.enable_grad():
            x = x.requires_grad_(True)
            g_x = self.g_net(x)
            grad = torch.autograd.grad(outputs=g_x, inputs=x, grad_outputs=torch.ones_like(g_x), retain_graph=True)[0]
        return grad

    def T(self, x, z, alpha):
        grad_gx = self.grad_g(x)
        gradient_update = x - alpha * (x - z + grad_gx)
        assert gradient_update.shape == x.shape
        return gradient_update

    def forward(self, z, tol=1e-2, max_iter=500, verbose=False):
        y = z #torch.zeros_like(z)
        ynew = y
        self.g_net.update_Lgrad(ynew, n_power_iter=20)
        alpha = self.g_net.get_stepsize(safety=1e-1)
        with torch.no_grad():
            for iter in range(1, max_iter+1):
                yold = ynew
                ynew = self.T(yold, z, alpha)
                grad_gy = self.grad_g(ynew)
                grad_norm = torch.norm((ynew - z + grad_gy), dim=1).max().item()
                if verbose:
                    print(f"iter: {iter:3d}  grad_norm: {grad_norm:.3e}")
                if grad_norm < tol:
                    depth = iter
                    break
        if iter == max_iter:
            depth = max_iter
            # print('Warning: ImplicitNet forward did not converge within max_iter')
        if not self.training:
            output = ynew
            return output, depth, grad_norm
        else:
            output = self.T(ynew, z, alpha)
            assert output.requires_grad
            return output, depth, grad_norm
    
    def convexify(self):
        self.g_net.convexify()
