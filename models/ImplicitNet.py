import torch
import torch.nn as nn
import torch.nn.functional as F

class ScalarNet(nn.Module):
    """
    Simple scalar-valued network (no convexity constraint).
    Uses fully-connected layers with a configurable activation between them.
    """
    def __init__(self, input_dim: int, hidden_structure=(64, 64), activation: str = 'softplus', r = 25):
        super().__init__()
        hidden = list(hidden_structure)
        if len(hidden) < 1:
            raise ValueError('hidden must contain at least one layer width')

        self.hidden_dim = hidden[0]
        self.input_dim = input_dim
        self.rankA = r # rank of quadratic matrix

        # parameters for quadratic portion
        self.A  = nn.Parameter(torch.zeros(self.rankA, input_dim) , requires_grad=True)
        self.A  = nn.init.xavier_uniform_(self.A)
        self.c  = nn.Linear( input_dim  , 1  , bias=True)  # b'*[x;t] + c
        self.w  = nn.Linear( self.hidden_dim    , 1  , bias=False)

        self.h = 1 / len(hidden) # "scalar to multiply each resnet update"

        # activation functions
        if activation == 'tanh':
            self.phi = torch.tanh
        elif activation == 'relu':
            self.phi = F.relu
        elif activation == 'gelu':
            self.phi = F.gelu
        elif activation == 'elu':
            self.phi = F.elu
        elif activation == 'softplus':
            self.phi = F.softplus
        else:
            raise ValueError('Unsupported activation')

        self.opening_layer = nn.Linear(self.input_dim, self.hidden_dim)
        resnet_widths = hidden_structure
        self.layers = nn.ModuleList([nn.Linear(resnet_widths[i], resnet_widths[i+1]) for i in range(len(resnet_widths)-1)])
        self.register_buffer('Lgrad', torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # nonlinear portion
        y = self.phi(self.opening_layer(x))
        for layer in self.layers[:-1]:
            y = y + self.h*self.phi(layer(y))

        nonlinear_term = self.w(y)
        
        # quadratic portion 
        # force A to be symmetric
        symA = torch.matmul(torch.t(self.A), self.A) # A'A
        quadratic_term =  0.5 * torch.sum( torch.matmul(x , symA) * x , dim=1, keepdims=True) + self.c(x)

        return quadratic_term + nonlinear_term

    @torch.no_grad()
    def get_stepsize(self, eps: float = 1e-8, safety: float = 1.0) -> float:
        return float(safety / (float(self.Lgrad.item()) + eps))

    def update_Lgrad(self, x: torch.Tensor, n_power_iter: int = 20, eps: float = 1e-12, take_max_over_batch: bool = True) -> torch.Tensor:
        x = x.detach().requires_grad_(True)
        def f_scalar(x_in: torch.Tensor) -> torch.Tensor:
            return self.forward(x_in).squeeze(-1)
        v = torch.randn_like(x)
        v = v / (v.norm(dim=1, keepdim=True) + eps)
        for _ in range(n_power_iter):
            y = f_scalar(x)
            g = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
            gv = (g * v).sum()
            Hv = torch.autograd.grad(gv, x, retain_graph=True)[0]
            Hv_norm = Hv.norm(dim=1, keepdim=True) + eps
            v = Hv / Hv_norm
        y = f_scalar(x)
        g = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
        gv = (g * v).sum()
        Hv = torch.autograd.grad(gv, x, retain_graph=False)[0]
        sigma = Hv.norm(dim=1)
        est = sigma.max() if take_max_over_batch else sigma.mean()
        self.Lgrad.copy_(est.detach())
        return est.detach()
    


class ImplicitNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 10, hidden_structure=(64, 64,), activation: str = 'tanh'):
        super().__init__()
        self.input_dim = input_dim        # latent dim (x only)
        self.num_classes = num_classes
        # ScalarNet takes [x; c_onehot] as input
        self.g_net = ScalarNet(input_dim=input_dim + num_classes, hidden_structure=hidden_structure, activation=activation)

    def grad_g(self, x, c_onehot, create_graph=False):
        """Compute ∇_x g(x, c) — gradient only w.r.t. x, not c."""
        with torch.enable_grad():
            x = x.requires_grad_(True)
            xc = torch.cat([x, c_onehot], dim=1)
            g_xc = self.g_net(xc)
            grad = torch.autograd.grad(outputs=g_xc, inputs=x, grad_outputs=torch.ones_like(g_xc),
                                       retain_graph=True, create_graph=create_graph)[0]
        return grad

    def T(self, x, z, alpha, c_onehot):
        grad_gx = self.grad_g(x, c_onehot)
        gradient_update = x - alpha * (x - z + grad_gx)
        assert gradient_update.shape == x.shape
        return gradient_update

    def forward(self, z, c_onehot, tol=1e-2, max_iter=500, verbose=False, y_init=None):
        if y_init is not None and y_init.shape == z.shape:
            ynew = y_init.detach().clone()
        else:
            ynew = torch.zeros_like(z)
        # Estimate Lipschitz constant using [y; c] input
        yc = torch.cat([ynew, c_onehot], dim=1)
        self.g_net.update_Lgrad(yc, n_power_iter=20)

        # choose alpha to be minimum between self.get_stepsize() and 1e-2 for stability
        alpha = self.g_net.get_stepsize(safety=1e-1)
        alpha = min(alpha, 1e-2)
        with torch.no_grad():
            for iter in range(1, max_iter+1):
                yold = ynew
                ynew = self.T(yold, z, alpha, c_onehot)
                grad_gy = self.grad_g(ynew, c_onehot)
                grad_norm = torch.norm((ynew - z + grad_gy), dim=1).max().item()
                if verbose:
                    print(f"iter: {iter:3d}  grad_norm: {grad_norm:.3e}")
                if grad_norm < tol:
                    depth = iter
                    break
        if iter == max_iter:
            depth = max_iter
        if not self.training:
            output = ynew
            return output, depth, grad_norm
        else:
            output = self.T(ynew, z, alpha, c_onehot)
            assert output.requires_grad
            return output, depth, grad_norm