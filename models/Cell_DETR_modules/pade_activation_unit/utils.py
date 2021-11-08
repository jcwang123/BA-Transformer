import torch
from torch import nn


class Swish(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * i.sigmoid()
        ctx.save_for_backward(result, i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, i = ctx.saved_variables
        sigmoid_x = i.sigmoid()
        return grad_output * (result + sigmoid_x * (1 - result))


swish = Swish.apply


class Swish_module(nn.Module):
    def forward(self, x):
        return swish(x)


ACTIVATION_FUNCTIONS = dict({
    # pau
    "pade_optimized_leakyrelu_abs": "pade_optimized_leakyrelu_abs",
    "sigmoid": nn.Sigmoid(),
    "relu": nn.ReLU, "selu": nn.SELU, "leakyrelu": nn.LeakyReLU, "celu": nn.CELU,
    "elu": nn.ELU,
    "tanh": nn.Tanh,
    "relu6": nn.ReLU6,
    "swish": Swish_module,
    "softplus": nn.Softplus,
    "prelu": nn.PReLU,
    "rrelu": nn.RReLU})

from .cuda.python_imp.Pade import PADEACTIVATION_Function_based, PADEACTIVATION_F_cpp, \
    PADEACTIVATION_F_abs_cpp


class activationfunc():
    def __init__(self, selected_activation_func):
        self.selected_activation_func = selected_activation_func

        assert "pade" in selected_activation_func or selected_activation_func in ACTIVATION_FUNCTIONS, "unknown activation function %s" % selected_activation_func

    def get_activationfunc(self):
        if "pade" in self.selected_activation_func:
            PADEACTIVATION_F_abs_cpp.config_cuda(5, 4, 0.)
            init_coefficients = self.selected_activation_func.replace("_abs", "").replace("_cuda", "")
            if "_abs" in self.selected_activation_func:
                return PADEACTIVATION_Function_based(init_coefficients=init_coefficients,
                                                     act_func_cls=PADEACTIVATION_F_abs_cpp)
            else:
                return PADEACTIVATION_Function_based(init_coefficients=init_coefficients,
                                                     act_func_cls=PADEACTIVATION_F_cpp)
        else:
            return ACTIVATION_FUNCTIONS[self.selected_activation_func]()


def PAU():
    PADEACTIVATION_F_abs_cpp.config_cuda(5, 4, 0.)
    return PADEACTIVATION_Function_based(init_coefficients="pade_optimized_leakyrelu",
                                         act_func_cls=PADEACTIVATION_F_abs_cpp)
