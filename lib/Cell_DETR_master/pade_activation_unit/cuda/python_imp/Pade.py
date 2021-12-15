from time import time

import numpy as np
import torch
import torch.nn as nn
from numpy.random.mtrand import RandomState


def get_constants_for_inits(name, seed=17):
    # (numerator: [x, x.pow(1), x.pow(2), x.pow(3), x.pow(4, x.pow(5)], denominator: (x, x.pow(2), center)

    if name == "pade_sigmoid_3":
        return ((1 / 2, 1 / 4, 1 / 20, 1 / 240),
                (0., 1 / 10),
                (0,))
    elif name == "pade_sigmoid_5":
        return ((1 / 2, 1 / 4, 17 / 336, 1 / 224, 0, - 1 / 40320),
                (0., 1 / 10),
                (0,))
    elif name == "pade_softplus":
        return ((np.log(2), 1 / 2, (15 + 8 * np.log(2)) / 120, 1 / 30, 1 / 320),
                (0.01, 1 / 15),
                (0,))
    elif name == "pade_optimized_avg":
        return [(0.15775171, 0.74704865, 0.82560348, 1.61369449, 0.6371632, 0.10474671),
                (0.38940287, 2.19787666, 0.30977883, 0.15976778),
                (0.,)]
    elif name == "pade_optimized_leakyrelu":
        return [(3.35583603e-02, 5.05000375e-01, 1.65343934e+00, 2.01001052e+00, 9.31901999e-01, 1.52424124e-01),
                (3.30847488e-06, 3.98021568e+00, 5.12471206e-07, 3.01830109e-01),
                (0,)]
    elif name == "pade_optimized_leakyrelu2":
        return [(0.1494, 0.8779, 1.8259, 2.4658, 1.6976, 0.4414),
                (0.0878, 3.3983, 0.0055, 0.3488),
                (0,)]
    elif name == "pade_random":
        rng = RandomState(seed)
        return (rng.standard_normal(5), rng.standard_normal(4), (0,))
    elif name == "pade_optmized":
        return [(0.0034586860882628158, -0.41459839329894876, 4.562452712166459, -16.314813244428276,
                 18.091669531543833, 0.23550876048241304),
                (3.0849791873233383e-28, 3.2072596311394997e-27, 1.0781647589819156e-28, 11.493453196161223),
                (0,)]


class PADEACTIVATION(nn.Module):

    def __init__(self, init_coefficients="pade_optimized_leakyrelu"):
        super(PADEACTIVATION, self).__init__()
        constants_for_inits = get_constants_for_inits(init_coefficients)

        self.n_numerator = len(constants_for_inits[0])
        self.n_denominator = len(constants_for_inits[1])

        self.weight_numerator = nn.Parameter(torch.FloatTensor(constants_for_inits[0]), requires_grad=True)
        self.weight_denominator = nn.Parameter(torch.FloatTensor(constants_for_inits[1]), requires_grad=True)

    def forward(self, x):
        raise NotImplementedError()


class PADEACTIVATION_Function_based(PADEACTIVATION):

    def __init__(self, init_coefficients="pade_optimized_leakyrelu", act_func_cls=None):
        super(PADEACTIVATION_Function_based, self).__init__(init_coefficients=init_coefficients)

        if act_func_cls is None:
            act_func_cls = PADEACTIVATION_F_python

        self.activation_function = act_func_cls.apply

    def forward(self, x):
        out = self.activation_function(x, self.weight_numerator, self.weight_denominator)
        return out


class PADEACTIVATION_F_abs_cpp(torch.autograd.Function):
    forward_f = None
    backward_f = None
    alpha = 0.1

    @classmethod
    def config_cuda(cls, num, den, alpha):
        cls.alpha = alpha

        if num == 5 and den == 4:
            from pau_cuda import forward_5_4 as pau_forward_cuda
            from pau_cuda import backward_5_4 as pau_backward_cuda

        elif num == 4 and den == 4:
            from pau_cuda import forward_4_4 as pau_forward_cuda
            from pau_cuda import backward_4_4 as pau_backward_cuda
        elif num == 5 and den == 5:
            from pau_cuda import forward_5_5 as pau_forward_cuda
            from pau_cuda import backward_5_5 as pau_backward_cuda
        else:
            raise ValueError("not implemented")

        cls.forward_f = pau_forward_cuda
        cls.backward_f = pau_backward_cuda

    @staticmethod
    def forward(ctx, input, weight_numerator, weight_denominator):
        """import pickle
        with open("data.pt", "wb") as file:
            pickle.dump({"x": input.detach().cpu(),
                     "weight_numerator": weight_numerator.detach().cpu(),
                     "weight_denominator": weight_numerator.detach().cpu()}, file)"""
        ctx.save_for_backward(input, weight_numerator, weight_denominator)

        x = PADEACTIVATION_F_abs_cpp.forward_f(input, weight_numerator, weight_denominator)

        return x

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous():  # TODO this check is necessary if efficientnet is used
            grad_output = grad_output.contiguous()
        x, weight_numerator, weight_denominator = ctx.saved_tensors
        d_x, d_weight_numerator, d_weight_denominator = PADEACTIVATION_F_abs_cpp.backward_f(grad_output, x, weight_numerator,
                                                                              weight_denominator)

        return d_x, d_weight_numerator, d_weight_denominator


class PADEACTIVATION_F_cpp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight_numerator, weight_denominator):
        ctx.save_for_backward(input, weight_numerator, weight_denominator)
        x = pau_forward_cuda(input, weight_numerator, weight_denominator)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, weight_numerator, weight_denominator = ctx.saved_tensors
        d_x, d_weight_numerator, d_weight_denominator = pau_backward_cuda(grad_output, x, weight_numerator,
                                                                          weight_denominator)
        return d_x, d_weight_numerator, d_weight_denominator


class PADEACTIVATION_F_python(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight_numerator, weight_denominator):
        ctx.save_for_backward(input, weight_numerator, weight_denominator)

        z = input

        clamped_n = weight_numerator
        clamped_d = weight_denominator.abs()

        numerator = z.mul(clamped_n[1]) + clamped_n[0]
        xps = list()
        # xp = z
        xps.append(z)
        for c_n in clamped_n[2:]:
            xp = xps[-1].mul(z)
            xps.append(xp)
            numerator = numerator + c_n.mul(xp)

        denominator = z.abs() * clamped_d[0] + 1
        for idx, c_d in enumerate(clamped_d[1:]):
            xp = xps[idx + 1].abs()
            denominator = denominator + c_d.mul(xp)

        return numerator.div(denominator)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight_numerator, weight_denominator = ctx.saved_tensors

        clamped_n = weight_numerator  # .clamp(min=0, max=1.)
        clamped_d = weight_denominator.abs()
        numerator = x.mul(clamped_n[1]) + clamped_n[0]
        xps = list()
        # xp = z
        xps.append(x)
        for c_n in clamped_n[2:]:
            xp = xps[-1].mul(x)
            xps.append(xp)
            numerator = numerator + c_n.mul(xp)

        denominator = x.abs() * clamped_d[0] + 1
        for idx, c_d in enumerate(clamped_d[1:]):
            xp = xps[idx + 1].abs()
            denominator = denominator + c_d.mul(xp)

        xps = torch.stack(xps)
        P = numerator
        Q = denominator
        dfdn = torch.cat(((1.0 / Q).unsqueeze(dim=0), xps.div(Q)))

        dfdd_tmp = (-P.div((Q.mul(Q))))
        dfdd = dfdd_tmp.mul(xps[0:clamped_d.size()[0]].abs())

        for idx in range(dfdd.shape[0]):
            dfdd[idx] = dfdd[idx].mul(weight_denominator[idx].sign())

        dfdx1 = 2.0 * clamped_n[2].mul(xps[0]) + clamped_n[1]
        for idx, xp in enumerate(xps[1:clamped_n.size()[0] - 2]):
            i = (idx + 3)
            dfdx1 += i * clamped_n[i].mul(xp)
        dfdx1 = dfdx1.div(Q)

        dfdx2 = 2.0 * clamped_d[1].mul(xps[0].abs()) + clamped_d[0]
        for idx, xp in enumerate(xps[1:clamped_d.size()[0] - 1]):
            i = (idx + 3)
            dfdx2 += i * clamped_d[idx + 2].mul(xp.abs())
        dfdx2_ = dfdx2.mul(xps[0].sign())
        dfdx2 = dfdx2_.mul(dfdd_tmp)

        dfdx = dfdx1 + dfdx2

        rdfdn = torch.mul(grad_output, dfdn)
        rdfdd = torch.mul(grad_output, dfdd)

        dfdn = rdfdn
        dfdd = rdfdd
        for _ in range(len(P.shape)):
            dfdn = dfdn.sum(-1)
            dfdd = dfdd.sum(-1)
        dfdx = grad_output.mul(dfdx)

        return dfdx, dfdn, dfdd


def exec_act(x, actv):
    forward = 0
    backward = 0

    start = time()
    for _ in range(10000):
        new_x = actv(x)
    torch.cuda.synchronize()
    forward += time() - start

    start = time()
    for _ in range(10000):
        (new_x.sum()).backward(retain_graph=True)
    torch.cuda.synchronize()
    backward += time() - start

    print('Forward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6 / 1e5, backward * 1e6 / 1e5))
    return new_x.cpu().detach().numpy()


def test_v2():
    seed = 0
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    assert torch.cuda.is_available()
    cuda_device = torch.device("cuda")

    actv_v1 = PADEACTIVATION_Function_based().to(cuda_device)
    actv_v2 = PADEACTIVATION_Function_based(act_func_cls=PADEACTIVATION_F_cpp).to(cuda_device)

    torch.manual_seed(seed)
    x = torch.randn([64, 500], device=cuda_device) * 10

    out_v2_np = exec_act(x, actv_v2)

    out_v1_np = exec_act(x, actv_v1)

    # print(out_v1_np)
    # print("--" * 42)
    # print(out_v2_np)

    # assert np.all(np.isclose(out_v1_np, out_v2_np))


if __name__ == '__main__':
    test_v2()
