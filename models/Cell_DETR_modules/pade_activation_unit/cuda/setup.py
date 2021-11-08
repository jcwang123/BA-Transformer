import airspeed
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# degrees
coefficients = [(3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (5, 4)]


def generate_cpp_module(fname='pau_cuda.cpp', coefficients=coefficients):
    file_content = airspeed.Template("""
\#include <torch/extension.h>
\#include <vector>
\#include <iostream>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#foreach ($coef in $coefficients)
at::Tensor pau_cuda_forward_$coef[0]_$coef[1](torch::Tensor x, torch::Tensor n, torch::Tensor d);
std::vector<torch::Tensor> pau_cuda_backward_$coef[0]_$coef[1](torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);
#end

#foreach ($coef in $coefficients)
at::Tensor pau_forward__$coef[0]_$coef[1](torch::Tensor x, torch::Tensor n, torch::Tensor d) {
    CHECK_INPUT(x);
    CHECK_INPUT(n);
    CHECK_INPUT(d);

    return pau_cuda_forward_$coef[0]_$coef[1](x, n, d);
}
std::vector<torch::Tensor> pau_backward__$coef[0]_$coef[1](torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(x);
    CHECK_INPUT(n);
    CHECK_INPUT(d);

    return pau_cuda_backward_$coef[0]_$coef[1](grad_output, x, n, d);
}
#end

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#foreach ($coef in $coefficients)
    m.def("forward_$coef[0]_$coef[1]", &pau_forward__$coef[0]_$coef[1], "PAU forward _$coef[0]_$coef[1]");
    m.def("backward_$coef[0]_$coef[1]", &pau_backward__$coef[0]_$coef[1], "PAU backward _$coef[0]_$coef[1]");
#end
}
    """)

    content = file_content.merge(locals())

    with open(fname, "w") as text_file:
        text_file.write(content)


def generate_cpp_kernels_module(fname='pau_cuda_kernels.cu', coefficients=coefficients):
    coefficients = [[c[0], c[1], max(c[0], c[1])] for c in coefficients]

    file_content = airspeed.Template("""
\#include <torch/extension.h>
\#include <ATen/cuda/CUDAContext.h>
\#include <cuda.h>
\#include <cuda_runtime.h>
\#include <vector>
\#include <stdlib.h>


constexpr uint32_t THREADS_PER_BLOCK = 512;


#foreach ($coef in $coefficients)
template <typename scalar_t>
__global__ void pau_cuda_forward_kernel_$coef[0]_$coef[1]( const scalar_t* __restrict__ x, const scalar_t* __restrict__ n,
    const scalar_t* __restrict__ d, scalar_t* __restrict__ result, size_t x_size) {


    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
        index < x_size;
        index += blockDim.x * gridDim.x){

        scalar_t xp1 = x[index];
        scalar_t axp1 = abs(xp1);

        #foreach( $idx in [2..$coef[2]] )#set( $value = $idx - 1 )

        scalar_t xp$idx = xp$value * xp1;
        scalar_t axp$idx = abs(xp$idx);
        #end

        #foreach( $idx in [0..$coef[0]] )
        scalar_t n_$idx = n[$idx];
        #end

        #foreach( $idx in [0..$coef[1]] )
        scalar_t d_$idx = d[$idx];
        scalar_t ad_$idx = abs(d_$idx);
        #end

        scalar_t P = n_0
        #foreach( $idx in [1..$coef[0]] )
        + xp$idx*n_$idx
        #end
        ;

        scalar_t Q = scalar_t(1.0)
        #foreach( $idx in [1..$coef[1]] )#set( $value = $idx - 1 )
        + axp$idx*ad_$value
        #end
        ;

        result[index] = P/Q;
    }
}

at::Tensor pau_cuda_forward_$coef[0]_$coef[1](torch::Tensor x, torch::Tensor n, torch::Tensor d){
    auto result = at::empty_like(x);
    const auto x_size = x.numel();

    int blockSize = THREADS_PER_BLOCK;
    int numBlocks = (x_size + blockSize - 1) / blockSize;

    AT_DISPATCH_FLOATING_TYPES(x.type(), "pau_cuda_forward_$coef[0]_$coef[1]", ([&] {
    pau_cuda_forward_kernel_$coef[0]_$coef[1]<scalar_t>
        <<<numBlocks, blockSize>>>(
            x.data<scalar_t>(),
            n.data<scalar_t>(),
            d.data<scalar_t>(),
            result.data<scalar_t>(),
            x_size);
        }));

    return result;
}


template <typename scalar_t>
__global__ void pau_cuda_backward_kernel_$coef[0]_$coef[1](
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ n,
    const scalar_t* __restrict__ d,
    scalar_t* __restrict__ d_x,
    double* __restrict__ d_n,
    double* __restrict__ d_d,
    size_t x_size) {

    __shared__ double sdd[$coef[0]];
    __shared__ double sdn[$coef[1]];


    if( threadIdx.x == 0){
        #foreach( $idx in [0..$coef[0]] )
        sdn[$idx] = 0;
        #end
        #set( $value = $coef[1] - 1 )
        #foreach( $idx in [0..$value] )
        sdd[$idx] = 0;
        #end
    }

    __syncthreads();
    #foreach( $idx in [0..$coef[0]] )
    scalar_t d_n$idx = 0;
    #end
    #set( $value = $coef[1] - 1 )
    #foreach( $idx in [0..$value] )
    scalar_t d_d$idx = 0;
    #end


    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < x_size;
         index += blockDim.x * gridDim.x)
      {

        scalar_t xp1 = x[index];
        scalar_t axp1 = abs(xp1);

        #foreach( $idx in [2..$coef[2]] )#set( $value = $idx - 1 )

        scalar_t xp$idx = xp$value * xp1;
        scalar_t axp$idx = abs(xp$idx);
        #end

        #foreach( $idx in [0..$coef[0]] )
        scalar_t n_$idx = n[$idx];
        #end

        #foreach( $idx in [0..$coef[1]] )
        scalar_t d_$idx = d[$idx];
        scalar_t ad_$idx = abs(d_$idx);
        #end

        scalar_t P = n_0
        #foreach( $idx in [1..$coef[0]] )
        + xp$idx*n_$idx
        #end
        ;

        scalar_t Q = scalar_t(1.0)
        #foreach( $idx in [1..$coef[1]] )#set( $value = $idx - 1 )
        + axp$idx*ad_$value
        #end
        ;

        scalar_t R = n_1
        #set( $value = $coef[0] - 1 )
        #foreach( $idx in [1..$value] )#set( $value2 = $idx + 1 )
        + scalar_t($value2.0)*n_$value2*xp$idx
        #end
        ;
        scalar_t S = copysign( scalar_t(1.0), xp1 ) * (ad_0 

        #foreach( $idx in [2..$coef[1]] )#set( $value = $idx - 1 )
        + scalar_t($idx.0)*ad_$value*axp$value
        #end
         );

        scalar_t mpq2 = -P/(Q*Q);

        scalar_t grad_o = grad_output[index];

        scalar_t d_i_x = (R/Q + S*mpq2); 
        d_x[index] = d_i_x * grad_o;


        #foreach( $idx in [1..$coef[1]] )#set( $value = $idx - 1 )
        scalar_t d_i_d$value = (mpq2*axp$idx*copysign( scalar_t(1.0), d_$value ));
        d_d$value += d_i_d$value * grad_o;
        #end


        scalar_t d_i_n0 = scalar_t(1.0)/Q; 
        d_n0 += d_i_n0 * grad_o;

        #foreach( $idx in [1..$coef[0]] )#set( $value = $idx - 1 )
        scalar_t d_i_n$idx  = xp$idx/Q;
        d_n$idx += d_i_n$idx * grad_o;
        #end

    }

    #foreach( $idx in [0..$coef[0]] )
    atomicAdd(&sdn[$idx], d_n$idx);
    #end
    #set( $value = $coef[1] - 1 )
    #foreach( $idx in [0..$value] )
    atomicAdd(&sdd[$idx], d_d$idx);
    #end


    __syncthreads();

    if( threadIdx.x == 0){
        #foreach( $idx in [0..$coef[0]] )
        atomicAdd(&d_n[$idx], sdn[$idx]);
        #end
        #set( $value = $coef[1] - 1 )
        #foreach( $idx in [0..$value] )
        atomicAdd(&d_d[$idx], sdd[$idx]);
        #end

    }


}

std::vector<torch::Tensor> pau_cuda_backward_$coef[0]_$coef[1](torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d){
    const auto x_size = x.numel();
    auto d_x = at::empty_like(x);
    auto d_n = at::zeros_like(n).toType(at::kDouble);
    auto d_d = at::zeros_like(d).toType(at::kDouble);

    int blockSize = THREADS_PER_BLOCK;

    AT_DISPATCH_FLOATING_TYPES(x.type(), "pau_cuda_backward_$coef[0]_$coef[1]", ([&] {
    pau_cuda_backward_kernel_$coef[0]_$coef[1]<scalar_t>
        <<<16, blockSize>>>(
            grad_output.data<scalar_t>(),
            x.data<scalar_t>(),
            n.data<scalar_t>(),
            d.data<scalar_t>(),
            d_x.data<scalar_t>(),
            d_n.data<double>(),
            d_d.data<double>(),
            x_size);
    }));

    return {d_x, d_n.toType(at::kFloat), d_d.toType(at::kFloat)};
}
#end
        """)

    content = file_content.merge(locals())

    with open(fname, "w") as text_file:
        text_file.write(content)


generate_cpp_module(fname='pau_cuda.cpp')
generate_cpp_kernels_module(fname='pau_cuda_kernels.cu')

setup(
    name='pau',
    version='0.0.2',
    ext_modules=[
        CUDAExtension('pau_cuda', [
            'pau_cuda.cpp',
            'pau_cuda_kernels.cu',
        ],
                      extra_compile_args={'cxx': [],
                                          'nvcc': ['-gencode=arch=compute_60,code="sm_60,compute_60"', '-lineinfo',
                                                   "-ccbin=gcc-6.3.0"]}
                      ),
        # CUDAExtension('pau_cuda_unrestricted', [
        #    'pau_cuda_unrestricted.cpp',
        #    'pau_cuda_kernels_unrestricted.cu',
        # ],
        #              extra_compile_args={'cxx': [],
        #                                  'nvcc': ['-gencode=arch=compute_60,code="sm_60,compute_60"', '-lineinfo']}
        #              )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
