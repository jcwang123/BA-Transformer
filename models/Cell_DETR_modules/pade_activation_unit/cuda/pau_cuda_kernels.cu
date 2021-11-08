
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdlib.h>


constexpr uint32_t THREADS_PER_BLOCK = 512;



template <typename scalar_t>
__global__ void pau_cuda_forward_kernel_3_3( const scalar_t* __restrict__ x, const scalar_t* __restrict__ n,
    const scalar_t* __restrict__ d, scalar_t* __restrict__ result, size_t x_size) {


    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
        index < x_size;
        index += blockDim.x * gridDim.x){

        scalar_t xp1 = x[index];
        scalar_t axp1 = abs(xp1);

        
        scalar_t xp2 = xp1 * xp1;
        scalar_t axp2 = abs(xp2);
        
        scalar_t xp3 = xp2 * xp1;
        scalar_t axp3 = abs(xp3);
        
        
        scalar_t n_0 = n[0];
        
        scalar_t n_1 = n[1];
        
        scalar_t n_2 = n[2];
        
        scalar_t n_3 = n[3];
        
        
        scalar_t d_0 = d[0];
        scalar_t ad_0 = abs(d_0);
        
        scalar_t d_1 = d[1];
        scalar_t ad_1 = abs(d_1);
        
        scalar_t d_2 = d[2];
        scalar_t ad_2 = abs(d_2);
        
        scalar_t d_3 = d[3];
        scalar_t ad_3 = abs(d_3);
        
        scalar_t P = n_0
        
        + xp1*n_1
        
        + xp2*n_2
        
        + xp3*n_3
                ;

        scalar_t Q = scalar_t(1.0)
                + axp1*ad_0
                + axp2*ad_1
                + axp3*ad_2
                ;

        result[index] = P/Q;
    }
}

at::Tensor pau_cuda_forward_3_3(torch::Tensor x, torch::Tensor n, torch::Tensor d){
    auto result = at::empty_like(x);
    const auto x_size = x.numel();

    int blockSize = THREADS_PER_BLOCK;
    int numBlocks = (x_size + blockSize - 1) / blockSize;

    AT_DISPATCH_FLOATING_TYPES(x.type(), "pau_cuda_forward_3_3", ([&] {
    pau_cuda_forward_kernel_3_3<scalar_t>
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
__global__ void pau_cuda_backward_kernel_3_3(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ n,
    const scalar_t* __restrict__ d,
    scalar_t* __restrict__ d_x,
    double* __restrict__ d_n,
    double* __restrict__ d_d,
    size_t x_size) {

    __shared__ double sdd[3];
    __shared__ double sdn[3];


    if( threadIdx.x == 0){
        
        sdn[0] = 0;
        
        sdn[1] = 0;
        
        sdn[2] = 0;
        
        sdn[3] = 0;
                        
        sdd[0] = 0;
        
        sdd[1] = 0;
        
        sdd[2] = 0;
            }

    __syncthreads();
    
    scalar_t d_n0 = 0;
    
    scalar_t d_n1 = 0;
    
    scalar_t d_n2 = 0;
    
    scalar_t d_n3 = 0;
            
    scalar_t d_d0 = 0;
    
    scalar_t d_d1 = 0;
    
    scalar_t d_d2 = 0;
    

    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < x_size;
         index += blockDim.x * gridDim.x)
      {

        scalar_t xp1 = x[index];
        scalar_t axp1 = abs(xp1);

        
        scalar_t xp2 = xp1 * xp1;
        scalar_t axp2 = abs(xp2);
        
        scalar_t xp3 = xp2 * xp1;
        scalar_t axp3 = abs(xp3);
        
        
        scalar_t n_0 = n[0];
        
        scalar_t n_1 = n[1];
        
        scalar_t n_2 = n[2];
        
        scalar_t n_3 = n[3];
        
        
        scalar_t d_0 = d[0];
        scalar_t ad_0 = abs(d_0);
        
        scalar_t d_1 = d[1];
        scalar_t ad_1 = abs(d_1);
        
        scalar_t d_2 = d[2];
        scalar_t ad_2 = abs(d_2);
        
        scalar_t d_3 = d[3];
        scalar_t ad_3 = abs(d_3);
        
        scalar_t P = n_0
        
        + xp1*n_1
        
        + xp2*n_2
        
        + xp3*n_3
                ;

        scalar_t Q = scalar_t(1.0)
                + axp1*ad_0
                + axp2*ad_1
                + axp3*ad_2
                ;

        scalar_t R = n_1
                        + scalar_t(2.0)*n_2*xp1
                + scalar_t(3.0)*n_3*xp2
                ;
        scalar_t S = copysign( scalar_t(1.0), xp1 ) * (ad_0 

                + scalar_t(2.0)*ad_1*axp1
                + scalar_t(3.0)*ad_2*axp2
                 );

        scalar_t mpq2 = -P/(Q*Q);

        scalar_t grad_o = grad_output[index];

        scalar_t d_i_x = (R/Q + S*mpq2); 
        d_x[index] = d_i_x * grad_o;


                scalar_t d_i_d0 = (mpq2*axp1*copysign( scalar_t(1.0), d_0 ));
        d_d0 += d_i_d0 * grad_o;
                scalar_t d_i_d1 = (mpq2*axp2*copysign( scalar_t(1.0), d_1 ));
        d_d1 += d_i_d1 * grad_o;
                scalar_t d_i_d2 = (mpq2*axp3*copysign( scalar_t(1.0), d_2 ));
        d_d2 += d_i_d2 * grad_o;
        

        scalar_t d_i_n0 = scalar_t(1.0)/Q; 
        d_n0 += d_i_n0 * grad_o;

                scalar_t d_i_n1  = xp1/Q;
        d_n1 += d_i_n1 * grad_o;
                scalar_t d_i_n2  = xp2/Q;
        d_n2 += d_i_n2 * grad_o;
                scalar_t d_i_n3  = xp3/Q;
        d_n3 += d_i_n3 * grad_o;
        
    }

    
    atomicAdd(&sdn[0], d_n0);
    
    atomicAdd(&sdn[1], d_n1);
    
    atomicAdd(&sdn[2], d_n2);
    
    atomicAdd(&sdn[3], d_n3);
            
    atomicAdd(&sdd[0], d_d0);
    
    atomicAdd(&sdd[1], d_d1);
    
    atomicAdd(&sdd[2], d_d2);
    

    __syncthreads();

    if( threadIdx.x == 0){
        
        atomicAdd(&d_n[0], sdn[0]);
        
        atomicAdd(&d_n[1], sdn[1]);
        
        atomicAdd(&d_n[2], sdn[2]);
        
        atomicAdd(&d_n[3], sdn[3]);
                        
        atomicAdd(&d_d[0], sdd[0]);
        
        atomicAdd(&d_d[1], sdd[1]);
        
        atomicAdd(&d_d[2], sdd[2]);
        
    }


}

std::vector<torch::Tensor> pau_cuda_backward_3_3(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d){
    const auto x_size = x.numel();
    auto d_x = at::empty_like(x);
    auto d_n = at::zeros_like(n).toType(at::kDouble);
    auto d_d = at::zeros_like(d).toType(at::kDouble);

    int blockSize = THREADS_PER_BLOCK;

    AT_DISPATCH_FLOATING_TYPES(x.type(), "pau_cuda_backward_3_3", ([&] {
    pau_cuda_backward_kernel_3_3<scalar_t>
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

template <typename scalar_t>
__global__ void pau_cuda_forward_kernel_4_4( const scalar_t* __restrict__ x, const scalar_t* __restrict__ n,
    const scalar_t* __restrict__ d, scalar_t* __restrict__ result, size_t x_size) {


    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
        index < x_size;
        index += blockDim.x * gridDim.x){

        scalar_t xp1 = x[index];
        scalar_t axp1 = abs(xp1);

        
        scalar_t xp2 = xp1 * xp1;
        scalar_t axp2 = abs(xp2);
        
        scalar_t xp3 = xp2 * xp1;
        scalar_t axp3 = abs(xp3);
        
        scalar_t xp4 = xp3 * xp1;
        scalar_t axp4 = abs(xp4);
        
        
        scalar_t n_0 = n[0];
        
        scalar_t n_1 = n[1];
        
        scalar_t n_2 = n[2];
        
        scalar_t n_3 = n[3];
        
        scalar_t n_4 = n[4];
        
        
        scalar_t d_0 = d[0];
        scalar_t ad_0 = abs(d_0);
        
        scalar_t d_1 = d[1];
        scalar_t ad_1 = abs(d_1);
        
        scalar_t d_2 = d[2];
        scalar_t ad_2 = abs(d_2);
        
        scalar_t d_3 = d[3];
        scalar_t ad_3 = abs(d_3);
        
        scalar_t d_4 = d[4];
        scalar_t ad_4 = abs(d_4);
        
        scalar_t P = n_0
        
        + xp1*n_1
        
        + xp2*n_2
        
        + xp3*n_3
        
        + xp4*n_4
                ;

        scalar_t Q = scalar_t(1.0)
                + axp1*ad_0
                + axp2*ad_1
                + axp3*ad_2
                + axp4*ad_3
                ;

        result[index] = P/Q;
    }
}

at::Tensor pau_cuda_forward_4_4(torch::Tensor x, torch::Tensor n, torch::Tensor d){
    auto result = at::empty_like(x);
    const auto x_size = x.numel();

    int blockSize = THREADS_PER_BLOCK;
    int numBlocks = (x_size + blockSize - 1) / blockSize;

    AT_DISPATCH_FLOATING_TYPES(x.type(), "pau_cuda_forward_4_4", ([&] {
    pau_cuda_forward_kernel_4_4<scalar_t>
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
__global__ void pau_cuda_backward_kernel_4_4(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ n,
    const scalar_t* __restrict__ d,
    scalar_t* __restrict__ d_x,
    double* __restrict__ d_n,
    double* __restrict__ d_d,
    size_t x_size) {

    __shared__ double sdd[4];
    __shared__ double sdn[4];


    if( threadIdx.x == 0){
        
        sdn[0] = 0;
        
        sdn[1] = 0;
        
        sdn[2] = 0;
        
        sdn[3] = 0;
        
        sdn[4] = 0;
                        
        sdd[0] = 0;
        
        sdd[1] = 0;
        
        sdd[2] = 0;
        
        sdd[3] = 0;
            }

    __syncthreads();
    
    scalar_t d_n0 = 0;
    
    scalar_t d_n1 = 0;
    
    scalar_t d_n2 = 0;
    
    scalar_t d_n3 = 0;
    
    scalar_t d_n4 = 0;
            
    scalar_t d_d0 = 0;
    
    scalar_t d_d1 = 0;
    
    scalar_t d_d2 = 0;
    
    scalar_t d_d3 = 0;
    

    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < x_size;
         index += blockDim.x * gridDim.x)
      {

        scalar_t xp1 = x[index];
        scalar_t axp1 = abs(xp1);

        
        scalar_t xp2 = xp1 * xp1;
        scalar_t axp2 = abs(xp2);
        
        scalar_t xp3 = xp2 * xp1;
        scalar_t axp3 = abs(xp3);
        
        scalar_t xp4 = xp3 * xp1;
        scalar_t axp4 = abs(xp4);
        
        
        scalar_t n_0 = n[0];
        
        scalar_t n_1 = n[1];
        
        scalar_t n_2 = n[2];
        
        scalar_t n_3 = n[3];
        
        scalar_t n_4 = n[4];
        
        
        scalar_t d_0 = d[0];
        scalar_t ad_0 = abs(d_0);
        
        scalar_t d_1 = d[1];
        scalar_t ad_1 = abs(d_1);
        
        scalar_t d_2 = d[2];
        scalar_t ad_2 = abs(d_2);
        
        scalar_t d_3 = d[3];
        scalar_t ad_3 = abs(d_3);
        
        scalar_t d_4 = d[4];
        scalar_t ad_4 = abs(d_4);
        
        scalar_t P = n_0
        
        + xp1*n_1
        
        + xp2*n_2
        
        + xp3*n_3
        
        + xp4*n_4
                ;

        scalar_t Q = scalar_t(1.0)
                + axp1*ad_0
                + axp2*ad_1
                + axp3*ad_2
                + axp4*ad_3
                ;

        scalar_t R = n_1
                        + scalar_t(2.0)*n_2*xp1
                + scalar_t(3.0)*n_3*xp2
                + scalar_t(4.0)*n_4*xp3
                ;
        scalar_t S = copysign( scalar_t(1.0), xp1 ) * (ad_0 

                + scalar_t(2.0)*ad_1*axp1
                + scalar_t(3.0)*ad_2*axp2
                + scalar_t(4.0)*ad_3*axp3
                 );

        scalar_t mpq2 = -P/(Q*Q);

        scalar_t grad_o = grad_output[index];

        scalar_t d_i_x = (R/Q + S*mpq2); 
        d_x[index] = d_i_x * grad_o;


                scalar_t d_i_d0 = (mpq2*axp1*copysign( scalar_t(1.0), d_0 ));
        d_d0 += d_i_d0 * grad_o;
                scalar_t d_i_d1 = (mpq2*axp2*copysign( scalar_t(1.0), d_1 ));
        d_d1 += d_i_d1 * grad_o;
                scalar_t d_i_d2 = (mpq2*axp3*copysign( scalar_t(1.0), d_2 ));
        d_d2 += d_i_d2 * grad_o;
                scalar_t d_i_d3 = (mpq2*axp4*copysign( scalar_t(1.0), d_3 ));
        d_d3 += d_i_d3 * grad_o;
        

        scalar_t d_i_n0 = scalar_t(1.0)/Q; 
        d_n0 += d_i_n0 * grad_o;

                scalar_t d_i_n1  = xp1/Q;
        d_n1 += d_i_n1 * grad_o;
                scalar_t d_i_n2  = xp2/Q;
        d_n2 += d_i_n2 * grad_o;
                scalar_t d_i_n3  = xp3/Q;
        d_n3 += d_i_n3 * grad_o;
                scalar_t d_i_n4  = xp4/Q;
        d_n4 += d_i_n4 * grad_o;
        
    }

    
    atomicAdd(&sdn[0], d_n0);
    
    atomicAdd(&sdn[1], d_n1);
    
    atomicAdd(&sdn[2], d_n2);
    
    atomicAdd(&sdn[3], d_n3);
    
    atomicAdd(&sdn[4], d_n4);
            
    atomicAdd(&sdd[0], d_d0);
    
    atomicAdd(&sdd[1], d_d1);
    
    atomicAdd(&sdd[2], d_d2);
    
    atomicAdd(&sdd[3], d_d3);
    

    __syncthreads();

    if( threadIdx.x == 0){
        
        atomicAdd(&d_n[0], sdn[0]);
        
        atomicAdd(&d_n[1], sdn[1]);
        
        atomicAdd(&d_n[2], sdn[2]);
        
        atomicAdd(&d_n[3], sdn[3]);
        
        atomicAdd(&d_n[4], sdn[4]);
                        
        atomicAdd(&d_d[0], sdd[0]);
        
        atomicAdd(&d_d[1], sdd[1]);
        
        atomicAdd(&d_d[2], sdd[2]);
        
        atomicAdd(&d_d[3], sdd[3]);
        
    }


}

std::vector<torch::Tensor> pau_cuda_backward_4_4(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d){
    const auto x_size = x.numel();
    auto d_x = at::empty_like(x);
    auto d_n = at::zeros_like(n).toType(at::kDouble);
    auto d_d = at::zeros_like(d).toType(at::kDouble);

    int blockSize = THREADS_PER_BLOCK;

    AT_DISPATCH_FLOATING_TYPES(x.type(), "pau_cuda_backward_4_4", ([&] {
    pau_cuda_backward_kernel_4_4<scalar_t>
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

template <typename scalar_t>
__global__ void pau_cuda_forward_kernel_5_5( const scalar_t* __restrict__ x, const scalar_t* __restrict__ n,
    const scalar_t* __restrict__ d, scalar_t* __restrict__ result, size_t x_size) {


    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
        index < x_size;
        index += blockDim.x * gridDim.x){

        scalar_t xp1 = x[index];
        scalar_t axp1 = abs(xp1);

        
        scalar_t xp2 = xp1 * xp1;
        scalar_t axp2 = abs(xp2);
        
        scalar_t xp3 = xp2 * xp1;
        scalar_t axp3 = abs(xp3);
        
        scalar_t xp4 = xp3 * xp1;
        scalar_t axp4 = abs(xp4);
        
        scalar_t xp5 = xp4 * xp1;
        scalar_t axp5 = abs(xp5);
        
        
        scalar_t n_0 = n[0];
        
        scalar_t n_1 = n[1];
        
        scalar_t n_2 = n[2];
        
        scalar_t n_3 = n[3];
        
        scalar_t n_4 = n[4];
        
        scalar_t n_5 = n[5];
        
        
        scalar_t d_0 = d[0];
        scalar_t ad_0 = abs(d_0);
        
        scalar_t d_1 = d[1];
        scalar_t ad_1 = abs(d_1);
        
        scalar_t d_2 = d[2];
        scalar_t ad_2 = abs(d_2);
        
        scalar_t d_3 = d[3];
        scalar_t ad_3 = abs(d_3);
        
        scalar_t d_4 = d[4];
        scalar_t ad_4 = abs(d_4);
        
        scalar_t d_5 = d[5];
        scalar_t ad_5 = abs(d_5);
        
        scalar_t P = n_0
        
        + xp1*n_1
        
        + xp2*n_2
        
        + xp3*n_3
        
        + xp4*n_4
        
        + xp5*n_5
                ;

        scalar_t Q = scalar_t(1.0)
                + axp1*ad_0
                + axp2*ad_1
                + axp3*ad_2
                + axp4*ad_3
                + axp5*ad_4
                ;

        result[index] = P/Q;
    }
}

at::Tensor pau_cuda_forward_5_5(torch::Tensor x, torch::Tensor n, torch::Tensor d){
    auto result = at::empty_like(x);
    const auto x_size = x.numel();

    int blockSize = THREADS_PER_BLOCK;
    int numBlocks = (x_size + blockSize - 1) / blockSize;

    AT_DISPATCH_FLOATING_TYPES(x.type(), "pau_cuda_forward_5_5", ([&] {
    pau_cuda_forward_kernel_5_5<scalar_t>
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
__global__ void pau_cuda_backward_kernel_5_5(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ n,
    const scalar_t* __restrict__ d,
    scalar_t* __restrict__ d_x,
    double* __restrict__ d_n,
    double* __restrict__ d_d,
    size_t x_size) {

    __shared__ double sdd[5];
    __shared__ double sdn[5];


    if( threadIdx.x == 0){
        
        sdn[0] = 0;
        
        sdn[1] = 0;
        
        sdn[2] = 0;
        
        sdn[3] = 0;
        
        sdn[4] = 0;
        
        sdn[5] = 0;
                        
        sdd[0] = 0;
        
        sdd[1] = 0;
        
        sdd[2] = 0;
        
        sdd[3] = 0;
        
        sdd[4] = 0;
            }

    __syncthreads();
    
    scalar_t d_n0 = 0;
    
    scalar_t d_n1 = 0;
    
    scalar_t d_n2 = 0;
    
    scalar_t d_n3 = 0;
    
    scalar_t d_n4 = 0;
    
    scalar_t d_n5 = 0;
            
    scalar_t d_d0 = 0;
    
    scalar_t d_d1 = 0;
    
    scalar_t d_d2 = 0;
    
    scalar_t d_d3 = 0;
    
    scalar_t d_d4 = 0;
    

    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < x_size;
         index += blockDim.x * gridDim.x)
      {

        scalar_t xp1 = x[index];
        scalar_t axp1 = abs(xp1);

        
        scalar_t xp2 = xp1 * xp1;
        scalar_t axp2 = abs(xp2);
        
        scalar_t xp3 = xp2 * xp1;
        scalar_t axp3 = abs(xp3);
        
        scalar_t xp4 = xp3 * xp1;
        scalar_t axp4 = abs(xp4);
        
        scalar_t xp5 = xp4 * xp1;
        scalar_t axp5 = abs(xp5);
        
        
        scalar_t n_0 = n[0];
        
        scalar_t n_1 = n[1];
        
        scalar_t n_2 = n[2];
        
        scalar_t n_3 = n[3];
        
        scalar_t n_4 = n[4];
        
        scalar_t n_5 = n[5];
        
        
        scalar_t d_0 = d[0];
        scalar_t ad_0 = abs(d_0);
        
        scalar_t d_1 = d[1];
        scalar_t ad_1 = abs(d_1);
        
        scalar_t d_2 = d[2];
        scalar_t ad_2 = abs(d_2);
        
        scalar_t d_3 = d[3];
        scalar_t ad_3 = abs(d_3);
        
        scalar_t d_4 = d[4];
        scalar_t ad_4 = abs(d_4);
        
        scalar_t d_5 = d[5];
        scalar_t ad_5 = abs(d_5);
        
        scalar_t P = n_0
        
        + xp1*n_1
        
        + xp2*n_2
        
        + xp3*n_3
        
        + xp4*n_4
        
        + xp5*n_5
                ;

        scalar_t Q = scalar_t(1.0)
                + axp1*ad_0
                + axp2*ad_1
                + axp3*ad_2
                + axp4*ad_3
                + axp5*ad_4
                ;

        scalar_t R = n_1
                        + scalar_t(2.0)*n_2*xp1
                + scalar_t(3.0)*n_3*xp2
                + scalar_t(4.0)*n_4*xp3
                + scalar_t(5.0)*n_5*xp4
                ;
        scalar_t S = copysign( scalar_t(1.0), xp1 ) * (ad_0 

                + scalar_t(2.0)*ad_1*axp1
                + scalar_t(3.0)*ad_2*axp2
                + scalar_t(4.0)*ad_3*axp3
                + scalar_t(5.0)*ad_4*axp4
                 );

        scalar_t mpq2 = -P/(Q*Q);

        scalar_t grad_o = grad_output[index];

        scalar_t d_i_x = (R/Q + S*mpq2); 
        d_x[index] = d_i_x * grad_o;


                scalar_t d_i_d0 = (mpq2*axp1*copysign( scalar_t(1.0), d_0 ));
        d_d0 += d_i_d0 * grad_o;
                scalar_t d_i_d1 = (mpq2*axp2*copysign( scalar_t(1.0), d_1 ));
        d_d1 += d_i_d1 * grad_o;
                scalar_t d_i_d2 = (mpq2*axp3*copysign( scalar_t(1.0), d_2 ));
        d_d2 += d_i_d2 * grad_o;
                scalar_t d_i_d3 = (mpq2*axp4*copysign( scalar_t(1.0), d_3 ));
        d_d3 += d_i_d3 * grad_o;
                scalar_t d_i_d4 = (mpq2*axp5*copysign( scalar_t(1.0), d_4 ));
        d_d4 += d_i_d4 * grad_o;
        

        scalar_t d_i_n0 = scalar_t(1.0)/Q; 
        d_n0 += d_i_n0 * grad_o;

                scalar_t d_i_n1  = xp1/Q;
        d_n1 += d_i_n1 * grad_o;
                scalar_t d_i_n2  = xp2/Q;
        d_n2 += d_i_n2 * grad_o;
                scalar_t d_i_n3  = xp3/Q;
        d_n3 += d_i_n3 * grad_o;
                scalar_t d_i_n4  = xp4/Q;
        d_n4 += d_i_n4 * grad_o;
                scalar_t d_i_n5  = xp5/Q;
        d_n5 += d_i_n5 * grad_o;
        
    }

    
    atomicAdd(&sdn[0], d_n0);
    
    atomicAdd(&sdn[1], d_n1);
    
    atomicAdd(&sdn[2], d_n2);
    
    atomicAdd(&sdn[3], d_n3);
    
    atomicAdd(&sdn[4], d_n4);
    
    atomicAdd(&sdn[5], d_n5);
            
    atomicAdd(&sdd[0], d_d0);
    
    atomicAdd(&sdd[1], d_d1);
    
    atomicAdd(&sdd[2], d_d2);
    
    atomicAdd(&sdd[3], d_d3);
    
    atomicAdd(&sdd[4], d_d4);
    

    __syncthreads();

    if( threadIdx.x == 0){
        
        atomicAdd(&d_n[0], sdn[0]);
        
        atomicAdd(&d_n[1], sdn[1]);
        
        atomicAdd(&d_n[2], sdn[2]);
        
        atomicAdd(&d_n[3], sdn[3]);
        
        atomicAdd(&d_n[4], sdn[4]);
        
        atomicAdd(&d_n[5], sdn[5]);
                        
        atomicAdd(&d_d[0], sdd[0]);
        
        atomicAdd(&d_d[1], sdd[1]);
        
        atomicAdd(&d_d[2], sdd[2]);
        
        atomicAdd(&d_d[3], sdd[3]);
        
        atomicAdd(&d_d[4], sdd[4]);
        
    }


}

std::vector<torch::Tensor> pau_cuda_backward_5_5(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d){
    const auto x_size = x.numel();
    auto d_x = at::empty_like(x);
    auto d_n = at::zeros_like(n).toType(at::kDouble);
    auto d_d = at::zeros_like(d).toType(at::kDouble);

    int blockSize = THREADS_PER_BLOCK;

    AT_DISPATCH_FLOATING_TYPES(x.type(), "pau_cuda_backward_5_5", ([&] {
    pau_cuda_backward_kernel_5_5<scalar_t>
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

template <typename scalar_t>
__global__ void pau_cuda_forward_kernel_6_6( const scalar_t* __restrict__ x, const scalar_t* __restrict__ n,
    const scalar_t* __restrict__ d, scalar_t* __restrict__ result, size_t x_size) {


    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
        index < x_size;
        index += blockDim.x * gridDim.x){

        scalar_t xp1 = x[index];
        scalar_t axp1 = abs(xp1);

        
        scalar_t xp2 = xp1 * xp1;
        scalar_t axp2 = abs(xp2);
        
        scalar_t xp3 = xp2 * xp1;
        scalar_t axp3 = abs(xp3);
        
        scalar_t xp4 = xp3 * xp1;
        scalar_t axp4 = abs(xp4);
        
        scalar_t xp5 = xp4 * xp1;
        scalar_t axp5 = abs(xp5);
        
        scalar_t xp6 = xp5 * xp1;
        scalar_t axp6 = abs(xp6);
        
        
        scalar_t n_0 = n[0];
        
        scalar_t n_1 = n[1];
        
        scalar_t n_2 = n[2];
        
        scalar_t n_3 = n[3];
        
        scalar_t n_4 = n[4];
        
        scalar_t n_5 = n[5];
        
        scalar_t n_6 = n[6];
        
        
        scalar_t d_0 = d[0];
        scalar_t ad_0 = abs(d_0);
        
        scalar_t d_1 = d[1];
        scalar_t ad_1 = abs(d_1);
        
        scalar_t d_2 = d[2];
        scalar_t ad_2 = abs(d_2);
        
        scalar_t d_3 = d[3];
        scalar_t ad_3 = abs(d_3);
        
        scalar_t d_4 = d[4];
        scalar_t ad_4 = abs(d_4);
        
        scalar_t d_5 = d[5];
        scalar_t ad_5 = abs(d_5);
        
        scalar_t d_6 = d[6];
        scalar_t ad_6 = abs(d_6);
        
        scalar_t P = n_0
        
        + xp1*n_1
        
        + xp2*n_2
        
        + xp3*n_3
        
        + xp4*n_4
        
        + xp5*n_5
        
        + xp6*n_6
                ;

        scalar_t Q = scalar_t(1.0)
                + axp1*ad_0
                + axp2*ad_1
                + axp3*ad_2
                + axp4*ad_3
                + axp5*ad_4
                + axp6*ad_5
                ;

        result[index] = P/Q;
    }
}

at::Tensor pau_cuda_forward_6_6(torch::Tensor x, torch::Tensor n, torch::Tensor d){
    auto result = at::empty_like(x);
    const auto x_size = x.numel();

    int blockSize = THREADS_PER_BLOCK;
    int numBlocks = (x_size + blockSize - 1) / blockSize;

    AT_DISPATCH_FLOATING_TYPES(x.type(), "pau_cuda_forward_6_6", ([&] {
    pau_cuda_forward_kernel_6_6<scalar_t>
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
__global__ void pau_cuda_backward_kernel_6_6(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ n,
    const scalar_t* __restrict__ d,
    scalar_t* __restrict__ d_x,
    double* __restrict__ d_n,
    double* __restrict__ d_d,
    size_t x_size) {

    __shared__ double sdd[6];
    __shared__ double sdn[6];


    if( threadIdx.x == 0){
        
        sdn[0] = 0;
        
        sdn[1] = 0;
        
        sdn[2] = 0;
        
        sdn[3] = 0;
        
        sdn[4] = 0;
        
        sdn[5] = 0;
        
        sdn[6] = 0;
                        
        sdd[0] = 0;
        
        sdd[1] = 0;
        
        sdd[2] = 0;
        
        sdd[3] = 0;
        
        sdd[4] = 0;
        
        sdd[5] = 0;
            }

    __syncthreads();
    
    scalar_t d_n0 = 0;
    
    scalar_t d_n1 = 0;
    
    scalar_t d_n2 = 0;
    
    scalar_t d_n3 = 0;
    
    scalar_t d_n4 = 0;
    
    scalar_t d_n5 = 0;
    
    scalar_t d_n6 = 0;
            
    scalar_t d_d0 = 0;
    
    scalar_t d_d1 = 0;
    
    scalar_t d_d2 = 0;
    
    scalar_t d_d3 = 0;
    
    scalar_t d_d4 = 0;
    
    scalar_t d_d5 = 0;
    

    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < x_size;
         index += blockDim.x * gridDim.x)
      {

        scalar_t xp1 = x[index];
        scalar_t axp1 = abs(xp1);

        
        scalar_t xp2 = xp1 * xp1;
        scalar_t axp2 = abs(xp2);
        
        scalar_t xp3 = xp2 * xp1;
        scalar_t axp3 = abs(xp3);
        
        scalar_t xp4 = xp3 * xp1;
        scalar_t axp4 = abs(xp4);
        
        scalar_t xp5 = xp4 * xp1;
        scalar_t axp5 = abs(xp5);
        
        scalar_t xp6 = xp5 * xp1;
        scalar_t axp6 = abs(xp6);
        
        
        scalar_t n_0 = n[0];
        
        scalar_t n_1 = n[1];
        
        scalar_t n_2 = n[2];
        
        scalar_t n_3 = n[3];
        
        scalar_t n_4 = n[4];
        
        scalar_t n_5 = n[5];
        
        scalar_t n_6 = n[6];
        
        
        scalar_t d_0 = d[0];
        scalar_t ad_0 = abs(d_0);
        
        scalar_t d_1 = d[1];
        scalar_t ad_1 = abs(d_1);
        
        scalar_t d_2 = d[2];
        scalar_t ad_2 = abs(d_2);
        
        scalar_t d_3 = d[3];
        scalar_t ad_3 = abs(d_3);
        
        scalar_t d_4 = d[4];
        scalar_t ad_4 = abs(d_4);
        
        scalar_t d_5 = d[5];
        scalar_t ad_5 = abs(d_5);
        
        scalar_t d_6 = d[6];
        scalar_t ad_6 = abs(d_6);
        
        scalar_t P = n_0
        
        + xp1*n_1
        
        + xp2*n_2
        
        + xp3*n_3
        
        + xp4*n_4
        
        + xp5*n_5
        
        + xp6*n_6
                ;

        scalar_t Q = scalar_t(1.0)
                + axp1*ad_0
                + axp2*ad_1
                + axp3*ad_2
                + axp4*ad_3
                + axp5*ad_4
                + axp6*ad_5
                ;

        scalar_t R = n_1
                        + scalar_t(2.0)*n_2*xp1
                + scalar_t(3.0)*n_3*xp2
                + scalar_t(4.0)*n_4*xp3
                + scalar_t(5.0)*n_5*xp4
                + scalar_t(6.0)*n_6*xp5
                ;
        scalar_t S = copysign( scalar_t(1.0), xp1 ) * (ad_0 

                + scalar_t(2.0)*ad_1*axp1
                + scalar_t(3.0)*ad_2*axp2
                + scalar_t(4.0)*ad_3*axp3
                + scalar_t(5.0)*ad_4*axp4
                + scalar_t(6.0)*ad_5*axp5
                 );

        scalar_t mpq2 = -P/(Q*Q);

        scalar_t grad_o = grad_output[index];

        scalar_t d_i_x = (R/Q + S*mpq2); 
        d_x[index] = d_i_x * grad_o;


                scalar_t d_i_d0 = (mpq2*axp1*copysign( scalar_t(1.0), d_0 ));
        d_d0 += d_i_d0 * grad_o;
                scalar_t d_i_d1 = (mpq2*axp2*copysign( scalar_t(1.0), d_1 ));
        d_d1 += d_i_d1 * grad_o;
                scalar_t d_i_d2 = (mpq2*axp3*copysign( scalar_t(1.0), d_2 ));
        d_d2 += d_i_d2 * grad_o;
                scalar_t d_i_d3 = (mpq2*axp4*copysign( scalar_t(1.0), d_3 ));
        d_d3 += d_i_d3 * grad_o;
                scalar_t d_i_d4 = (mpq2*axp5*copysign( scalar_t(1.0), d_4 ));
        d_d4 += d_i_d4 * grad_o;
                scalar_t d_i_d5 = (mpq2*axp6*copysign( scalar_t(1.0), d_5 ));
        d_d5 += d_i_d5 * grad_o;
        

        scalar_t d_i_n0 = scalar_t(1.0)/Q; 
        d_n0 += d_i_n0 * grad_o;

                scalar_t d_i_n1  = xp1/Q;
        d_n1 += d_i_n1 * grad_o;
                scalar_t d_i_n2  = xp2/Q;
        d_n2 += d_i_n2 * grad_o;
                scalar_t d_i_n3  = xp3/Q;
        d_n3 += d_i_n3 * grad_o;
                scalar_t d_i_n4  = xp4/Q;
        d_n4 += d_i_n4 * grad_o;
                scalar_t d_i_n5  = xp5/Q;
        d_n5 += d_i_n5 * grad_o;
                scalar_t d_i_n6  = xp6/Q;
        d_n6 += d_i_n6 * grad_o;
        
    }

    
    atomicAdd(&sdn[0], d_n0);
    
    atomicAdd(&sdn[1], d_n1);
    
    atomicAdd(&sdn[2], d_n2);
    
    atomicAdd(&sdn[3], d_n3);
    
    atomicAdd(&sdn[4], d_n4);
    
    atomicAdd(&sdn[5], d_n5);
    
    atomicAdd(&sdn[6], d_n6);
            
    atomicAdd(&sdd[0], d_d0);
    
    atomicAdd(&sdd[1], d_d1);
    
    atomicAdd(&sdd[2], d_d2);
    
    atomicAdd(&sdd[3], d_d3);
    
    atomicAdd(&sdd[4], d_d4);
    
    atomicAdd(&sdd[5], d_d5);
    

    __syncthreads();

    if( threadIdx.x == 0){
        
        atomicAdd(&d_n[0], sdn[0]);
        
        atomicAdd(&d_n[1], sdn[1]);
        
        atomicAdd(&d_n[2], sdn[2]);
        
        atomicAdd(&d_n[3], sdn[3]);
        
        atomicAdd(&d_n[4], sdn[4]);
        
        atomicAdd(&d_n[5], sdn[5]);
        
        atomicAdd(&d_n[6], sdn[6]);
                        
        atomicAdd(&d_d[0], sdd[0]);
        
        atomicAdd(&d_d[1], sdd[1]);
        
        atomicAdd(&d_d[2], sdd[2]);
        
        atomicAdd(&d_d[3], sdd[3]);
        
        atomicAdd(&d_d[4], sdd[4]);
        
        atomicAdd(&d_d[5], sdd[5]);
        
    }


}

std::vector<torch::Tensor> pau_cuda_backward_6_6(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d){
    const auto x_size = x.numel();
    auto d_x = at::empty_like(x);
    auto d_n = at::zeros_like(n).toType(at::kDouble);
    auto d_d = at::zeros_like(d).toType(at::kDouble);

    int blockSize = THREADS_PER_BLOCK;

    AT_DISPATCH_FLOATING_TYPES(x.type(), "pau_cuda_backward_6_6", ([&] {
    pau_cuda_backward_kernel_6_6<scalar_t>
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

template <typename scalar_t>
__global__ void pau_cuda_forward_kernel_7_7( const scalar_t* __restrict__ x, const scalar_t* __restrict__ n,
    const scalar_t* __restrict__ d, scalar_t* __restrict__ result, size_t x_size) {


    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
        index < x_size;
        index += blockDim.x * gridDim.x){

        scalar_t xp1 = x[index];
        scalar_t axp1 = abs(xp1);

        
        scalar_t xp2 = xp1 * xp1;
        scalar_t axp2 = abs(xp2);
        
        scalar_t xp3 = xp2 * xp1;
        scalar_t axp3 = abs(xp3);
        
        scalar_t xp4 = xp3 * xp1;
        scalar_t axp4 = abs(xp4);
        
        scalar_t xp5 = xp4 * xp1;
        scalar_t axp5 = abs(xp5);
        
        scalar_t xp6 = xp5 * xp1;
        scalar_t axp6 = abs(xp6);
        
        scalar_t xp7 = xp6 * xp1;
        scalar_t axp7 = abs(xp7);
        
        
        scalar_t n_0 = n[0];
        
        scalar_t n_1 = n[1];
        
        scalar_t n_2 = n[2];
        
        scalar_t n_3 = n[3];
        
        scalar_t n_4 = n[4];
        
        scalar_t n_5 = n[5];
        
        scalar_t n_6 = n[6];
        
        scalar_t n_7 = n[7];
        
        
        scalar_t d_0 = d[0];
        scalar_t ad_0 = abs(d_0);
        
        scalar_t d_1 = d[1];
        scalar_t ad_1 = abs(d_1);
        
        scalar_t d_2 = d[2];
        scalar_t ad_2 = abs(d_2);
        
        scalar_t d_3 = d[3];
        scalar_t ad_3 = abs(d_3);
        
        scalar_t d_4 = d[4];
        scalar_t ad_4 = abs(d_4);
        
        scalar_t d_5 = d[5];
        scalar_t ad_5 = abs(d_5);
        
        scalar_t d_6 = d[6];
        scalar_t ad_6 = abs(d_6);
        
        scalar_t d_7 = d[7];
        scalar_t ad_7 = abs(d_7);
        
        scalar_t P = n_0
        
        + xp1*n_1
        
        + xp2*n_2
        
        + xp3*n_3
        
        + xp4*n_4
        
        + xp5*n_5
        
        + xp6*n_6
        
        + xp7*n_7
                ;

        scalar_t Q = scalar_t(1.0)
                + axp1*ad_0
                + axp2*ad_1
                + axp3*ad_2
                + axp4*ad_3
                + axp5*ad_4
                + axp6*ad_5
                + axp7*ad_6
                ;

        result[index] = P/Q;
    }
}

at::Tensor pau_cuda_forward_7_7(torch::Tensor x, torch::Tensor n, torch::Tensor d){
    auto result = at::empty_like(x);
    const auto x_size = x.numel();

    int blockSize = THREADS_PER_BLOCK;
    int numBlocks = (x_size + blockSize - 1) / blockSize;

    AT_DISPATCH_FLOATING_TYPES(x.type(), "pau_cuda_forward_7_7", ([&] {
    pau_cuda_forward_kernel_7_7<scalar_t>
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
__global__ void pau_cuda_backward_kernel_7_7(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ n,
    const scalar_t* __restrict__ d,
    scalar_t* __restrict__ d_x,
    double* __restrict__ d_n,
    double* __restrict__ d_d,
    size_t x_size) {

    __shared__ double sdd[7];
    __shared__ double sdn[7];


    if( threadIdx.x == 0){
        
        sdn[0] = 0;
        
        sdn[1] = 0;
        
        sdn[2] = 0;
        
        sdn[3] = 0;
        
        sdn[4] = 0;
        
        sdn[5] = 0;
        
        sdn[6] = 0;
        
        sdn[7] = 0;
                        
        sdd[0] = 0;
        
        sdd[1] = 0;
        
        sdd[2] = 0;
        
        sdd[3] = 0;
        
        sdd[4] = 0;
        
        sdd[5] = 0;
        
        sdd[6] = 0;
            }

    __syncthreads();
    
    scalar_t d_n0 = 0;
    
    scalar_t d_n1 = 0;
    
    scalar_t d_n2 = 0;
    
    scalar_t d_n3 = 0;
    
    scalar_t d_n4 = 0;
    
    scalar_t d_n5 = 0;
    
    scalar_t d_n6 = 0;
    
    scalar_t d_n7 = 0;
            
    scalar_t d_d0 = 0;
    
    scalar_t d_d1 = 0;
    
    scalar_t d_d2 = 0;
    
    scalar_t d_d3 = 0;
    
    scalar_t d_d4 = 0;
    
    scalar_t d_d5 = 0;
    
    scalar_t d_d6 = 0;
    

    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < x_size;
         index += blockDim.x * gridDim.x)
      {

        scalar_t xp1 = x[index];
        scalar_t axp1 = abs(xp1);

        
        scalar_t xp2 = xp1 * xp1;
        scalar_t axp2 = abs(xp2);
        
        scalar_t xp3 = xp2 * xp1;
        scalar_t axp3 = abs(xp3);
        
        scalar_t xp4 = xp3 * xp1;
        scalar_t axp4 = abs(xp4);
        
        scalar_t xp5 = xp4 * xp1;
        scalar_t axp5 = abs(xp5);
        
        scalar_t xp6 = xp5 * xp1;
        scalar_t axp6 = abs(xp6);
        
        scalar_t xp7 = xp6 * xp1;
        scalar_t axp7 = abs(xp7);
        
        
        scalar_t n_0 = n[0];
        
        scalar_t n_1 = n[1];
        
        scalar_t n_2 = n[2];
        
        scalar_t n_3 = n[3];
        
        scalar_t n_4 = n[4];
        
        scalar_t n_5 = n[5];
        
        scalar_t n_6 = n[6];
        
        scalar_t n_7 = n[7];
        
        
        scalar_t d_0 = d[0];
        scalar_t ad_0 = abs(d_0);
        
        scalar_t d_1 = d[1];
        scalar_t ad_1 = abs(d_1);
        
        scalar_t d_2 = d[2];
        scalar_t ad_2 = abs(d_2);
        
        scalar_t d_3 = d[3];
        scalar_t ad_3 = abs(d_3);
        
        scalar_t d_4 = d[4];
        scalar_t ad_4 = abs(d_4);
        
        scalar_t d_5 = d[5];
        scalar_t ad_5 = abs(d_5);
        
        scalar_t d_6 = d[6];
        scalar_t ad_6 = abs(d_6);
        
        scalar_t d_7 = d[7];
        scalar_t ad_7 = abs(d_7);
        
        scalar_t P = n_0
        
        + xp1*n_1
        
        + xp2*n_2
        
        + xp3*n_3
        
        + xp4*n_4
        
        + xp5*n_5
        
        + xp6*n_6
        
        + xp7*n_7
                ;

        scalar_t Q = scalar_t(1.0)
                + axp1*ad_0
                + axp2*ad_1
                + axp3*ad_2
                + axp4*ad_3
                + axp5*ad_4
                + axp6*ad_5
                + axp7*ad_6
                ;

        scalar_t R = n_1
                        + scalar_t(2.0)*n_2*xp1
                + scalar_t(3.0)*n_3*xp2
                + scalar_t(4.0)*n_4*xp3
                + scalar_t(5.0)*n_5*xp4
                + scalar_t(6.0)*n_6*xp5
                + scalar_t(7.0)*n_7*xp6
                ;
        scalar_t S = copysign( scalar_t(1.0), xp1 ) * (ad_0 

                + scalar_t(2.0)*ad_1*axp1
                + scalar_t(3.0)*ad_2*axp2
                + scalar_t(4.0)*ad_3*axp3
                + scalar_t(5.0)*ad_4*axp4
                + scalar_t(6.0)*ad_5*axp5
                + scalar_t(7.0)*ad_6*axp6
                 );

        scalar_t mpq2 = -P/(Q*Q);

        scalar_t grad_o = grad_output[index];

        scalar_t d_i_x = (R/Q + S*mpq2); 
        d_x[index] = d_i_x * grad_o;


                scalar_t d_i_d0 = (mpq2*axp1*copysign( scalar_t(1.0), d_0 ));
        d_d0 += d_i_d0 * grad_o;
                scalar_t d_i_d1 = (mpq2*axp2*copysign( scalar_t(1.0), d_1 ));
        d_d1 += d_i_d1 * grad_o;
                scalar_t d_i_d2 = (mpq2*axp3*copysign( scalar_t(1.0), d_2 ));
        d_d2 += d_i_d2 * grad_o;
                scalar_t d_i_d3 = (mpq2*axp4*copysign( scalar_t(1.0), d_3 ));
        d_d3 += d_i_d3 * grad_o;
                scalar_t d_i_d4 = (mpq2*axp5*copysign( scalar_t(1.0), d_4 ));
        d_d4 += d_i_d4 * grad_o;
                scalar_t d_i_d5 = (mpq2*axp6*copysign( scalar_t(1.0), d_5 ));
        d_d5 += d_i_d5 * grad_o;
                scalar_t d_i_d6 = (mpq2*axp7*copysign( scalar_t(1.0), d_6 ));
        d_d6 += d_i_d6 * grad_o;
        

        scalar_t d_i_n0 = scalar_t(1.0)/Q; 
        d_n0 += d_i_n0 * grad_o;

                scalar_t d_i_n1  = xp1/Q;
        d_n1 += d_i_n1 * grad_o;
                scalar_t d_i_n2  = xp2/Q;
        d_n2 += d_i_n2 * grad_o;
                scalar_t d_i_n3  = xp3/Q;
        d_n3 += d_i_n3 * grad_o;
                scalar_t d_i_n4  = xp4/Q;
        d_n4 += d_i_n4 * grad_o;
                scalar_t d_i_n5  = xp5/Q;
        d_n5 += d_i_n5 * grad_o;
                scalar_t d_i_n6  = xp6/Q;
        d_n6 += d_i_n6 * grad_o;
                scalar_t d_i_n7  = xp7/Q;
        d_n7 += d_i_n7 * grad_o;
        
    }

    
    atomicAdd(&sdn[0], d_n0);
    
    atomicAdd(&sdn[1], d_n1);
    
    atomicAdd(&sdn[2], d_n2);
    
    atomicAdd(&sdn[3], d_n3);
    
    atomicAdd(&sdn[4], d_n4);
    
    atomicAdd(&sdn[5], d_n5);
    
    atomicAdd(&sdn[6], d_n6);
    
    atomicAdd(&sdn[7], d_n7);
            
    atomicAdd(&sdd[0], d_d0);
    
    atomicAdd(&sdd[1], d_d1);
    
    atomicAdd(&sdd[2], d_d2);
    
    atomicAdd(&sdd[3], d_d3);
    
    atomicAdd(&sdd[4], d_d4);
    
    atomicAdd(&sdd[5], d_d5);
    
    atomicAdd(&sdd[6], d_d6);
    

    __syncthreads();

    if( threadIdx.x == 0){
        
        atomicAdd(&d_n[0], sdn[0]);
        
        atomicAdd(&d_n[1], sdn[1]);
        
        atomicAdd(&d_n[2], sdn[2]);
        
        atomicAdd(&d_n[3], sdn[3]);
        
        atomicAdd(&d_n[4], sdn[4]);
        
        atomicAdd(&d_n[5], sdn[5]);
        
        atomicAdd(&d_n[6], sdn[6]);
        
        atomicAdd(&d_n[7], sdn[7]);
                        
        atomicAdd(&d_d[0], sdd[0]);
        
        atomicAdd(&d_d[1], sdd[1]);
        
        atomicAdd(&d_d[2], sdd[2]);
        
        atomicAdd(&d_d[3], sdd[3]);
        
        atomicAdd(&d_d[4], sdd[4]);
        
        atomicAdd(&d_d[5], sdd[5]);
        
        atomicAdd(&d_d[6], sdd[6]);
        
    }


}

std::vector<torch::Tensor> pau_cuda_backward_7_7(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d){
    const auto x_size = x.numel();
    auto d_x = at::empty_like(x);
    auto d_n = at::zeros_like(n).toType(at::kDouble);
    auto d_d = at::zeros_like(d).toType(at::kDouble);

    int blockSize = THREADS_PER_BLOCK;

    AT_DISPATCH_FLOATING_TYPES(x.type(), "pau_cuda_backward_7_7", ([&] {
    pau_cuda_backward_kernel_7_7<scalar_t>
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

template <typename scalar_t>
__global__ void pau_cuda_forward_kernel_8_8( const scalar_t* __restrict__ x, const scalar_t* __restrict__ n,
    const scalar_t* __restrict__ d, scalar_t* __restrict__ result, size_t x_size) {


    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
        index < x_size;
        index += blockDim.x * gridDim.x){

        scalar_t xp1 = x[index];
        scalar_t axp1 = abs(xp1);

        
        scalar_t xp2 = xp1 * xp1;
        scalar_t axp2 = abs(xp2);
        
        scalar_t xp3 = xp2 * xp1;
        scalar_t axp3 = abs(xp3);
        
        scalar_t xp4 = xp3 * xp1;
        scalar_t axp4 = abs(xp4);
        
        scalar_t xp5 = xp4 * xp1;
        scalar_t axp5 = abs(xp5);
        
        scalar_t xp6 = xp5 * xp1;
        scalar_t axp6 = abs(xp6);
        
        scalar_t xp7 = xp6 * xp1;
        scalar_t axp7 = abs(xp7);
        
        scalar_t xp8 = xp7 * xp1;
        scalar_t axp8 = abs(xp8);
        
        
        scalar_t n_0 = n[0];
        
        scalar_t n_1 = n[1];
        
        scalar_t n_2 = n[2];
        
        scalar_t n_3 = n[3];
        
        scalar_t n_4 = n[4];
        
        scalar_t n_5 = n[5];
        
        scalar_t n_6 = n[6];
        
        scalar_t n_7 = n[7];
        
        scalar_t n_8 = n[8];
        
        
        scalar_t d_0 = d[0];
        scalar_t ad_0 = abs(d_0);
        
        scalar_t d_1 = d[1];
        scalar_t ad_1 = abs(d_1);
        
        scalar_t d_2 = d[2];
        scalar_t ad_2 = abs(d_2);
        
        scalar_t d_3 = d[3];
        scalar_t ad_3 = abs(d_3);
        
        scalar_t d_4 = d[4];
        scalar_t ad_4 = abs(d_4);
        
        scalar_t d_5 = d[5];
        scalar_t ad_5 = abs(d_5);
        
        scalar_t d_6 = d[6];
        scalar_t ad_6 = abs(d_6);
        
        scalar_t d_7 = d[7];
        scalar_t ad_7 = abs(d_7);
        
        scalar_t d_8 = d[8];
        scalar_t ad_8 = abs(d_8);
        
        scalar_t P = n_0
        
        + xp1*n_1
        
        + xp2*n_2
        
        + xp3*n_3
        
        + xp4*n_4
        
        + xp5*n_5
        
        + xp6*n_6
        
        + xp7*n_7
        
        + xp8*n_8
                ;

        scalar_t Q = scalar_t(1.0)
                + axp1*ad_0
                + axp2*ad_1
                + axp3*ad_2
                + axp4*ad_3
                + axp5*ad_4
                + axp6*ad_5
                + axp7*ad_6
                + axp8*ad_7
                ;

        result[index] = P/Q;
    }
}

at::Tensor pau_cuda_forward_8_8(torch::Tensor x, torch::Tensor n, torch::Tensor d){
    auto result = at::empty_like(x);
    const auto x_size = x.numel();

    int blockSize = THREADS_PER_BLOCK;
    int numBlocks = (x_size + blockSize - 1) / blockSize;

    AT_DISPATCH_FLOATING_TYPES(x.type(), "pau_cuda_forward_8_8", ([&] {
    pau_cuda_forward_kernel_8_8<scalar_t>
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
__global__ void pau_cuda_backward_kernel_8_8(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ n,
    const scalar_t* __restrict__ d,
    scalar_t* __restrict__ d_x,
    double* __restrict__ d_n,
    double* __restrict__ d_d,
    size_t x_size) {

    __shared__ double sdd[8];
    __shared__ double sdn[8];


    if( threadIdx.x == 0){
        
        sdn[0] = 0;
        
        sdn[1] = 0;
        
        sdn[2] = 0;
        
        sdn[3] = 0;
        
        sdn[4] = 0;
        
        sdn[5] = 0;
        
        sdn[6] = 0;
        
        sdn[7] = 0;
        
        sdn[8] = 0;
                        
        sdd[0] = 0;
        
        sdd[1] = 0;
        
        sdd[2] = 0;
        
        sdd[3] = 0;
        
        sdd[4] = 0;
        
        sdd[5] = 0;
        
        sdd[6] = 0;
        
        sdd[7] = 0;
            }

    __syncthreads();
    
    scalar_t d_n0 = 0;
    
    scalar_t d_n1 = 0;
    
    scalar_t d_n2 = 0;
    
    scalar_t d_n3 = 0;
    
    scalar_t d_n4 = 0;
    
    scalar_t d_n5 = 0;
    
    scalar_t d_n6 = 0;
    
    scalar_t d_n7 = 0;
    
    scalar_t d_n8 = 0;
            
    scalar_t d_d0 = 0;
    
    scalar_t d_d1 = 0;
    
    scalar_t d_d2 = 0;
    
    scalar_t d_d3 = 0;
    
    scalar_t d_d4 = 0;
    
    scalar_t d_d5 = 0;
    
    scalar_t d_d6 = 0;
    
    scalar_t d_d7 = 0;
    

    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < x_size;
         index += blockDim.x * gridDim.x)
      {

        scalar_t xp1 = x[index];
        scalar_t axp1 = abs(xp1);

        
        scalar_t xp2 = xp1 * xp1;
        scalar_t axp2 = abs(xp2);
        
        scalar_t xp3 = xp2 * xp1;
        scalar_t axp3 = abs(xp3);
        
        scalar_t xp4 = xp3 * xp1;
        scalar_t axp4 = abs(xp4);
        
        scalar_t xp5 = xp4 * xp1;
        scalar_t axp5 = abs(xp5);
        
        scalar_t xp6 = xp5 * xp1;
        scalar_t axp6 = abs(xp6);
        
        scalar_t xp7 = xp6 * xp1;
        scalar_t axp7 = abs(xp7);
        
        scalar_t xp8 = xp7 * xp1;
        scalar_t axp8 = abs(xp8);
        
        
        scalar_t n_0 = n[0];
        
        scalar_t n_1 = n[1];
        
        scalar_t n_2 = n[2];
        
        scalar_t n_3 = n[3];
        
        scalar_t n_4 = n[4];
        
        scalar_t n_5 = n[5];
        
        scalar_t n_6 = n[6];
        
        scalar_t n_7 = n[7];
        
        scalar_t n_8 = n[8];
        
        
        scalar_t d_0 = d[0];
        scalar_t ad_0 = abs(d_0);
        
        scalar_t d_1 = d[1];
        scalar_t ad_1 = abs(d_1);
        
        scalar_t d_2 = d[2];
        scalar_t ad_2 = abs(d_2);
        
        scalar_t d_3 = d[3];
        scalar_t ad_3 = abs(d_3);
        
        scalar_t d_4 = d[4];
        scalar_t ad_4 = abs(d_4);
        
        scalar_t d_5 = d[5];
        scalar_t ad_5 = abs(d_5);
        
        scalar_t d_6 = d[6];
        scalar_t ad_6 = abs(d_6);
        
        scalar_t d_7 = d[7];
        scalar_t ad_7 = abs(d_7);
        
        scalar_t d_8 = d[8];
        scalar_t ad_8 = abs(d_8);
        
        scalar_t P = n_0
        
        + xp1*n_1
        
        + xp2*n_2
        
        + xp3*n_3
        
        + xp4*n_4
        
        + xp5*n_5
        
        + xp6*n_6
        
        + xp7*n_7
        
        + xp8*n_8
                ;

        scalar_t Q = scalar_t(1.0)
                + axp1*ad_0
                + axp2*ad_1
                + axp3*ad_2
                + axp4*ad_3
                + axp5*ad_4
                + axp6*ad_5
                + axp7*ad_6
                + axp8*ad_7
                ;

        scalar_t R = n_1
                        + scalar_t(2.0)*n_2*xp1
                + scalar_t(3.0)*n_3*xp2
                + scalar_t(4.0)*n_4*xp3
                + scalar_t(5.0)*n_5*xp4
                + scalar_t(6.0)*n_6*xp5
                + scalar_t(7.0)*n_7*xp6
                + scalar_t(8.0)*n_8*xp7
                ;
        scalar_t S = copysign( scalar_t(1.0), xp1 ) * (ad_0 

                + scalar_t(2.0)*ad_1*axp1
                + scalar_t(3.0)*ad_2*axp2
                + scalar_t(4.0)*ad_3*axp3
                + scalar_t(5.0)*ad_4*axp4
                + scalar_t(6.0)*ad_5*axp5
                + scalar_t(7.0)*ad_6*axp6
                + scalar_t(8.0)*ad_7*axp7
                 );

        scalar_t mpq2 = -P/(Q*Q);

        scalar_t grad_o = grad_output[index];

        scalar_t d_i_x = (R/Q + S*mpq2); 
        d_x[index] = d_i_x * grad_o;


                scalar_t d_i_d0 = (mpq2*axp1*copysign( scalar_t(1.0), d_0 ));
        d_d0 += d_i_d0 * grad_o;
                scalar_t d_i_d1 = (mpq2*axp2*copysign( scalar_t(1.0), d_1 ));
        d_d1 += d_i_d1 * grad_o;
                scalar_t d_i_d2 = (mpq2*axp3*copysign( scalar_t(1.0), d_2 ));
        d_d2 += d_i_d2 * grad_o;
                scalar_t d_i_d3 = (mpq2*axp4*copysign( scalar_t(1.0), d_3 ));
        d_d3 += d_i_d3 * grad_o;
                scalar_t d_i_d4 = (mpq2*axp5*copysign( scalar_t(1.0), d_4 ));
        d_d4 += d_i_d4 * grad_o;
                scalar_t d_i_d5 = (mpq2*axp6*copysign( scalar_t(1.0), d_5 ));
        d_d5 += d_i_d5 * grad_o;
                scalar_t d_i_d6 = (mpq2*axp7*copysign( scalar_t(1.0), d_6 ));
        d_d6 += d_i_d6 * grad_o;
                scalar_t d_i_d7 = (mpq2*axp8*copysign( scalar_t(1.0), d_7 ));
        d_d7 += d_i_d7 * grad_o;
        

        scalar_t d_i_n0 = scalar_t(1.0)/Q; 
        d_n0 += d_i_n0 * grad_o;

                scalar_t d_i_n1  = xp1/Q;
        d_n1 += d_i_n1 * grad_o;
                scalar_t d_i_n2  = xp2/Q;
        d_n2 += d_i_n2 * grad_o;
                scalar_t d_i_n3  = xp3/Q;
        d_n3 += d_i_n3 * grad_o;
                scalar_t d_i_n4  = xp4/Q;
        d_n4 += d_i_n4 * grad_o;
                scalar_t d_i_n5  = xp5/Q;
        d_n5 += d_i_n5 * grad_o;
                scalar_t d_i_n6  = xp6/Q;
        d_n6 += d_i_n6 * grad_o;
                scalar_t d_i_n7  = xp7/Q;
        d_n7 += d_i_n7 * grad_o;
                scalar_t d_i_n8  = xp8/Q;
        d_n8 += d_i_n8 * grad_o;
        
    }

    
    atomicAdd(&sdn[0], d_n0);
    
    atomicAdd(&sdn[1], d_n1);
    
    atomicAdd(&sdn[2], d_n2);
    
    atomicAdd(&sdn[3], d_n3);
    
    atomicAdd(&sdn[4], d_n4);
    
    atomicAdd(&sdn[5], d_n5);
    
    atomicAdd(&sdn[6], d_n6);
    
    atomicAdd(&sdn[7], d_n7);
    
    atomicAdd(&sdn[8], d_n8);
            
    atomicAdd(&sdd[0], d_d0);
    
    atomicAdd(&sdd[1], d_d1);
    
    atomicAdd(&sdd[2], d_d2);
    
    atomicAdd(&sdd[3], d_d3);
    
    atomicAdd(&sdd[4], d_d4);
    
    atomicAdd(&sdd[5], d_d5);
    
    atomicAdd(&sdd[6], d_d6);
    
    atomicAdd(&sdd[7], d_d7);
    

    __syncthreads();

    if( threadIdx.x == 0){
        
        atomicAdd(&d_n[0], sdn[0]);
        
        atomicAdd(&d_n[1], sdn[1]);
        
        atomicAdd(&d_n[2], sdn[2]);
        
        atomicAdd(&d_n[3], sdn[3]);
        
        atomicAdd(&d_n[4], sdn[4]);
        
        atomicAdd(&d_n[5], sdn[5]);
        
        atomicAdd(&d_n[6], sdn[6]);
        
        atomicAdd(&d_n[7], sdn[7]);
        
        atomicAdd(&d_n[8], sdn[8]);
                        
        atomicAdd(&d_d[0], sdd[0]);
        
        atomicAdd(&d_d[1], sdd[1]);
        
        atomicAdd(&d_d[2], sdd[2]);
        
        atomicAdd(&d_d[3], sdd[3]);
        
        atomicAdd(&d_d[4], sdd[4]);
        
        atomicAdd(&d_d[5], sdd[5]);
        
        atomicAdd(&d_d[6], sdd[6]);
        
        atomicAdd(&d_d[7], sdd[7]);
        
    }


}

std::vector<torch::Tensor> pau_cuda_backward_8_8(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d){
    const auto x_size = x.numel();
    auto d_x = at::empty_like(x);
    auto d_n = at::zeros_like(n).toType(at::kDouble);
    auto d_d = at::zeros_like(d).toType(at::kDouble);

    int blockSize = THREADS_PER_BLOCK;

    AT_DISPATCH_FLOATING_TYPES(x.type(), "pau_cuda_backward_8_8", ([&] {
    pau_cuda_backward_kernel_8_8<scalar_t>
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

template <typename scalar_t>
__global__ void pau_cuda_forward_kernel_5_4( const scalar_t* __restrict__ x, const scalar_t* __restrict__ n,
    const scalar_t* __restrict__ d, scalar_t* __restrict__ result, size_t x_size) {


    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
        index < x_size;
        index += blockDim.x * gridDim.x){

        scalar_t xp1 = x[index];
        scalar_t axp1 = abs(xp1);

        
        scalar_t xp2 = xp1 * xp1;
        scalar_t axp2 = abs(xp2);
        
        scalar_t xp3 = xp2 * xp1;
        scalar_t axp3 = abs(xp3);
        
        scalar_t xp4 = xp3 * xp1;
        scalar_t axp4 = abs(xp4);
        
        scalar_t xp5 = xp4 * xp1;
        scalar_t axp5 = abs(xp5);
        
        
        scalar_t n_0 = n[0];
        
        scalar_t n_1 = n[1];
        
        scalar_t n_2 = n[2];
        
        scalar_t n_3 = n[3];
        
        scalar_t n_4 = n[4];
        
        scalar_t n_5 = n[5];
        
        
        scalar_t d_0 = d[0];
        scalar_t ad_0 = abs(d_0);
        
        scalar_t d_1 = d[1];
        scalar_t ad_1 = abs(d_1);
        
        scalar_t d_2 = d[2];
        scalar_t ad_2 = abs(d_2);
        
        scalar_t d_3 = d[3];
        scalar_t ad_3 = abs(d_3);
        
        scalar_t d_4 = d[4];
        scalar_t ad_4 = abs(d_4);
        
        scalar_t P = n_0
        
        + xp1*n_1
        
        + xp2*n_2
        
        + xp3*n_3
        
        + xp4*n_4
        
        + xp5*n_5
                ;

        scalar_t Q = scalar_t(1.0)
                + axp1*ad_0
                + axp2*ad_1
                + axp3*ad_2
                + axp4*ad_3
                ;

        result[index] = P/Q;
    }
}

at::Tensor pau_cuda_forward_5_4(torch::Tensor x, torch::Tensor n, torch::Tensor d){
    auto result = at::empty_like(x);
    const auto x_size = x.numel();

    int blockSize = THREADS_PER_BLOCK;
    int numBlocks = (x_size + blockSize - 1) / blockSize;

    AT_DISPATCH_FLOATING_TYPES(x.type(), "pau_cuda_forward_5_4", ([&] {
    pau_cuda_forward_kernel_5_4<scalar_t>
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
__global__ void pau_cuda_backward_kernel_5_4(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ n,
    const scalar_t* __restrict__ d,
    scalar_t* __restrict__ d_x,
    double* __restrict__ d_n,
    double* __restrict__ d_d,
    size_t x_size) {

    __shared__ double sdd[5];
    __shared__ double sdn[4];


    if( threadIdx.x == 0){
        
        sdn[0] = 0;
        
        sdn[1] = 0;
        
        sdn[2] = 0;
        
        sdn[3] = 0;
        
        sdn[4] = 0;
        
        sdn[5] = 0;
                        
        sdd[0] = 0;
        
        sdd[1] = 0;
        
        sdd[2] = 0;
        
        sdd[3] = 0;
            }

    __syncthreads();
    
    scalar_t d_n0 = 0;
    
    scalar_t d_n1 = 0;
    
    scalar_t d_n2 = 0;
    
    scalar_t d_n3 = 0;
    
    scalar_t d_n4 = 0;
    
    scalar_t d_n5 = 0;
            
    scalar_t d_d0 = 0;
    
    scalar_t d_d1 = 0;
    
    scalar_t d_d2 = 0;
    
    scalar_t d_d3 = 0;
    

    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < x_size;
         index += blockDim.x * gridDim.x)
      {

        scalar_t xp1 = x[index];
        scalar_t axp1 = abs(xp1);

        
        scalar_t xp2 = xp1 * xp1;
        scalar_t axp2 = abs(xp2);
        
        scalar_t xp3 = xp2 * xp1;
        scalar_t axp3 = abs(xp3);
        
        scalar_t xp4 = xp3 * xp1;
        scalar_t axp4 = abs(xp4);
        
        scalar_t xp5 = xp4 * xp1;
        scalar_t axp5 = abs(xp5);
        
        
        scalar_t n_0 = n[0];
        
        scalar_t n_1 = n[1];
        
        scalar_t n_2 = n[2];
        
        scalar_t n_3 = n[3];
        
        scalar_t n_4 = n[4];
        
        scalar_t n_5 = n[5];
        
        
        scalar_t d_0 = d[0];
        scalar_t ad_0 = abs(d_0);
        
        scalar_t d_1 = d[1];
        scalar_t ad_1 = abs(d_1);
        
        scalar_t d_2 = d[2];
        scalar_t ad_2 = abs(d_2);
        
        scalar_t d_3 = d[3];
        scalar_t ad_3 = abs(d_3);
        
        scalar_t d_4 = d[4];
        scalar_t ad_4 = abs(d_4);
        
        scalar_t P = n_0
        
        + xp1*n_1
        
        + xp2*n_2
        
        + xp3*n_3
        
        + xp4*n_4
        
        + xp5*n_5
                ;

        scalar_t Q = scalar_t(1.0)
                + axp1*ad_0
                + axp2*ad_1
                + axp3*ad_2
                + axp4*ad_3
                ;

        scalar_t R = n_1
                        + scalar_t(2.0)*n_2*xp1
                + scalar_t(3.0)*n_3*xp2
                + scalar_t(4.0)*n_4*xp3
                + scalar_t(5.0)*n_5*xp4
                ;
        scalar_t S = copysign( scalar_t(1.0), xp1 ) * (ad_0 

                + scalar_t(2.0)*ad_1*axp1
                + scalar_t(3.0)*ad_2*axp2
                + scalar_t(4.0)*ad_3*axp3
                 );

        scalar_t mpq2 = -P/(Q*Q);

        scalar_t grad_o = grad_output[index];

        scalar_t d_i_x = (R/Q + S*mpq2); 
        d_x[index] = d_i_x * grad_o;


                scalar_t d_i_d0 = (mpq2*axp1*copysign( scalar_t(1.0), d_0 ));
        d_d0 += d_i_d0 * grad_o;
                scalar_t d_i_d1 = (mpq2*axp2*copysign( scalar_t(1.0), d_1 ));
        d_d1 += d_i_d1 * grad_o;
                scalar_t d_i_d2 = (mpq2*axp3*copysign( scalar_t(1.0), d_2 ));
        d_d2 += d_i_d2 * grad_o;
                scalar_t d_i_d3 = (mpq2*axp4*copysign( scalar_t(1.0), d_3 ));
        d_d3 += d_i_d3 * grad_o;
        

        scalar_t d_i_n0 = scalar_t(1.0)/Q; 
        d_n0 += d_i_n0 * grad_o;

                scalar_t d_i_n1  = xp1/Q;
        d_n1 += d_i_n1 * grad_o;
                scalar_t d_i_n2  = xp2/Q;
        d_n2 += d_i_n2 * grad_o;
                scalar_t d_i_n3  = xp3/Q;
        d_n3 += d_i_n3 * grad_o;
                scalar_t d_i_n4  = xp4/Q;
        d_n4 += d_i_n4 * grad_o;
                scalar_t d_i_n5  = xp5/Q;
        d_n5 += d_i_n5 * grad_o;
        
    }

    
    atomicAdd(&sdn[0], d_n0);
    
    atomicAdd(&sdn[1], d_n1);
    
    atomicAdd(&sdn[2], d_n2);
    
    atomicAdd(&sdn[3], d_n3);
    
    atomicAdd(&sdn[4], d_n4);
    
    atomicAdd(&sdn[5], d_n5);
            
    atomicAdd(&sdd[0], d_d0);
    
    atomicAdd(&sdd[1], d_d1);
    
    atomicAdd(&sdd[2], d_d2);
    
    atomicAdd(&sdd[3], d_d3);
    

    __syncthreads();

    if( threadIdx.x == 0){
        
        atomicAdd(&d_n[0], sdn[0]);
        
        atomicAdd(&d_n[1], sdn[1]);
        
        atomicAdd(&d_n[2], sdn[2]);
        
        atomicAdd(&d_n[3], sdn[3]);
        
        atomicAdd(&d_n[4], sdn[4]);
        
        atomicAdd(&d_n[5], sdn[5]);
                        
        atomicAdd(&d_d[0], sdd[0]);
        
        atomicAdd(&d_d[1], sdd[1]);
        
        atomicAdd(&d_d[2], sdd[2]);
        
        atomicAdd(&d_d[3], sdd[3]);
        
    }


}

std::vector<torch::Tensor> pau_cuda_backward_5_4(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d){
    const auto x_size = x.numel();
    auto d_x = at::empty_like(x);
    auto d_n = at::zeros_like(n).toType(at::kDouble);
    auto d_d = at::zeros_like(d).toType(at::kDouble);

    int blockSize = THREADS_PER_BLOCK;

    AT_DISPATCH_FLOATING_TYPES(x.type(), "pau_cuda_backward_5_4", ([&] {
    pau_cuda_backward_kernel_5_4<scalar_t>
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
        