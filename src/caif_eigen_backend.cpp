// Copyright 2026 Eric Malloy
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "caif_eigen_backend.h"
#include "caif_constants.h"
#include "caif_exception.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <limits>
#include <cmath>
#include <cstring>

namespace instance
{

CAIF_EigenBackend::CAIF_EigenBackend()
{
}

CAIF_EigenBackend::~CAIF_EigenBackend()
{
}

std::unique_ptr<CAIF_TensorData> CAIF_EigenBackend::CreateTensor(
                                                                const std::vector<uint32_t> &shape,
                                                                const CAIF_DataType &dtype
                                                               )
{
  return std::make_unique<CAIF_CPUTensorData>(shape, dtype);
}

void CAIF_EigenBackend::MatrixMultiply(
                                      const CAIF_TensorData &a,
                                      const CAIF_TensorData &b,
                                      CAIF_TensorData &result
                                     )
{
  // Validate dimensions first
  ValidateMatrixDimensions(a, b, result);
  
  // Cast to Eigen tensor data for shape access
  const CAIF_CPUTensorData &eigen_a=static_cast<const CAIF_CPUTensorData &>(a);
  const CAIF_CPUTensorData &eigen_b=static_cast<const CAIF_CPUTensorData &>(b);
  CAIF_CPUTensorData &eigen_result=static_cast<CAIF_CPUTensorData &>(result);
  (void)eigen_result;
  
  const auto &shape_a=eigen_a.Shape();
  const auto &shape_b=eigen_b.Shape();
  
  // Ensure we have 2D matrices
  if(shape_a.size()!=g_caif_2d_matrix_dimensions||shape_b.size()!=g_caif_2d_matrix_dimensions)
  {
    THROW_CAIFE("Matrix multiplication requires 2D tensors");
  }
  
  const size_t rows_a=shape_a[0];
  const size_t cols_a=shape_a[1];
  const size_t rows_b=shape_b[0];
  const size_t cols_b=shape_b[1];
  
  if(cols_a!=rows_b)
  {
    THROW_CAIFE("Incompatible matrix dimensions for multiplication");
  }
  
  // Get data pointers
  const float *a_data=static_cast<const float *>(a.RawData());
  const float *b_data=static_cast<const float *>(b.RawData());
  float *result_data=static_cast<float *>(result.MutableRawData());
  
  // Use Eigen for matrix multiplication with row-major layout to match tensor storage
  using RowMajorMatrix=Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  Eigen::Map<const RowMajorMatrix> eigen_a_map(a_data, rows_a, cols_a);
  Eigen::Map<const RowMajorMatrix> eigen_b_map(b_data, rows_b, cols_b);
  Eigen::Map<RowMajorMatrix> eigen_result_map(result_data, rows_a, cols_b);
  
  eigen_result_map=eigen_a_map*eigen_b_map;
  
  return;
}

void CAIF_EigenBackend::MatrixMultiplyEx(
                                        const CAIF_TensorData &a,
                                        const CAIF_TensorData &b,
                                        CAIF_TensorData &result,
                                        const Transpose_e trans_a,
                                        const Transpose_e trans_b
                                       )
{
  const auto &shape_a=a.Shape();
  const auto &shape_b=b.Shape();

  if(shape_a.size()!=g_caif_2d_matrix_dimensions||shape_b.size()!=g_caif_2d_matrix_dimensions)
  {
    THROW_CAIFE("Matrix multiplication requires 2D tensors");
  }

  const int a_rows=static_cast<int>(shape_a[0]);
  const int a_cols=static_cast<int>(shape_a[1]);
  const int b_rows=static_cast<int>(shape_b[0]);
  const int b_cols=static_cast<int>(shape_b[1]);

  const bool ta=(trans_a==Transpose_e::Trans);
  const bool tb=(trans_b==Transpose_e::Trans);

  int m;
  if(ta==true)
  {
    m=a_cols;
  }
  else
  {
    m=a_rows;
  }

  int k;
  if(ta==true)
  {
    k=a_rows;
  }
  else
  {
    k=a_cols;
  }

  int n;
  if(tb==true)
  {
    n=b_rows;
  }
  else
  {
    n=b_cols;
  }

  const float *a_data=static_cast<const float *>(a.RawData());
  const float *b_data=static_cast<const float *>(b.RawData());
  float *result_data=static_cast<float *>(result.MutableRawData());

  using RowMajorMatrix=Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  Eigen::Map<const RowMajorMatrix> eigen_a_map(a_data, a_rows, a_cols);
  Eigen::Map<const RowMajorMatrix> eigen_b_map(b_data, b_rows, b_cols);
  Eigen::Map<RowMajorMatrix> eigen_result_map(result_data, m, n);

  if(ta==false&&tb==false)
  {
    eigen_result_map=eigen_a_map*eigen_b_map;
  }
  else if(ta==true&&tb==false)
  {
    eigen_result_map=eigen_a_map.transpose()*eigen_b_map;
  }
  else if(ta==false&&tb==true)
  {
    eigen_result_map=eigen_a_map*eigen_b_map.transpose();
  }
  else
  {
    eigen_result_map=eigen_a_map.transpose()*eigen_b_map.transpose();
  }

  return;
}

void CAIF_EigenBackend::Convolution2D(
                                     const CAIF_TensorData &input,
                                     const CAIF_TensorData &kernel,
                                     CAIF_TensorData &output,
                                     const CAIF_TensorBackend::ConvolutionParams &params
                                    )
{
  // Cast to Eigen tensor data for access to shape information
  const CAIF_CPUTensorData &eigen_input=static_cast<const CAIF_CPUTensorData &>(input);
  const CAIF_CPUTensorData &eigen_kernel=static_cast<const CAIF_CPUTensorData &>(kernel);
  CAIF_CPUTensorData &eigen_output=static_cast<CAIF_CPUTensorData &>(output);
  
  const auto &input_shape=eigen_input.Shape();
  const auto &kernel_shape=eigen_kernel.Shape();
  const auto &output_shape=eigen_output.Shape();
  
  // Validate input shapes (expecting NHWC format: batch, height, width, channels)
  if(input_shape.size()!=4||kernel_shape.size()!=4||output_shape.size()!=4)
  {
    THROW_CAIFE("Convolution requires 4D tensors (NHWC format)");
  }
  
  const uint32_t batch_size=input_shape[0];
  const uint32_t input_height=input_shape[1];
  const uint32_t input_width=input_shape[2];
  const uint32_t input_channels=input_shape[3];
  
  const uint32_t kernel_height=kernel_shape[0];
  const uint32_t kernel_width=kernel_shape[1];
  const uint32_t kernel_input_channels=kernel_shape[2];
  const uint32_t kernel_output_channels=kernel_shape[3];
  
  const uint32_t output_height=output_shape[1];
  const uint32_t output_width=output_shape[2];
  const uint32_t output_channels=output_shape[3];
  
  // Validate channel compatibility
  if(input_channels!=kernel_input_channels||output_channels!=kernel_output_channels)
  {
    THROW_CAIFE("Channel dimensions mismatch in convolution");
  }
  
  // Calculate expected output dimensions
  const uint32_t expected_output_height=(input_height+2*params.padding_y-kernel_height)/params.stride_y+1;
  const uint32_t expected_output_width=(input_width+2*params.padding_x-kernel_width)/params.stride_x+1;
  
  if(output_height!=expected_output_height||output_width!=expected_output_width)
  {
    THROW_CAIFE("Output dimensions don't match expected convolution result");
  }
  
  // Get data pointers
  const float *input_data=static_cast<const float *>(input.RawData());
  const float *kernel_data=static_cast<const float *>(kernel.RawData());
  float *output_data=static_cast<float *>(output.MutableRawData());
  
  // Perform convolution using Eigen tensors
  // This is a simplified implementation - a full implementation would use optimized convolution algorithms
  for(uint32_t b=0;b<batch_size;++b)
  {
    for(uint32_t oc=0;oc<output_channels;++oc)
    {
      for(uint32_t oh=0;oh<output_height;++oh)
      {
        for(uint32_t ow=0;ow<output_width;++ow)
        {
          float sum=0.0f;
          
          for(uint32_t kh=0;kh<kernel_height;++kh)
          {
            for(uint32_t kw=0;kw<kernel_width;++kw)
            {
              for(uint32_t ic=0;ic<input_channels;++ic)
              {
                const int32_t ih=
                  static_cast<int32_t>(oh*params.stride_y+kh)-
                  static_cast<int32_t>(params.padding_y);
                const int32_t iw=
                  static_cast<int32_t>(ow*params.stride_x+kw)-
                  static_cast<int32_t>(params.padding_x);

                // Check bounds (padding)
                if(ih>=0&&ih<static_cast<int32_t>(input_height)&&
                   iw>=0&&iw<static_cast<int32_t>(input_width))
                {
                  const size_t input_idx=b*
                                         input_height*
                                         input_width*
                                         input_channels+
                                         ih*input_width*
                                         input_channels+
                                         iw*
                                         input_channels+
                                         ic;
                  
                  const size_t kernel_idx=kh*
                                          kernel_width*
                                          kernel_input_channels*
                                          kernel_output_channels+
                                          kw*
                                          kernel_input_channels*
                                          kernel_output_channels+
                                          ic*
                                          kernel_output_channels+
                                          oc;
                  
                  sum+=input_data[input_idx]*kernel_data[kernel_idx];
                }
              }
            }
          }
          
          const size_t output_idx=b*
                                  output_height*
                                  output_width*
                                  output_channels+
                                  oh*
                                  output_width*
                                  output_channels+
                                  ow*
                                  output_channels+
                                  oc;
          
          output_data[output_idx]=sum;
        }
      }
    }
  }
  
  return;
}

void CAIF_EigenBackend::ValidateMatrixDimensions(
                                                const CAIF_TensorData &a,
                                                const CAIF_TensorData &b,
                                                const CAIF_TensorData &result
                                               )const
{
  // Check that pointers are valid
  if(a.RawData()==nullptr||b.RawData()==nullptr||result.RawData()==nullptr)
  {
    THROW_CAIFE("Invalid tensor data pointers");
  }
  
  // Cast to Eigen tensor data to access shape information
  const CAIF_CPUTensorData &eigen_a=static_cast<const CAIF_CPUTensorData &>(a);
  const CAIF_CPUTensorData &eigen_b=static_cast<const CAIF_CPUTensorData &>(b);
  const CAIF_CPUTensorData &eigen_result=static_cast<const CAIF_CPUTensorData &>(result);
  
  const auto &shape_a=eigen_a.Shape();
  const auto &shape_b=eigen_b.Shape();
  const auto &shape_result=eigen_result.Shape();
  
  // Validate 2D matrix dimensions
  if(shape_a.size()!=g_caif_2d_matrix_dimensions ||
     shape_b.size()!=g_caif_2d_matrix_dimensions ||
     shape_result.size()!=g_caif_2d_matrix_dimensions)
  {
    THROW_CAIFE("Matrix multiplication requires 2D tensors");
  }
  
  // Check dimension compatibility: A(m,n) * B(n,p) = C(m,p)
  if(shape_a[1]!=shape_b[0])
  {
    THROW_CAIFE("Matrix A columns must equal matrix B rows");
  }
  
  if(shape_result[0]!=shape_a[0]||shape_result[1]!=shape_b[1])
  {
    THROW_CAIFE("Result matrix dimensions are incorrect");
  }
  return;
}

void CAIF_EigenBackend::MaxPooling2D(const CAIF_TensorData &input,
                                    CAIF_TensorData &output,
                                    CAIF_TensorData *indices,
                                    const PoolingParams &params)
{
  const auto &in_shape=input.Shape();
  const auto &out_shape=output.Shape();

  if(in_shape.size()!=4||out_shape.size()!=4)
  {
    THROW_CAIFE("Pooling requires 4D tensors (NHWC)");
  }

  const uint32_t batch_size=in_shape[0];
  const uint32_t input_height=in_shape[1];
  const uint32_t input_width=in_shape[2];
  const uint32_t channels=in_shape[3];
  const uint32_t output_height=out_shape[1];
  const uint32_t output_width=out_shape[2];

  const float *input_data=static_cast<const float *>(input.RawData());
  float *output_data=static_cast<float *>(output.MutableRawData());
  uint32_t *indices_data=nullptr;
  if(indices!=nullptr)
  {
    indices_data=static_cast<uint32_t *>(indices->MutableRawData());
  }

  for(uint32_t b=0;b<batch_size;++b)
  {
    for(uint32_t c=0;c<channels;++c)
    {
      for(uint32_t oh=0;oh<output_height;++oh)
      {
        for(uint32_t ow=0;ow<output_width;++ow)
        {
          float max_val=-std::numeric_limits<float>::infinity();
          uint32_t max_idx=0;

          const uint32_t h_start=oh*params.stride_y;
          const uint32_t w_start=ow*params.stride_x;

          for(uint32_t ph=0;ph<params.pool_height;++ph)
          {
            for(uint32_t pw=0;pw<params.pool_width;++pw)
            {
              const uint32_t h=h_start+ph;
              const uint32_t w=w_start+pw;

              if(h<input_height&&w<input_width)
              {
                const uint32_t input_idx=b*input_height*input_width*channels+
                                        h*input_width*channels+
                                        w*channels+c;
                if(input_data[input_idx]>max_val)
                {
                  max_val=input_data[input_idx];
                  max_idx=input_idx;
                }
              }
            }
          }

          const uint32_t output_idx=b*output_height*output_width*channels+
                                   oh*output_width*channels+
                                   ow*channels+c;
          output_data[output_idx]=max_val;
          if(indices_data!=nullptr)
          {
            indices_data[output_idx]=max_idx;
          }
        }
      }
    }
  }
}

void CAIF_EigenBackend::AveragePooling2D(const CAIF_TensorData &input,
                                        CAIF_TensorData &output,
                                        const PoolingParams &params)
{
  const auto &in_shape=input.Shape();
  const auto &out_shape=output.Shape();

  if(in_shape.size()!=4||out_shape.size()!=4)
  {
    THROW_CAIFE("Pooling requires 4D tensors (NHWC)");
  }

  const uint32_t batch_size=in_shape[0];
  const uint32_t input_height=in_shape[1];
  const uint32_t input_width=in_shape[2];
  const uint32_t channels=in_shape[3];
  const uint32_t output_height=out_shape[1];
  const uint32_t output_width=out_shape[2];

  const float *input_data=static_cast<const float *>(input.RawData());
  float *output_data=static_cast<float *>(output.MutableRawData());

  for(uint32_t b=0;b<batch_size;++b)
  {
    for(uint32_t c=0;c<channels;++c)
    {
      for(uint32_t oh=0;oh<output_height;++oh)
      {
        for(uint32_t ow=0;ow<output_width;++ow)
        {
          float sum=0.0f;
          uint32_t count=0;

          const uint32_t h_start=oh*params.stride_y;
          const uint32_t w_start=ow*params.stride_x;

          for(uint32_t ph=0;ph<params.pool_height;++ph)
          {
            for(uint32_t pw=0;pw<params.pool_width;++pw)
            {
              const uint32_t h=h_start+ph;
              const uint32_t w=w_start+pw;

              if(h<input_height&&w<input_width)
              {
                const uint32_t input_idx=b*input_height*input_width*channels+
                                        h*input_width*channels+
                                        w*channels+c;
                sum+=input_data[input_idx];
                ++count;
              }
            }
          }

          const uint32_t output_idx=b*output_height*output_width*channels+
                                   oh*output_width*channels+
                                   ow*channels+c;
          if(count>0)
          {
            output_data[output_idx]=sum/static_cast<float>(count);
          }
          else
          {
            output_data[output_idx]=0.0f;
          }
        }
      }
    }
  }
}

void CAIF_EigenBackend::BatchNormForward(const CAIF_TensorData &input,
                                        CAIF_TensorData &output,
                                        const CAIF_TensorData &scale,
                                        const CAIF_TensorData &bias,
                                        CAIF_TensorData &running_mean,
                                        CAIF_TensorData &running_var,
                                        CAIF_TensorData &saved_mean,
                                        CAIF_TensorData &saved_inv_var,
                                        const BatchNormParams &params,
                                        const bool training)
{
  const auto &in_shape=input.Shape();
  if(in_shape.empty())
  {
    THROW_CAIFE("Batch norm requires non-empty input");
  }

  const uint32_t num_features=in_shape.back();
  // Calculate total elements from shape
  size_t num_elements=1;
  for(const auto &dim:in_shape)
  {
    num_elements*=dim;
  }
  const size_t num_elements_per_feature=num_elements/num_features;

  const float *input_data=static_cast<const float *>(input.RawData());
  float *output_data=static_cast<float *>(output.MutableRawData());
  const float *scale_data=static_cast<const float *>(scale.RawData());
  const float *bias_data=static_cast<const float *>(bias.RawData());
  float *running_mean_data=static_cast<float *>(running_mean.MutableRawData());
  float *running_var_data=static_cast<float *>(running_var.MutableRawData());
  float *saved_mean_data=static_cast<float *>(saved_mean.MutableRawData());
  float *saved_inv_var_data=static_cast<float *>(saved_inv_var.MutableRawData());

  std::vector<float> batch_mean(num_features,0.0f);
  std::vector<float> batch_var(num_features,0.0f);

  if(training==true)
  {
    // Compute batch mean
    for(size_t i=0;i<num_elements;++i)
    {
      const uint32_t feature_idx=i%num_features;
      batch_mean[feature_idx]+=input_data[i];
    }
    for(uint32_t f=0;f<num_features;++f)
    {
      batch_mean[f]/=static_cast<float>(num_elements_per_feature);
    }

    // Compute batch variance
    for(size_t i=0;i<num_elements;++i)
    {
      const uint32_t feature_idx=i%num_features;
      const float diff=input_data[i]-batch_mean[feature_idx];
      batch_var[feature_idx]+=diff*diff;
    }
    for(uint32_t f=0;f<num_features;++f)
    {
      batch_var[f]/=static_cast<float>(num_elements_per_feature);
    }

    // Update running statistics
    for(uint32_t f=0;f<num_features;++f)
    {
      running_mean_data[f]=params.momentum*running_mean_data[f]+
                           (1.0f-params.momentum)*batch_mean[f];
      running_var_data[f]=params.momentum*running_var_data[f]+
                          (1.0f-params.momentum)*batch_var[f];
      saved_mean_data[f]=batch_mean[f];
      saved_inv_var_data[f]=1.0f/std::sqrt(batch_var[f]+params.epsilon);
    }

    // Apply normalization
    for(size_t i=0;i<num_elements;++i)
    {
      const uint32_t feature_idx=i%num_features;
      const float normalized=(input_data[i]-batch_mean[feature_idx])/
                              std::sqrt(batch_var[feature_idx]+params.epsilon);
      output_data[i]=scale_data[feature_idx]*normalized+bias_data[feature_idx];
    }
  }
  else
  {
    // Use running statistics for inference
    for(size_t i=0;i<num_elements;++i)
    {
      const uint32_t feature_idx=i%num_features;
      const float normalized=(input_data[i]-running_mean_data[feature_idx])/
                              std::sqrt(running_var_data[feature_idx]+params.epsilon);
      output_data[i]=scale_data[feature_idx]*normalized+bias_data[feature_idx];
    }
  }
}

void CAIF_EigenBackend::ActivationForward(const CAIF_TensorData &input,
                                         CAIF_TensorData &output,
                                         const ActivationType_e activation_type)
{
  // Calculate total elements from shape
  const auto &act_shape=input.Shape();
  size_t num_elements=1;
  for(const auto &dim:act_shape)
  {
    num_elements*=dim;
  }
  const float *input_data=static_cast<const float *>(input.RawData());
  float *output_data=static_cast<float *>(output.MutableRawData());

  if(activation_type==ActivationType_e::Identity)
  {
    std::memcpy(output_data,input_data,num_elements*sizeof(float));
  }
  else if(activation_type==ActivationType_e::ReLU)
  {
    for(size_t i=0;i<num_elements;++i)
    {
      if(input_data[i]>0.0f)
      {
        output_data[i]=input_data[i];
      }
      else
      {
        output_data[i]=0.0f;
      }
    }
  }
  else if(activation_type==ActivationType_e::Sigmoid)
  {
    for(size_t i=0;i<num_elements;++i)
    {
      output_data[i]=1.0f/(1.0f+std::exp(-input_data[i]));
    }
  }
  else if(activation_type==ActivationType_e::Tanh)
  {
    for(size_t i=0;i<num_elements;++i)
    {
      output_data[i]=std::tanh(input_data[i]);
    }
  }
  else if(activation_type==ActivationType_e::LeakyReLU)
  {
    constexpr float alpha=0.01f;
    for(size_t i=0;i<num_elements;++i)
    {
      if(input_data[i]>0.0f)
      {
        output_data[i]=input_data[i];
      }
      else
      {
        output_data[i]=alpha*input_data[i];
      }
    }
  }
  else if(activation_type==ActivationType_e::ELU)
  {
    constexpr float alpha=1.0f;
    for(size_t i=0;i<num_elements;++i)
    {
      if(input_data[i]>0.0f)
      {
        output_data[i]=input_data[i];
      }
      else
      {
        output_data[i]=alpha*(std::exp(input_data[i])-1.0f);
      }
    }
  }
  else if(activation_type==ActivationType_e::GELU)
  {
    // GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    constexpr float sqrt2_inv=0.7071067811865476f;
    for(size_t i=0;i<num_elements;++i)
    {
      output_data[i]=input_data[i]*0.5f*(1.0f+std::erf(input_data[i]*sqrt2_inv));
    }
  }
  else if(activation_type==ActivationType_e::Swish)
  {
    // Swish(x) = x * sigmoid(x)
    for(size_t i=0;i<num_elements;++i)
    {
      const float sigmoid=1.0f/(1.0f+std::exp(-input_data[i]));
      output_data[i]=input_data[i]*sigmoid;
    }
  }
  else
  {
    THROW_CAIFE("Unknown activation type");
  }
}

void CAIF_EigenBackend::SoftmaxForward(const CAIF_TensorData &input,CAIF_TensorData &output)
{
  const auto &shape=input.Shape();
  if(shape.size()!=2)
  {
    THROW_CAIFE("Softmax currently only supports 2D tensors [batch, features]");
  }

  const uint32_t batch=shape[0];
  const uint32_t features=shape[1];

  const float *input_data=static_cast<const float *>(input.RawData());
  float *output_data=static_cast<float *>(output.MutableRawData());

  for(uint32_t b=0;b<batch;++b)
  {
    // Find max for numerical stability
    float max_val=input_data[b*features];
    for(uint32_t f=1;f<features;++f)
    {
      if(input_data[b*features+f]>max_val)
      {
        max_val=input_data[b*features+f];
      }
    }

    // Compute exp and sum
    float sum=0.0f;
    for(uint32_t f=0;f<features;++f)
    {
      output_data[b*features+f]=std::exp(input_data[b*features+f]-max_val);
      sum+=output_data[b*features+f];
    }

    // Normalize
    for(uint32_t f=0;f<features;++f)
    {
      output_data[b*features+f]/=sum;
    }
  }
}

void CAIF_EigenBackend::Convolution2DBackwardData(const CAIF_TensorData &grad_output,
                                                 const CAIF_TensorData &kernel,
                                                 CAIF_TensorData &grad_input,
                                                 const ConvolutionParams &params)
{
  const auto &grad_out_shape=grad_output.Shape();
  const auto &ker_shape=kernel.Shape();
  const auto &grad_in_shape=grad_input.Shape();

  if(grad_out_shape.size()!=4||ker_shape.size()!=4||grad_in_shape.size()!=4)
  {
    THROW_CAIFE("Convolution backward data requires 4D tensors (NHWC)");
  }

  const uint32_t batch=grad_out_shape[0];
  const uint32_t out_h=grad_out_shape[1];
  const uint32_t out_w=grad_out_shape[2];
  const uint32_t out_c=grad_out_shape[3];

  const uint32_t k_h=ker_shape[0];
  const uint32_t k_w=ker_shape[1];
  const uint32_t in_c=ker_shape[2];

  const uint32_t in_h=grad_in_shape[1];
  const uint32_t in_w=grad_in_shape[2];

  const float *grad_out_data=static_cast<const float *>(grad_output.RawData());
  const float *kernel_data=static_cast<const float *>(kernel.RawData());
  float *grad_in_data=static_cast<float *>(grad_input.MutableRawData());

  // Initialize gradient to zero
  const size_t grad_in_size=batch*in_h*in_w*in_c;
  std::memset(grad_in_data,0,grad_in_size*sizeof(float));

  for(uint32_t b=0;b<batch;++b)
  {
    for(uint32_t oc=0;oc<out_c;++oc)
    {
      for(uint32_t oh=0;oh<out_h;++oh)
      {
        for(uint32_t ow=0;ow<out_w;++ow)
        {
          const size_t grad_out_idx=b*out_h*out_w*out_c+oh*out_w*out_c+ow*out_c+oc;
          const float grad_out_val=grad_out_data[grad_out_idx];

          for(uint32_t kh=0;kh<k_h;++kh)
          {
            for(uint32_t kw=0;kw<k_w;++kw)
            {
              for(uint32_t ic=0;ic<in_c;++ic)
              {
                const int32_t ih=
                  static_cast<int32_t>(oh*params.stride_y+kh)-
                  static_cast<int32_t>(params.padding_y);
                const int32_t iw=
                  static_cast<int32_t>(ow*params.stride_x+kw)-
                  static_cast<int32_t>(params.padding_x);

                if(ih>=0&&ih<static_cast<int32_t>(in_h) &&
                   iw>=0&&iw<static_cast<int32_t>(in_w))
                {
                  const size_t grad_in_idx=b*in_h*in_w*in_c+ih*in_w*in_c+iw*in_c+ic;
                  const size_t ker_idx=kh*k_w*in_c*out_c+kw*in_c*out_c+ic*out_c+oc;
                  grad_in_data[grad_in_idx]+=grad_out_val*kernel_data[ker_idx];
                }
              }
            }
          }
        }
      }
    }
  }
}

void CAIF_EigenBackend::Convolution2DBackwardFilter(const CAIF_TensorData &input,
                                                   const CAIF_TensorData &grad_output,
                                                   CAIF_TensorData &grad_kernel,
                                                   const ConvolutionParams &params)
{
  const auto &in_shape=input.Shape();
  const auto &grad_out_shape=grad_output.Shape();
  const auto &grad_ker_shape=grad_kernel.Shape();

  if(in_shape.size()!=4||grad_out_shape.size()!=4||grad_ker_shape.size()!=4)
  {
    THROW_CAIFE("Convolution backward filter requires 4D tensors (NHWC)");
  }

  const uint32_t batch=in_shape[0];
  const uint32_t in_h=in_shape[1];
  const uint32_t in_w=in_shape[2];
  const uint32_t in_c=in_shape[3];

  const uint32_t out_h=grad_out_shape[1];
  const uint32_t out_w=grad_out_shape[2];
  const uint32_t out_c=grad_out_shape[3];

  const uint32_t k_h=grad_ker_shape[0];
  const uint32_t k_w=grad_ker_shape[1];

  const float *input_data=static_cast<const float *>(input.RawData());
  const float *grad_out_data=static_cast<const float *>(grad_output.RawData());
  float *grad_ker_data=static_cast<float *>(grad_kernel.MutableRawData());

  // Initialize gradient to zero
  const size_t grad_ker_size=k_h*k_w*in_c*out_c;
  std::memset(grad_ker_data,0,grad_ker_size*sizeof(float));

  for(uint32_t b=0;b<batch;++b)
  {
    for(uint32_t oc=0;oc<out_c;++oc)
    {
      for(uint32_t oh=0;oh<out_h;++oh)
      {
        for(uint32_t ow=0;ow<out_w;++ow)
        {
          const size_t grad_out_idx=b*out_h*out_w*out_c+oh*out_w*out_c+ow*out_c+oc;
          const float grad_out_val=grad_out_data[grad_out_idx];

          for(uint32_t kh=0;kh<k_h;++kh)
          {
            for(uint32_t kw=0;kw<k_w;++kw)
            {
              for(uint32_t ic=0;ic<in_c;++ic)
              {
                const int32_t ih=
                  static_cast<int32_t>(oh*params.stride_y+kh)-
                  static_cast<int32_t>(params.padding_y);
                const int32_t iw=
                  static_cast<int32_t>(ow*params.stride_x+kw)-
                  static_cast<int32_t>(params.padding_x);

                if(ih>=0&&ih<static_cast<int32_t>(in_h) &&
                   iw>=0&&iw<static_cast<int32_t>(in_w))
                {
                  const size_t in_idx=b*in_h*in_w*in_c+ih*in_w*in_c+iw*in_c+ic;
                  const size_t ker_idx=kh*k_w*in_c*out_c+kw*in_c*out_c+ic*out_c+oc;
                  grad_ker_data[ker_idx]+=grad_out_val*input_data[in_idx];
                }
              }
            }
          }
        }
      }
    }
  }
}

void CAIF_EigenBackend::MaxPooling2DBackward(const CAIF_TensorData &grad_output,
                                            const CAIF_TensorData *indices,
                                            const CAIF_TensorData &input,
                                            CAIF_TensorData &grad_input,
                                            const PoolingParams &params)
{
  const auto &in_shape=input.Shape();
  const auto &grad_out_shape=grad_output.Shape();

  if(in_shape.size()!=4||grad_out_shape.size()!=4)
  {
    THROW_CAIFE("Max pooling backward requires 4D tensors (NHWC)");
  }

  const uint32_t batch=in_shape[0];
  const uint32_t in_h=in_shape[1];
  const uint32_t in_w=in_shape[2];
  const uint32_t channels=in_shape[3];
  const uint32_t out_h=grad_out_shape[1];
  const uint32_t out_w=grad_out_shape[2];

  const float *grad_out_data=static_cast<const float *>(grad_output.RawData());
  const float *input_data=static_cast<const float *>(input.RawData());
  float *grad_in_data=static_cast<float *>(grad_input.MutableRawData());

  // Initialize gradient to zero
  const size_t grad_in_size=batch*in_h*in_w*channels;
  std::memset(grad_in_data,0,grad_in_size*sizeof(float));

  if(indices!=nullptr)
  {
    // Use stored indices
    const uint32_t *indices_data=static_cast<const uint32_t *>(indices->RawData());
    for(uint32_t b=0;b<batch;++b)
    {
      for(uint32_t c=0;c<channels;++c)
      {
        for(uint32_t oh=0;oh<out_h;++oh)
        {
          for(uint32_t ow=0;ow<out_w;++ow)
          {
            const uint32_t out_idx=b*out_h*out_w*channels+oh*out_w*channels+ow*channels+c;
            const uint32_t max_idx=indices_data[out_idx];
            grad_in_data[max_idx]+=grad_out_data[out_idx];
          }
        }
      }
    }
  }
  else
  {
    // Recompute max indices
    for(uint32_t b=0;b<batch;++b)
    {
      for(uint32_t c=0;c<channels;++c)
      {
        for(uint32_t oh=0;oh<out_h;++oh)
        {
          for(uint32_t ow=0;ow<out_w;++ow)
          {
            const uint32_t h_start=oh*params.stride_y;
            const uint32_t w_start=ow*params.stride_x;

            float max_val=-std::numeric_limits<float>::infinity();
            uint32_t max_idx=0;

            for(uint32_t ph=0;ph<params.pool_height;++ph)
            {
              for(uint32_t pw=0;pw<params.pool_width;++pw)
              {
                const uint32_t h=h_start+ph;
                const uint32_t w=w_start+pw;

                if(h<in_h&&w<in_w)
                {
                  const uint32_t in_idx=b*in_h*in_w*channels+h*in_w*channels+w*channels+c;
                  if(input_data[in_idx]>max_val)
                  {
                    max_val=input_data[in_idx];
                    max_idx=in_idx;
                  }
                }
              }
            }

            const uint32_t out_idx=b*out_h*out_w*channels+oh*out_w*channels+ow*channels+c;
            grad_in_data[max_idx]+=grad_out_data[out_idx];
          }
        }
      }
    }
  }
}

void CAIF_EigenBackend::AveragePooling2DBackward(const CAIF_TensorData &grad_output,
                                                CAIF_TensorData &grad_input,
                                                const PoolingParams &params)
{
  const auto &grad_in_shape=grad_input.Shape();
  const auto &grad_out_shape=grad_output.Shape();

  if(grad_in_shape.size()!=4||grad_out_shape.size()!=4)
  {
    THROW_CAIFE("Avg pooling backward requires 4D tensors (NHWC)");
  }

  const uint32_t batch=grad_in_shape[0];
  const uint32_t in_h=grad_in_shape[1];
  const uint32_t in_w=grad_in_shape[2];
  const uint32_t channels=grad_in_shape[3];
  const uint32_t out_h=grad_out_shape[1];
  const uint32_t out_w=grad_out_shape[2];

  const float *grad_out_data=static_cast<const float *>(grad_output.RawData());
  float *grad_in_data=static_cast<float *>(grad_input.MutableRawData());

  // Initialize gradient to zero
  const size_t grad_in_size=batch*in_h*in_w*channels;
  std::memset(grad_in_data,0,grad_in_size*sizeof(float));

  for(uint32_t b=0;b<batch;++b)
  {
    for(uint32_t c=0;c<channels;++c)
    {
      for(uint32_t oh=0;oh<out_h;++oh)
      {
        for(uint32_t ow=0;ow<out_w;++ow)
        {
          const uint32_t h_start=oh*params.stride_y;
          const uint32_t w_start=ow*params.stride_x;

          // Count elements in pool window
          uint32_t count=0;
          for(uint32_t ph=0;ph<params.pool_height;++ph)
          {
            for(uint32_t pw=0;pw<params.pool_width;++pw)
            {
              const uint32_t h=h_start+ph;
              const uint32_t w=w_start+pw;
              if(h<in_h&&w<in_w)
              {
                ++count;
              }
            }
          }

          const uint32_t out_idx=b*out_h*out_w*channels+oh*out_w*channels+ow*channels+c;
          const float grad_per_element=(count>0)?grad_out_data[out_idx]/static_cast<float>(count):0.0f;

          // Distribute gradient evenly
          for(uint32_t ph=0;ph<params.pool_height;++ph)
          {
            for(uint32_t pw=0;pw<params.pool_width;++pw)
            {
              const uint32_t h=h_start+ph;
              const uint32_t w=w_start+pw;
              if(h<in_h&&w<in_w)
              {
                const uint32_t in_idx=b*in_h*in_w*channels+h*in_w*channels+w*channels+c;
                grad_in_data[in_idx]+=grad_per_element;
              }
            }
          }
        }
      }
    }
  }
}

void CAIF_EigenBackend::BatchNormBackward(const CAIF_TensorData &grad_output,
                                         const CAIF_TensorData &input,
                                         const CAIF_TensorData &scale,
                                         const CAIF_TensorData &saved_mean,
                                         const CAIF_TensorData &saved_inv_var,
                                         CAIF_TensorData &grad_input,
                                         CAIF_TensorData &grad_scale,
                                         CAIF_TensorData &grad_bias,
                                         const BatchNormParams &params)
{
  (void)params;
  const auto &in_shape=input.Shape();
  if(in_shape.empty())
  {
    THROW_CAIFE("Batch norm backward requires non-empty input");
  }

  const uint32_t num_features=in_shape.back();
  size_t num_elements=1;
  for(const auto &dim:in_shape)
  {
    num_elements*=dim;
  }
  const size_t num_elements_per_feature=num_elements/num_features;

  const float *grad_out_data=static_cast<const float *>(grad_output.RawData());
  const float *input_data=static_cast<const float *>(input.RawData());
  const float *scale_data=static_cast<const float *>(scale.RawData());
  const float *mean_data=static_cast<const float *>(saved_mean.RawData());
  const float *inv_var_data=static_cast<const float *>(saved_inv_var.RawData());
  float *grad_in_data=static_cast<float *>(grad_input.MutableRawData());
  float *grad_scale_data=static_cast<float *>(grad_scale.MutableRawData());
  float *grad_bias_data=static_cast<float *>(grad_bias.MutableRawData());

  // Initialize gradients to zero
  std::memset(grad_scale_data,0,num_features*sizeof(float));
  std::memset(grad_bias_data,0,num_features*sizeof(float));

  // Compute grad_scale and grad_bias
  for(size_t i=0;i<num_elements;++i)
  {
    const uint32_t f=i%num_features;
    const float x_hat=(input_data[i]-mean_data[f])*inv_var_data[f];
    grad_scale_data[f]+=grad_out_data[i]*x_hat;
    grad_bias_data[f]+=grad_out_data[i];
  }

  // Compute grad_input
  const float m=static_cast<float>(num_elements_per_feature);
  for(size_t i=0;i<num_elements;++i)
  {
    const uint32_t f=i%num_features;
    const float x_hat=(input_data[i]-mean_data[f])*inv_var_data[f];
    grad_in_data[i]=scale_data[f]*inv_var_data[f]*(grad_out_data[i]-grad_bias_data[f]/m-
                     x_hat*grad_scale_data[f]/m);
  }
}

void CAIF_EigenBackend::ActivationBackward(const CAIF_TensorData &grad_output,
                                          const CAIF_TensorData &input,
                                          const CAIF_TensorData &output,
                                          CAIF_TensorData &grad_input,
                                          const ActivationType_e activation_type)
{
  const auto &shape=input.Shape();
  size_t num_elements=1;
  for(const auto &dim:shape)
  {
    num_elements*=dim;
  }

  const float *grad_out_data=static_cast<const float *>(grad_output.RawData());
  const float *input_data=static_cast<const float *>(input.RawData());
  const float *output_data=static_cast<const float *>(output.RawData());
  float *grad_in_data=static_cast<float *>(grad_input.MutableRawData());

  if(activation_type==ActivationType_e::Identity)
  {
    std::memcpy(grad_in_data,grad_out_data,num_elements*sizeof(float));
  }
  else if(activation_type==ActivationType_e::ReLU)
  {
    for(size_t i=0;i<num_elements;++i)
    {
      if(input_data[i]>0.0f)
      {
        grad_in_data[i]=grad_out_data[i];
      }
      else
      {
        grad_in_data[i]=0.0f;
      }
    }
  }
  else if(activation_type==ActivationType_e::Sigmoid)
  {
    for(size_t i=0;i<num_elements;++i)
    {
      grad_in_data[i]=grad_out_data[i]*output_data[i]*(1.0f-output_data[i]);
    }
  }
  else if(activation_type==ActivationType_e::Tanh)
  {
    for(size_t i=0;i<num_elements;++i)
    {
      grad_in_data[i]=grad_out_data[i]*(1.0f-output_data[i]*output_data[i]);
    }
  }
  else if(activation_type==ActivationType_e::LeakyReLU)
  {
    constexpr float alpha=0.01f;
    for(size_t i=0;i<num_elements;++i)
    {
      if(input_data[i]>0.0f)
      {
        grad_in_data[i]=grad_out_data[i];
      }
      else
      {
        grad_in_data[i]=grad_out_data[i]*alpha;
      }
    }
  }
  else if(activation_type==ActivationType_e::ELU)
  {
    constexpr float alpha=1.0f;
    for(size_t i=0;i<num_elements;++i)
    {
      if(input_data[i]>0.0f)
      {
        grad_in_data[i]=grad_out_data[i];
      }
      else
      {
        grad_in_data[i]=grad_out_data[i]*(output_data[i]+alpha);
      }
    }
  }
  else if(activation_type==ActivationType_e::GELU)
  {
    // GELU derivative: 0.5 * (1 + erf(x/sqrt(2))) + x * exp(-x^2/2) / sqrt(2*pi)
    constexpr float sqrt2_inv=0.7071067811865476f;
    constexpr float sqrt2pi_inv=0.3989422804014327f;
    for(size_t i=0;i<num_elements;++i)
    {
      const float x=input_data[i];
      const float cdf=0.5f*(1.0f+std::erf(x*sqrt2_inv));
      const float pdf=sqrt2pi_inv*std::exp(-0.5f*x*x);
      grad_in_data[i]=grad_out_data[i]*(cdf+x*pdf);
    }
  }
  else if(activation_type==ActivationType_e::Swish)
  {
    // Swish derivative: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    //                 = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    for(size_t i=0;i<num_elements;++i)
    {
      const float x=input_data[i];
      const float sigmoid=1.0f/(1.0f+std::exp(-x));
      grad_in_data[i]=grad_out_data[i]*sigmoid*(1.0f+x*(1.0f-sigmoid));
    }
  }
  else
  {
    THROW_CAIFE("Unknown activation type in backward pass");
  }
}

void CAIF_EigenBackend::SoftmaxBackward(const CAIF_TensorData &grad_output,
                                       const CAIF_TensorData &output,
                                       CAIF_TensorData &grad_input)
{
  const auto &shape=output.Shape();
  if(shape.size()!=2)
  {
    THROW_CAIFE("Softmax backward currently only supports 2D tensors [batch, features]");
  }

  const uint32_t batch=shape[0];
  const uint32_t features=shape[1];

  const float *grad_out_data=static_cast<const float *>(grad_output.RawData());
  const float *output_data=static_cast<const float *>(output.RawData());
  float *grad_in_data=static_cast<float *>(grad_input.MutableRawData());

  for(uint32_t b=0;b<batch;++b)
  {
    // Compute dot product of grad_output and output for this batch
    float dot=0.0f;
    for(uint32_t f=0;f<features;++f)
    {
      dot+=grad_out_data[b*features+f]*output_data[b*features+f];
    }

    // Compute gradient: grad_input = output * (grad_output - dot)
    for(uint32_t f=0;f<features;++f)
    {
      grad_in_data[b*features+f]=output_data[b*features+f]*(grad_out_data[b*features+f]-dot);
    }
  }
}

void CAIF_EigenBackend::DropoutForward(const CAIF_TensorData &input,
                                      CAIF_TensorData &output,
                                      CAIF_TensorData &mask,
                                      const float dropout_rate,
                                      const bool training)
{
  const auto &shape=input.Shape();
  size_t num_elements=1;
  for(const auto &dim:shape)
  {
    num_elements*=dim;
  }

  const float *input_data=static_cast<const float *>(input.RawData());
  float *output_data=static_cast<float *>(output.MutableRawData());
  float *mask_data=static_cast<float *>(mask.MutableRawData());

  if(training==false||dropout_rate<=0.0f)
  {
    // Inference mode or no dropout - pass through
    std::memcpy(output_data,input_data,num_elements*sizeof(float));
    // Set mask to all ones
    for(size_t i=0;i<num_elements;++i)
    {
      mask_data[i]=1.0f;
    }
    return;
  }

  // Inverted dropout: scale by 1/(1-p) during training
  const float scale=1.0f/(1.0f-dropout_rate);

  for(size_t i=0;i<num_elements;++i)
  {
    const float rand_val=static_cast<float>(std::rand())/static_cast<float>(RAND_MAX);
    if(rand_val<dropout_rate)
    {
      mask_data[i]=0.0f;
      output_data[i]=0.0f;
    }
    else
    {
      mask_data[i]=scale;
      output_data[i]=input_data[i]*scale;
    }
  }
}

void CAIF_EigenBackend::DropoutBackward(const CAIF_TensorData &grad_output,
                                       const CAIF_TensorData &mask,
                                       CAIF_TensorData &grad_input,
                                       const float dropout_rate)
{
  (void)dropout_rate;  // Scale already applied in mask during forward

  const auto &shape=grad_output.Shape();
  size_t num_elements=1;
  for(const auto &dim:shape)
  {
    num_elements*=dim;
  }

  const float *grad_out_data=static_cast<const float *>(grad_output.RawData());
  const float *mask_data=static_cast<const float *>(mask.RawData());
  float *grad_in_data=static_cast<float *>(grad_input.MutableRawData());

  // Gradient is masked the same way as forward pass
  for(size_t i=0;i<num_elements;++i)
  {
    grad_in_data[i]=grad_out_data[i]*mask_data[i];
  }
}

//------------------------------------------------------------------------------
// Element-wise Operations
//------------------------------------------------------------------------------

void CAIF_EigenBackend::ElementwiseAdd(const CAIF_TensorData &a,
                                      const CAIF_TensorData &b,
                                      CAIF_TensorData &result)
{
  size_t num_elements=1;
  for(const auto &dim:a.Shape())
  {
    num_elements*=dim;
  }

  const float *a_data=static_cast<const float *>(a.RawData());
  const float *b_data=static_cast<const float *>(b.RawData());
  float *result_data=static_cast<float *>(result.MutableRawData());

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for(size_t i=0;i<num_elements;++i)
  {
    result_data[i]=a_data[i]+b_data[i];
  }
}

void CAIF_EigenBackend::ElementwiseAddScalar(const CAIF_TensorData &a,
                                            const float scalar,
                                            CAIF_TensorData &result)
{
  size_t num_elements=1;
  for(const auto &dim:a.Shape())
  {
    num_elements*=dim;
  }

  const float *a_data=static_cast<const float *>(a.RawData());
  float *result_data=static_cast<float *>(result.MutableRawData());

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for(size_t i=0;i<num_elements;++i)
  {
    result_data[i]=a_data[i]+scalar;
  }
}

void CAIF_EigenBackend::ElementwiseSub(const CAIF_TensorData &a,
                                      const CAIF_TensorData &b,
                                      CAIF_TensorData &result)
{
  size_t num_elements=1;
  for(const auto &dim:a.Shape())
  {
    num_elements*=dim;
  }

  const float *a_data=static_cast<const float *>(a.RawData());
  const float *b_data=static_cast<const float *>(b.RawData());
  float *result_data=static_cast<float *>(result.MutableRawData());

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for(size_t i=0;i<num_elements;++i)
  {
    result_data[i]=a_data[i]-b_data[i];
  }
}

void CAIF_EigenBackend::ElementwiseSubScalar(const CAIF_TensorData &a,
                                            const float scalar,
                                            CAIF_TensorData &result)
{
  size_t num_elements=1;
  for(const auto &dim:a.Shape())
  {
    num_elements*=dim;
  }

  const float *a_data=static_cast<const float *>(a.RawData());
  float *result_data=static_cast<float *>(result.MutableRawData());

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for(size_t i=0;i<num_elements;++i)
  {
    result_data[i]=a_data[i]-scalar;
  }
}

void CAIF_EigenBackend::ElementwiseMul(const CAIF_TensorData &a,
                                      const CAIF_TensorData &b,
                                      CAIF_TensorData &result)
{
  size_t num_elements=1;
  for(const auto &dim:a.Shape())
  {
    num_elements*=dim;
  }

  const float *a_data=static_cast<const float *>(a.RawData());
  const float *b_data=static_cast<const float *>(b.RawData());
  float *result_data=static_cast<float *>(result.MutableRawData());

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for(size_t i=0;i<num_elements;++i)
  {
    result_data[i]=a_data[i]*b_data[i];
  }
}

void CAIF_EigenBackend::ElementwiseMulScalar(const CAIF_TensorData &a,
                                            const float scalar,
                                            CAIF_TensorData &result)
{
  size_t num_elements=1;
  for(const auto &dim:a.Shape())
  {
    num_elements*=dim;
  }

  const float *a_data=static_cast<const float *>(a.RawData());
  float *result_data=static_cast<float *>(result.MutableRawData());

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for(size_t i=0;i<num_elements;++i)
  {
    result_data[i]=a_data[i]*scalar;
  }
}

void CAIF_EigenBackend::ElementwiseDiv(const CAIF_TensorData &a,
                                      const CAIF_TensorData &b,
                                      CAIF_TensorData &result)
{
  size_t num_elements=1;
  for(const auto &dim:a.Shape())
  {
    num_elements*=dim;
  }

  const float *a_data=static_cast<const float *>(a.RawData());
  const float *b_data=static_cast<const float *>(b.RawData());
  float *result_data=static_cast<float *>(result.MutableRawData());

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for(size_t i=0;i<num_elements;++i)
  {
    result_data[i]=a_data[i]/b_data[i];
  }
}

void CAIF_EigenBackend::ElementwiseDivScalar(const CAIF_TensorData &a,
                                            const float scalar,
                                            CAIF_TensorData &result)
{
  size_t num_elements=1;
  for(const auto &dim:a.Shape())
  {
    num_elements*=dim;
  }

  const float *a_data=static_cast<const float *>(a.RawData());
  float *result_data=static_cast<float *>(result.MutableRawData());

  const float inv_scalar=1.0f/scalar;
  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for(size_t i=0;i<num_elements;++i)
  {
    result_data[i]=a_data[i]*inv_scalar;
  }
}

void CAIF_EigenBackend::ElementwiseSqrt(const CAIF_TensorData &a,
                                       CAIF_TensorData &result)
{
  size_t num_elements=1;
  for(const auto &dim:a.Shape())
  {
    num_elements*=dim;
  }

  const float *a_data=static_cast<const float *>(a.RawData());
  float *result_data=static_cast<float *>(result.MutableRawData());

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for(size_t i=0;i<num_elements;++i)
  {
    result_data[i]=std::sqrt(a_data[i]);
  }
}

float CAIF_EigenBackend::ReduceSum(const CAIF_TensorData &a)
{
  size_t num_elements=1;
  for(const auto &dim:a.Shape())
  {
    num_elements*=dim;
  }

  const float *a_data=static_cast<const float *>(a.RawData());

  double sum=0.0;
  for(size_t i=0;i<num_elements;++i)
  {
    sum+=static_cast<double>(a_data[i]);
  }

  return static_cast<float>(sum);
}

float CAIF_EigenBackend::ReduceMean(const CAIF_TensorData &a)
{
  size_t num_elements=1;
  for(const auto &dim:a.Shape())
  {
    num_elements*=dim;
  }

  return ReduceSum(a)/static_cast<float>(num_elements);
}

//------------------------------------------------------------------------------
// Loss Function Operations
//------------------------------------------------------------------------------

void CAIF_EigenBackend::CrossEntropyLoss(const CAIF_TensorData &predictions,
                                        const CAIF_TensorData &targets,
                                        CAIF_TensorData &loss_per_sample,
                                        const float epsilon)
{
  const auto &shape=predictions.Shape();
  const uint32_t batch_size=shape[0];
  const uint32_t num_classes=shape[1];

  const float *pred_data=static_cast<const float *>(predictions.RawData());
  const float *target_data=static_cast<const float *>(targets.RawData());
  float *loss_data=static_cast<float *>(loss_per_sample.MutableRawData());

  for(uint32_t b=0;b<batch_size;++b)
  {
    float sample_loss=0.0f;
    for(uint32_t c=0;c<num_classes;++c)
    {
      const uint32_t idx=b*num_classes+c;
      float pred=pred_data[idx];
      if(pred<epsilon)
      {
        pred=epsilon;
      }
      else if(pred>1.0f-epsilon)
      {
        pred=1.0f-epsilon;
      }
      const float target=target_data[idx];
      if(target>0.0f)
      {
        sample_loss-=target*std::log(pred);
      }
    }
    loss_data[b]=sample_loss;
  }
}

void CAIF_EigenBackend::CrossEntropyGradient(const CAIF_TensorData &predictions,
                                            const CAIF_TensorData &targets,
                                            CAIF_TensorData &gradient,
                                            const float epsilon)
{
  const auto &shape=predictions.Shape();
  const uint32_t batch_size=shape[0];
  size_t num_elements=1;
  for(const auto &dim:shape)
  {
    num_elements*=dim;
  }

  const float *pred_data=static_cast<const float *>(predictions.RawData());
  const float *target_data=static_cast<const float *>(targets.RawData());
  float *grad_data=static_cast<float *>(gradient.MutableRawData());

  const float batch_size_f=static_cast<float>(batch_size);

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for(size_t i=0;i<num_elements;++i)
  {
    float pred=pred_data[i];
    if(pred<epsilon)
    {
      pred=epsilon;
    }
    else if(pred>1.0f-epsilon)
    {
      pred=1.0f-epsilon;
    }
    grad_data[i]=-target_data[i]/(pred*batch_size_f);
  }
}

void CAIF_EigenBackend::MSELoss(const CAIF_TensorData &predictions,
                               const CAIF_TensorData &targets,
                               CAIF_TensorData &loss_elements)
{
  size_t num_elements=1;
  for(const auto &dim:predictions.Shape())
  {
    num_elements*=dim;
  }

  const float *pred_data=static_cast<const float *>(predictions.RawData());
  const float *target_data=static_cast<const float *>(targets.RawData());
  float *loss_data=static_cast<float *>(loss_elements.MutableRawData());

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for(size_t i=0;i<num_elements;++i)
  {
    const float diff=pred_data[i]-target_data[i];
    loss_data[i]=diff*diff;
  }
}

void CAIF_EigenBackend::MSEGradient(const CAIF_TensorData &predictions,
                                   const CAIF_TensorData &targets,
                                   CAIF_TensorData &gradient)
{
  size_t num_elements=1;
  for(const auto &dim:predictions.Shape())
  {
    num_elements*=dim;
  }

  const float *pred_data=static_cast<const float *>(predictions.RawData());
  const float *target_data=static_cast<const float *>(targets.RawData());
  float *grad_data=static_cast<float *>(gradient.MutableRawData());

  const float scale=2.0f/static_cast<float>(num_elements);

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for(size_t i=0;i<num_elements;++i)
  {
    grad_data[i]=scale*(pred_data[i]-target_data[i]);
  }
}

//------------------------------------------------------------------------------
// Fused Adam Update (CPU implementation)
//------------------------------------------------------------------------------

void CAIF_EigenBackend::FusedAdamUpdate(
                                       CAIF_TensorData &param,
                                       const CAIF_TensorData &grad,
                                       CAIF_TensorData &m,
                                       CAIF_TensorData &v,
                                       const float lr,
                                       const float beta1,
                                       const float beta2,
                                       const float epsilon,
                                       const float weight_decay,
                                       const float bias_correction1,
                                       const float bias_correction2
                                      )
{
  size_t num_elements=1;
  for(const auto &dim:param.Shape())
  {
    num_elements*=dim;
  }

  float *param_data=static_cast<float *>(param.MutableRawData());
  const float *grad_data=static_cast<const float *>(grad.RawData());
  float *m_data=static_cast<float *>(m.MutableRawData());
  float *v_data=static_cast<float *>(v.MutableRawData());

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for(size_t i=0;i<num_elements;++i)
  {
    float p=param_data[i];
    float g=grad_data[i];

    // Update biased first moment estimate
    float m_val=beta1*m_data[i]+(1.0f-beta1)*g;
    m_data[i]=m_val;

    // Update biased second moment estimate
    float v_val=beta2*v_data[i]+(1.0f-beta2)*g*g;
    v_data[i]=v_val;

    // Compute bias-corrected moments
    const float m_hat=m_val/bias_correction1;
    const float v_hat=v_val/bias_correction2;

    // Update parameter
    p=p-lr*m_hat/(std::sqrt(v_hat)+epsilon);

    // Decoupled weight decay (AdamW)
    if(weight_decay>0.0f)
    {
      p=p-lr*weight_decay*p;
    }

    param_data[i]=p;
  }
}

//------------------------------------------------------------------------------
// Fused SGD with Momentum Update (CPU implementation)
//------------------------------------------------------------------------------

void CAIF_EigenBackend::FusedSGDMomentumUpdate(
                                              CAIF_TensorData &param,
                                              const CAIF_TensorData &grad,
                                              CAIF_TensorData &velocity,
                                              const float lr,
                                              const float momentum,
                                              const float weight_decay
                                             )
{
  size_t num_elements=1;
  for(const auto &dim:param.Shape())
  {
    num_elements*=dim;
  }

  float *param_data=static_cast<float *>(param.MutableRawData());
  const float *grad_data=static_cast<const float *>(grad.RawData());
  float *velocity_data=static_cast<float *>(velocity.MutableRawData());

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for(size_t i=0;i<num_elements;++i)
  {
    float p=param_data[i];
    float g=grad_data[i];

    // Apply weight decay
    if(weight_decay>0.0f)
    {
      g=g+weight_decay*p;
    }

    // Update velocity: v = momentum * v + grad
    float v=momentum*velocity_data[i]+g;
    velocity_data[i]=v;

    // Update parameter: param = param - lr * v
    param_data[i]=p-lr*v;
  }
}

}//end instance namespace
