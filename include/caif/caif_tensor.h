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

/**
 * @file aif_tensor.h
 * @brief Core tensor class for the CAIF (AI Framework)
 * @author CAIF Development Team
 * @version 1.0
 * @date 2024
 */

#pragma once
#include "caif_base.h"
#include "caif_constants.h"
#include "caif_data_type.h"
#include "caif_tensor_data.h"
#include "caif_error.h"
#include <vector>
#include <memory>
#include <string>
#include <concepts>
#include "caif_exception.h"
#include <span>
#include <ranges>
#include <algorithm>
#include <cstring>
#include <random>
#include <numeric> // Required for std::accumulate

namespace instance
{
  class CAIF_Framework;  // Forward declaration
/**
 * @brief Concept to ensure template parameter is a numeric type
 * @tparam T Type to check
 */
template<typename T>
concept NumericType=std::is_arithmetic_v<T>;

/**
 * @brief Multi-dimensional tensor class for AI computations
 * 
 * The CAIF_Tensor class provides a comprehensive tensor implementation
 * supporting various data types, mathematical operations, and memory
 * management optimized for AI/ML workloads.
 */
class CAIF_Tensor:public CAIF_Base
{
  public:
    /**
     * @brief Typedef for tensor shape representation
     */
    typedef std::vector<uint32_t> Shape_t;
    typedef std::vector<Shape_t> ShapeVec_t;

    /**
     * @brief Default constructor - creates an empty tensor
     * @param framework Reference to CAIF_Framework instance
     */
    explicit CAIF_Tensor(CAIF_Framework &framework);
    
    /**
     * @brief Parameterized constructor
     * @param framework Reference to CAIF_Framework instance
     * @param shape Vector defining the dimensions of the tensor
     * @param type Data type for tensor elements (default: Float32)
     * @throws std::invalid_argument if shape is invalid
     */
    CAIF_Tensor(CAIF_Framework &framework,
               const Shape_t &shape,
               const CAIF_DataType &type=
                 CAIF_DataType(CAIF_DataType::CAIF_DataType_e::Float32));

    // Convenience overload accepting nested enum
    CAIF_Tensor(
               CAIF_Framework &framework,
               const Shape_t &shape,
               const CAIF_DataType::CAIF_DataType_e type
              );
    
    /**
     * @brief Copy constructor
     * @param other Tensor to copy from
     */
    CAIF_Tensor(const CAIF_Tensor &other);
    
    /**
     * @brief Move constructor
     * @param other Tensor to move from
     */
    CAIF_Tensor(CAIF_Tensor &&other);
    
    /**
     * @brief Destructor
     */
    ~CAIF_Tensor()=default;

    /**
     * @brief Copy assignment operator
     * @param other Tensor to copy from
     * @return Reference to this tensor
     */
    CAIF_Tensor &operator=(const CAIF_Tensor &other);
    
    /**
     * @brief Move assignment operator
     * @param other Tensor to move from
     * @return Reference to this tensor
     */
    CAIF_Tensor &operator=(CAIF_Tensor &&other);

    // Getters - implemented inline for simple functions
    
    /**
     * @brief Get the shape of the tensor
     * @return Const reference to shape vector
     */
    const Shape_t &Shape()const
    {
      return _shape;
    }
    
    /**
     * @brief Get the data type of the tensor
     * @return CAIF_DataType_e enum value
     */
    CAIF_DataType Type()const
    {
      return _data_type;
    }
    
    /**
     * @brief Get the framework reference
     * @return Reference to CAIF_Framework instance
     */
    CAIF_Framework &Framework()const
    {
      return _framework;
    }
    
    /**
     * @brief Get the size of the tensor in bytes
     * @return Size in bytes
     */
    size_t Size()const
    {
      // Size of this view in bytes
      return NumElements()*_element_size;
    }
    
    /**
     * @brief Get the number of elements in the tensor
     * @return Number of elements
     */
    size_t NumElements()const
    {
      size_t elements=1;
      for(uint32_t dim:_shape)
      {
        elements*=dim;
      }
      return elements;
    }
    
    /**
     * @brief Get immutable pointer to tensor data
     * @return Const pointer to data or nullptr if invalid
     * @pre IsValid() must return true
     * @note This provides low-level access. Prefer typed GetAs() methods when possible.
     */
    const void *Data()const
    {
      if(_tensor_data!=nullptr)
      {
        const void *base=_tensor_data->RawData();
        if(base==nullptr)
        {
          return nullptr;
        }
        return static_cast<const uint8_t*>(base)+_byte_offset;
      }
      if(_buffer==nullptr)
      {
        return nullptr;
      }
      return static_cast<const void*>(_buffer->data()+_byte_offset);
    }
    
    /**
     * @brief Get mutable pointer to tensor data
     * @return Pointer to data or nullptr if invalid
     * @pre IsValid() must return true
     * @note This provides low-level access. Prefer typed GetAs() methods when possible.
     */
    void *Data()
    {
      if(_tensor_data!=nullptr)
      {
        void *base=_tensor_data->MutableRawData();
        if(base==nullptr)
        {
          return nullptr;
        }
        return static_cast<uint8_t*>(base)+_byte_offset;
      }
      if(_buffer==nullptr)
      {
        return nullptr;
      }
      return static_cast<void*>(_buffer->data()+_byte_offset);
    }

    /**
     * @brief Access underlying backend tensor data (may be null if not backend-backed)
     */
    CAIF_TensorData *TensorData(){return _tensor_data.get();}
    const CAIF_TensorData *TensorData()const{return _tensor_data.get();}

    /**
     * @brief Ensure backend tensor storage exists and is populated from host if needed
     */
    void EnsureBackendData();

    /**
     * @brief Get typed mutable data pointer with bounds checking
     * @tparam T Numeric type to cast data to
     * @return Expected containing typed pointer or error message
     */
    template<typename T>
    requires NumericType<T>
    T *MutableData()
    {
      if(IsValid()==false)
      {
        THROW_CAIFE("Tensor is not valid");
      }
      if(sizeof(T)!=_element_size)
      {
        THROW_CAIFE("Type size mismatch");
      }
      if(_tensor_data!=nullptr)
      {
        return static_cast<T*>(_tensor_data->MutableRawData());
      }
      return static_cast<T*>(Data());
    }

    /**
     * @brief Get typed const data pointer with bounds checking
     * @tparam T Numeric type to cast data to
     * @return Expected containing typed pointer or error message
     */
    template<typename T>
    requires NumericType<T>
    const T *ConstData()const
    {
      if(IsValid()==false)
      {
        THROW_CAIFE("Tensor is not valid");
      }
      if(sizeof(T)!=_element_size)
      {
        THROW_CAIFE("Type size mismatch");
      }
      if(_tensor_data!=nullptr)
      {
        return static_cast<const T*>(_tensor_data->RawData());
      }
      return static_cast<const T*>(Data());
    }

    /**
     * @brief Check if tensor is in a valid state
     * @return True if valid, false otherwise
     */
    bool IsValid()const
    {
      // Backend-owned storage is valid when shape/type are set and capacity is sufficient
      if(_tensor_data!=nullptr)
      {
        if(_shape.empty()==true||_element_size==0)
        {
          return false;
        }
        const size_t bytes_needed=NumElements()*_element_size;
        return _tensor_data->SizeBytes()>=bytes_needed;
      }
      const bool has_storage=_buffer!=nullptr;
      const bool has_shape=_shape.empty()==false;
      const bool has_elem_size=_element_size>0;
      if(has_storage==false||has_shape==false||has_elem_size==false)
      {
        return false;
      }
      size_t bytes_available=0;
      if(_buffer->size()>_byte_offset)
      {
        bytes_available=_buffer->size()-_byte_offset;
      }
      const size_t bytes_needed=NumElements()*_element_size;
      return bytes_available>=bytes_needed;
    }

    // Setters - implemented inline for simple functions
    
    /**
     * @brief Set the shape (dimensions) of the tensor
     * @param shape Vector defining new dimensions
     * @throws std::invalid_argument if shape is invalid
     */
    void SetShape(const Shape_t &shape)
    {
      ValidateShape(shape);
      _shape=shape;
    }
    
    /**
     * @brief Set the data type of tensor elements
     * @param type New data type
     */
    void SetDataType(const CAIF_DataType &type)
    {
      _data_type=type;
      _element_size=ElementSize();
    }
    
    /**
     * @brief Set tensor data from external source
     * @param data Pointer to source data
     * @param size Size of data in bytes
     */
    void SetData(const void *data,size_t size);
    
    /**
     * @brief Copy data from external source into tensor
     * @param data Pointer to source data
     * @param size Size of data in bytes
     * @return True if copy was successful, false otherwise
     */
    bool CopyFrom(const void *data,size_t size)
    {
      if(data==nullptr||size==0)
      {
        return false;
      }
      // Backend-owned storage path
      if(_tensor_data!=nullptr)
      {
        if(IsValid()==false)
        {
          return false;
        }
        const size_t max_size=NumElements()*_element_size;
        const size_t copy_size=std::min(size,max_size);
        void *dst=_tensor_data->MutableRawData();
        if(dst==nullptr)
        {
          return false;
        }
        std::memcpy(dst,data,copy_size);
        return true;
      }
      if(IsValid()==false)
      {
        return false;
      }
      const size_t max_size=NumElements()*_element_size;
      const size_t copy_size=std::min(size,max_size);
      if(_buffer==nullptr)
      {
        _buffer=std::make_shared<std::vector<uint8_t>>();
        _buffer->resize(max_size);
        _byte_offset=0;
      }
      std::memcpy(_buffer->data()+_byte_offset,data,copy_size);
      return true;
    }

    // Individual element access methods
    
    /**
     * @brief Get value at specific tensor coordinates
     * @param coordinates Vector of indices for each dimension
     * @return Expected containing element value as float or error message
     */
    float Value(const Shape_t &coordinates)const;
    
    /**
     * @brief Set value at specific tensor coordinates
     * @param coordinates Vector of indices for each dimension
     * @param value New value for the element
     * @return Empty expected on success, error string on failure
     */
    void SetValue(const Shape_t &coordinates,float value);

    // Tensor operations
    
    /**
     * @brief Reshape tensor to new dimensions
     * @param new_shape Vector defining new shape
     * @return New tensor with reshaped data
     * @throws std::invalid_argument if reshape is incompatible
     */
    CAIF_Tensor Reshape(const Shape_t &new_shape)const;
    
    /**
     * @brief Transpose tensor according to permutation
     * @param permutation Vector defining axis permutation
     * @return Transposed tensor
     * @throws std::invalid_argument if permutation is invalid
     */
    CAIF_Tensor Transpose(const Shape_t &permutation)const;
    
    /**
     * @brief Extract a slice from the tensor
     * @param ranges Vector of (start, end) pairs for each dimension
     * @return Sliced tensor
     * @throws std::out_of_range if ranges are invalid
     */
    CAIF_Tensor Slice(const std::vector<std::pair<uint32_t,uint32_t>> &ranges)const;

    /**
     * @brief Create a zero-copy view sliced only along the batch (dim 0)
     * @param batch_range (start, end) within the batch dimension
     * @return Tensor view sharing storage with adjusted shape and offset
     * @throws std::out_of_range if range is invalid
     */
    CAIF_Tensor SliceViewBatch(const std::pair<uint32_t,uint32_t> &batch_range)const;
    
    // Mathematical operations
    
    /**
     * @brief Element-wise addition with another tensor
     * @param other Tensor to add
     * @return Result tensor
     * @throws std::invalid_argument if shapes are incompatible
     */
    CAIF_Tensor Add(const CAIF_Tensor &other)const;
    
    /**
     * @brief Element-wise subtraction with another tensor
     * @param other Tensor to subtract
     * @return Result tensor
     * @throws std::invalid_argument if shapes are incompatible
     */
    CAIF_Tensor Subtract(const CAIF_Tensor &other)const;
    
    /**
     * @brief Element-wise multiplication with another tensor
     * @param other Tensor to multiply
     * @return Result tensor
     * @throws std::invalid_argument if shapes are incompatible
     */
    CAIF_Tensor Multiply(const CAIF_Tensor &other)const;
    
    /**
     * @brief Element-wise division with another tensor
     * @param other Tensor to divide by
     * @return Result tensor
     * @throws std::invalid_argument if shapes are incompatible
     * @throws std::runtime_error if division by zero
     */
    CAIF_Tensor Divide(const CAIF_Tensor &other)const;
    
    /**
     * @brief Matrix multiplication with another tensor
     * @param other Tensor to multiply with
     * @return Result tensor
     * @throws std::invalid_argument if dimensions are incompatible for matrix multiplication
     */
    CAIF_Tensor MatMul(const CAIF_Tensor &other)const;
    
    // Scalar operations
    
    /**
     * @brief Element-wise addition with scalar value
     * @param scalar Scalar value to add
     * @return Result tensor
     */
    CAIF_Tensor Add(float scalar)const;
    
    /**
     * @brief Element-wise subtraction with scalar value
     * @param scalar Scalar value to subtract
     * @return Result tensor
     */
    CAIF_Tensor Subtract(float scalar)const;
    
    /**
     * @brief Element-wise multiplication with scalar value
     * @param scalar Scalar value to multiply
     * @return Result tensor
     */
    CAIF_Tensor Multiply(float scalar)const;
    
    /**
     * @brief Element-wise division by scalar value
     * @param scalar Scalar value to divide by
     * @return Result tensor
     * @throws std::runtime_error if scalar is zero
     */
    CAIF_Tensor Divide(float scalar)const;
    
    // Element-wise operations
    
    /**
     * @brief Compute absolute value of all elements
     * @return Tensor with absolute values
     */
    CAIF_Tensor Abs()const;
    
    /**
     * @brief Compute exponential of all elements
     * @return Tensor with exponential values
     */
    CAIF_Tensor Exp()const;
    
    /**
     * @brief Compute natural logarithm of all elements
     * @return Tensor with logarithm values
     * @throws std::runtime_error if any element is <= 0
     */
    CAIF_Tensor NatLog()const;
    
    /**
     * @brief Compute square root of all elements
     * @return Tensor with square root values
     * @throws std::runtime_error if any element is < 0
     */
    CAIF_Tensor Sqrt()const;
    
    /**
     * @brief Raise all elements to a power
     * @param exponent Power to raise elements to
     * @return Tensor with powered values
     */
    CAIF_Tensor Pow(float exponent)const;

    // Activation functions
    
    /**
     * @brief Apply Linear activation function (identity)
     * @return Tensor with Linear applied (unchanged)
     */
    CAIF_Tensor Linear()const;
    
    /**
     * @brief Apply ReLU activation function
     * @return Tensor with ReLU applied
     */
    CAIF_Tensor ReLU()const;
    
    /**
     * @brief Apply Sigmoid activation function
     * @return Tensor with Sigmoid applied
     */
    CAIF_Tensor Sigmoid()const;
    
    /**
     * @brief Apply Tanh activation function
     * @return Tensor with Tanh applied
     */
    CAIF_Tensor Tanh()const;
    
    /**
     * @brief Apply Softmax activation function
     * @return Tensor with Softmax applied
     */
    CAIF_Tensor Softmax()const;
    
    /**
     * @brief Apply Leaky ReLU activation function
     * @param alpha Negative slope parameter
     * @return Tensor with Leaky ReLU applied
     */
    CAIF_Tensor LeakyReLU(float alpha=0.01f)const;
    
    /**
     * @brief Apply ELU activation function
     * @param alpha Alpha parameter
     * @return Tensor with ELU applied
     */
    CAIF_Tensor ELU(float alpha=1.0f)const;
    
    /**
     * @brief Apply GELU activation function
     * @return Tensor with GELU applied
     */
    CAIF_Tensor GELU()const;
    
    /**
     * @brief Apply Swish activation function
     * @return Tensor with Swish applied
     */
    CAIF_Tensor Swish()const;

    // Activation derivatives
    
    /**
     * @brief Apply Linear derivative (identity - gradient passes through unchanged)
     * @param gradient Gradient from next layer
     * @return Gradient with Linear derivative applied (unchanged)
     */
    CAIF_Tensor LinearDerivative(const CAIF_Tensor &gradient)const;
    
    /**
     * @brief Apply ReLU derivative
     * @param gradient Gradient from next layer
     * @return Gradient with ReLU derivative applied
     */
    CAIF_Tensor ReLUDerivative(const CAIF_Tensor &gradient)const;
    
    /**
     * @brief Apply Sigmoid derivative
     * @param gradient Gradient from next layer
     * @return Gradient with Sigmoid derivative applied
     */
    CAIF_Tensor SigmoidDerivative(const CAIF_Tensor &gradient)const;
    
    /**
     * @brief Apply Tanh derivative
     * @param gradient Gradient from next layer
     * @return Gradient with Tanh derivative applied
     */
    CAIF_Tensor TanhDerivative(const CAIF_Tensor &gradient)const;
    
    /**
     * @brief Apply Softmax derivative
     * @param gradient Gradient from next layer
     * @return Gradient with Softmax derivative applied
     */
    CAIF_Tensor SoftmaxDerivative(const CAIF_Tensor &gradient)const;
    
    /**
     * @brief Apply Leaky ReLU derivative
     * @param gradient Gradient from next layer
     * @param alpha Negative slope parameter
     * @return Gradient with Leaky ReLU derivative applied
     */
    CAIF_Tensor LeakyReLUDerivative(const CAIF_Tensor &gradient,float alpha=0.01f)const;
    
    /**
     * @brief Apply ELU derivative
     * @param gradient Gradient from next layer
     * @param alpha Alpha parameter
     * @return Gradient with ELU derivative applied
     */
    CAIF_Tensor ELUDerivative(const CAIF_Tensor &gradient,float alpha=1.0f)const;
    
    /**
     * @brief Apply GELU derivative
     * @param gradient Gradient from next layer
     * @return Gradient with GELU derivative applied
     */
    CAIF_Tensor GELUDerivative(const CAIF_Tensor &gradient)const;
    
    /**
     * @brief Apply Swish derivative
     * @param gradient Gradient from next layer
     * @return Gradient with Swish derivative applied
     */
    CAIF_Tensor SwishDerivative(const CAIF_Tensor &gradient)const;

    // Reduction operations
    
    /**
     * @brief Sum elements along specified axis
     * @param axis Axis to sum along
     * @return Reduced tensor
     * @throws std::out_of_range if axis is invalid
     */
    CAIF_Tensor Sum(uint32_t axis)const;
    
    /**
     * @brief Compute mean along specified axis
     * @param axis Axis to compute mean along
     * @return Reduced tensor with mean values
     * @throws std::out_of_range if axis is invalid
     */
    CAIF_Tensor Mean(uint32_t axis)const;
    
    /**
     * @brief Find maximum values along specified axis
     * @param axis Axis to find maximum along
     * @return Reduced tensor with maximum values
     * @throws std::out_of_range if axis is invalid
     */
    CAIF_Tensor Max(uint32_t axis)const;
    
    /**
     * @brief Find minimum values along specified axis
     * @param axis Axis to find minimum along
     * @return Reduced tensor with minimum values
     * @throws std::out_of_range if axis is invalid
     */
    CAIF_Tensor Min(uint32_t axis)const;

    // Utility functions
    
    /**
     * @brief Convert tensor to string representation
     * @return String representation of tensor
     */
    std::string ToString()const;
    
    /**
     * @brief Save tensor to file
     * @param filename Path to output file
     * @throws std::runtime_error if file cannot be written
     */
    void SaveToFile(const std::string &filename)const;
    
    /**
     * @brief Load tensor from file
     * @param filename Path to input file
     * @return Expected containing tensor or error message
     */
    static CAIF_Tensor LoadFromFile(CAIF_Framework &framework,const std::string &filename);

    // C++23 features
    
    /**
     * @brief Get typed span view of tensor data
     * @tparam T Numeric type to cast data to
     * @return Span of typed data
     * @throws std::bad_cast if type conversion is invalid
     */
    template<typename T>
    requires NumericType<T>
    std::span<T> AsSpan()
    {
      return std::span<T>(static_cast<T*>(Data()),NumElements());
    }

    /**
     * @brief Get const typed span view of tensor data
     * @tparam T Numeric type to cast data to
     * @return Const span of typed data
     * @throws std::bad_cast if type conversion is invalid
     */
    template<typename T>
    requires NumericType<T>
    std::span<const T> AsSpan()const
    {
      return std::span<const T>(static_cast<const T*>(Data()),NumElements());
    }

    /**
     * @brief Get ranges view of tensor shape
     * @return Ranges view of shape vector
     */
    auto ShapeView()const
    {
      return std::ranges::views::all(_shape);
    }

    // New method declaration
    /**
     * @brief Fill tensor with random values between 0 and 1
     * @param generator Random number generator to use
     * @return True if successful, false otherwise
     */
    bool FillWithRandom(std::mt19937 &generator);
    
    /**
     * @brief Fill tensor with random values between 0 and 1 using default generator
     * @return True if successful, false otherwise
     */
    bool FillWithRandom();

    /**
     * @brief Set data for a dynamic batch tensor
     * @param data Pointer to the data
     * @param batch_size New batch size
     * @return Expected void if successful, error message if failed
     */
    void SetBatchData(const void *data,const uint32_t batch_size)
    {
      if(IsDynamicBatch()==false){THROW_CAIFE("Tensor does not have dynamic batch size");}
      if(data==nullptr){THROW_CAIFE("Data pointer is null");}
      if(batch_size==0){THROW_CAIFE("Batch size cannot be zero");}

      // Calculate new shape and size
      Shape_t new_shape=_shape;
      new_shape[0]=batch_size;
      
      const size_t elements_per_batch=
        std::accumulate(_shape.begin()+1,
                        _shape.end(),
                        1UL,
                        std::multiplies<size_t>());
      const size_t total_elements=batch_size*elements_per_batch;
      const size_t total_size=total_elements*_element_size;

      // Resize data buffer
      if(_buffer==nullptr)
      {
        _buffer=std::make_shared<std::vector<uint8_t>>();
      }
      _buffer->resize(total_size);
      _shape=new_shape;

      // Copy data
      _byte_offset=0;
      std::memcpy(_buffer->data(),data,total_size);
    }

    /**
     * @brief Check if tensor has dynamic batch size
     * @return True if first dimension is dynamic (0)
     */
    bool IsDynamicBatch()const
    {
      return (!_shape.empty())&&(_shape[0]==0);
    }

    /**
     * @brief Perform 2D convolution operation
     * @param kernel Convolution kernel tensor
     * @param stride_h Vertical stride
     * @param stride_w Horizontal stride
     * @return Result of convolution operation
     * @throws std::invalid_argument if dimensions are incompatible or kernel is invalid
     */
    CAIF_Tensor Convolution2D(const CAIF_Tensor &kernel,uint32_t stride_h,uint32_t stride_w)const;

    // Public accessors using batch mapping when present
    const void *SampleData(const uint32_t logical_batch_index)const
    {
      if(_buffer==nullptr)
      {
        return nullptr;
      }
      return static_cast<const void*>(_buffer->data()+SampleByteOffset(logical_batch_index));
    }

    void *MutableSampleData(const uint32_t logical_batch_index)
    {
      if(_buffer==nullptr)
      {
        return nullptr;
      }
      return static_cast<void*>(_buffer->data()+SampleByteOffset(logical_batch_index));
    }

    /**
     * @brief Return a view that applies a batch order mapping (no copy)
     */
    CAIF_Tensor WithBatchOrder(const std::shared_ptr<std::vector<uint32_t>> &order)const
    {
      if(order==nullptr)
      {
        return *this;
      }
      if(_shape.empty()||order->empty())
      {
        return *this;
      }
      if(order->size()!=_shape[0])
      {
        throw std::invalid_argument("Batch order size must match batch dimension");
      }
      CAIF_Tensor view=*this;
      view._batch_index_map=order;
      return view;
    }

    // Template function declarations
    template<typename T,typename Op>
    CAIF_Tensor ElementWiseOpWithBroadcast(const CAIF_Tensor &other,Op op)const;

    template<typename T>
    CAIF_Tensor MatMulWithBroadcast(const CAIF_Tensor &other)const;

    template<typename Op>
    CAIF_Tensor ExecuteTypedOperation(const CAIF_Tensor &other,Op op)const;

    CAIF_Tensor ExecuteTypedMatMul(const CAIF_Tensor &other)const;

    template<typename Op>
    CAIF_Tensor ElementWiseOp(const CAIF_Tensor &other,Op op)const;

    template<typename Op>
    CAIF_Tensor ElementWiseScalarOp(float scalar,Op op)const;

    template<typename Op,typename InitFunc>
    CAIF_Tensor ReduceAlongAxis(uint32_t axis,Op op,InitFunc init_func)const;

    template<typename Op>
    CAIF_Tensor ApplyActivation(Op op)const;

    /**
     * @brief Calculate strides for a given shape
     * @param shape Shape to calculate strides for
     * @return Vector of strides for each dimension
     */
    static std::vector<size_t> CalculateStrides(const Shape_t &shape);

    /**
     * @brief Convert linear index to multi-dimensional indices
     * @param linear_idx Linear index to convert
     * @param shape Shape of the tensor
     * @param strides Strides for each dimension
     * @return Vector of indices for each dimension
     */
    static std::vector<size_t> LinearToMultiIndex(size_t linear_idx,
                                                  const Shape_t &shape,
                                                  const std::vector<size_t> &strides);

    /**
     * @brief Convert multi-dimensional indices to linear index
     * @param indices Multi-dimensional indices
     * @param strides Strides for each dimension
     * @return Linear index
     */
    static size_t MultiToLinearIndex(const std::vector<size_t> &indices, const std::vector<size_t> &strides);

    /**
     * @brief Validate and extract batch dimension information
     * @param shape Tensor shape to analyze
     * @return Expected containing batch size or error message
     */
    static uint32_t ValidateBatchDimension(const Shape_t &shape);

    /**
     * @brief Calculate total elements per sample (excluding batch dimension)
     * @param shape Tensor shape to analyze
     * @return Expected containing elements per sample or error message
     */
    static size_t ElementsPerSample(const Shape_t &shape);

    /**
     * @brief Validate batch compatibility between tensors
     * @param other Other tensor to validate against
     * @param operation Name of operation for error messages
     * @return Expected containing batch size or error message
     */
    uint32_t ValidateBatchOperation(const CAIF_Tensor &other,const std::string &operation)const;

    /**
     * @brief Allocate memory for tensor data
     * @throws std::bad_alloc if allocation fails
     */
    void AllocateMemory();
    
    /**
     * @brief Calculate stride information for tensor
     * @return Stride value
     */
    size_t CalculateStrides()const;
    
    /**
     * @brief Get size of single element based on data type
     * @return Element size in bytes
     */
    size_t ElementSize()const
    {
      return _data_type.ElementSizeBytes();
    }
    
    /**
     * @brief Validate tensor shape for correctness
     * @param shape Shape vector to validate
     * @throws std::invalid_argument if shape is invalid
     */
    void ValidateShape(const Shape_t &shape)const
    {
      if(shape.empty()||shape.size()>g_caif_max_tensor_dimensions)
      {
        throw std::invalid_argument("Invalid tensor shape");
      }
      for(uint32_t dim:shape)
      {
        if(dim==0)
        {
          throw std::invalid_argument("Tensor dimensions must be greater than 0");
        }
      }
    }
    
    /**
     * @brief Validate operation compatibility between tensors
     * @param other Other tensor to validate against
     * @param operation Name of operation for error messages
     * @throws std::invalid_argument if tensors are incompatible
     */
    void ValidateOperation(const CAIF_Tensor &other,const std::string &operation)const;

    // C++23 features
    
    /**
     * @brief Check if current shape is valid (compile-time optimized)
     * @return True if shape is valid, false otherwise
     */
    bool IsValidShape()const
    {
      if consteval
      {
        return (_shape.empty()==false)&&
                (_shape.size()<=g_caif_max_tensor_dimensions);
      }
      return (_shape.empty()==false)&&
              (_shape.size()<=g_caif_max_tensor_dimensions)&&
              std::ranges::all_of(_shape.begin(),_shape.end(),[](uint32_t dim){return (dim>0);});
    }

    /**
     * @brief Get the linear index for given coordinates
     * @param coordinates Vector of indices
     * @return Linear index
     * @throws std::out_of_range if coordinates are invalid
     */
    size_t Index(const Shape_t &coordinates)const;
  protected:
    // Protected members go here

  private:
    CAIF_Framework &_framework;                ///< Reference to framework instance
    Shape_t _shape;                          ///< Tensor dimensions
    CAIF_DataType _data_type;                 ///< Element data type
    // Backend-owned storage: if set, use backend data (CPU or CUDA). Fallback to host buffer.
    std::shared_ptr<CAIF_TensorData> _tensor_data;
    std::shared_ptr<std::vector<uint8_t>> _buffer; ///< Shared raw storage
    size_t _byte_offset=0;                   ///< Byte offset into shared storage for this view
    size_t _element_size;                    ///< Size of each element in bytes

    // Optional logical-to-physical mapping for batch samples (dim 0)
    ///< If set, maps logical batch index -> physical sample index
    std::shared_ptr<std::vector<uint32_t>> _batch_index_map;

    // Batch mapping helpers
    bool HasBatchIndexMap()const
    {
      return (_batch_index_map!=nullptr)&&(!_batch_index_map->empty());
    }

    uint32_t PhysicalBatchIndex(const uint32_t logical_batch_index)const
    {
      if(HasBatchIndexMap()==false)
      {
        return logical_batch_index;
      }
      return (*_batch_index_map)[logical_batch_index];
    }

    size_t BytesPerSample()const
    {
      if(_shape.empty())
      {
        return 0;
      }
      const size_t elems_per_sample=std::accumulate(_shape.begin()+1,_shape.end(),1UL,std::multiplies<size_t>());
      return elems_per_sample*_element_size;
    }

    size_t SampleByteOffset(const uint32_t logical_batch_index)const
    {
      const uint32_t physical_index=PhysicalBatchIndex(logical_batch_index);
      return _byte_offset+static_cast<size_t>(physical_index)*BytesPerSample();
    }

};

// Template function implementations
template<typename T,typename Op>
CAIF_Tensor CAIF_Tensor::ElementWiseOpWithBroadcast(const CAIF_Tensor &other,Op op)const
{
  const uint32_t batch_size=std::max(_shape[0],other._shape[0]);
  if(batch_size==0)
  {
    throw std::invalid_argument("At least one tensor must have non-zero batch size");
  }

  // Create new shapes with matched batch size
  Shape_t this_shape=_shape;
  Shape_t other_shape=other._shape;
  this_shape[0]=batch_size;
  other_shape[0]=batch_size;

  // Validate remaining dimensions match
  for(size_t i=1;i<_shape.size();++i)
  {
    if(this_shape[i]!=other_shape[i])
    {
      throw std::invalid_argument("Non-batch dimensions must match");
    }
  }

  // Create result tensor with matched batch size
  CAIF_Tensor result(_framework,this_shape,_data_type);

  // Perform operation with broadcasting
  const size_t elements_per_batch=std::accumulate(_shape.begin()+1,_shape.end(),1UL,std::multiplies<size_t>());
  
  const T *a=static_cast<const T*>(Data());
  const T *b=static_cast<const T*>(other.Data());
  T *c=static_cast<T*>(result.Data());

  for(uint32_t i=0;i<batch_size;++i)
  {
    const uint32_t a_batch=std::min(i,_shape[0]-1);
    const uint32_t b_batch=std::min(i,other._shape[0]-1);

    for(size_t j=0;j<elements_per_batch;++j)
    {
      c[i*elements_per_batch+j]=op(a[a_batch*elements_per_batch+j],
                                 b[b_batch*elements_per_batch+j]);
    }
  }

  return result;
}

template<typename T>
CAIF_Tensor CAIF_Tensor::MatMulWithBroadcast(const CAIF_Tensor &other)const
{
  const uint32_t batch_size=std::max(_shape[0],other._shape[0]);
  if(batch_size==0)
  {
    throw std::invalid_argument("At least one tensor must have non-zero batch size");
  }

  // Validate matrix dimensions
  if(_shape.size()!=3||other._shape.size()!=3)
  {
    throw std::invalid_argument("Matrix multiplication requires 3D tensors (batch, rows, cols)");
  }

  const uint32_t M=_shape[1];
  const uint32_t K=_shape[2];
  const uint32_t N=other._shape[2];

  if(K!=other._shape[1])
  {
    throw std::invalid_argument("Inner matrix dimensions must match");
  }

  // Create result tensor
  Shape_t result_shape={batch_size,M,N};
  CAIF_Tensor result(_framework,result_shape,_data_type);

  // Perform batched matrix multiplication with broadcasting
  const size_t a_elements_per_batch=M*K;
  const size_t b_elements_per_batch=K*N;
  const size_t c_elements_per_batch=M*N;

  const T *a=static_cast<const T*>(Data());
  const T *b=static_cast<const T*>(other.Data());
  T *c=static_cast<T*>(result.Data());

  for(uint32_t i=0;i<batch_size;++i)
  {
    const uint32_t a_batch=std::min(i,_shape[0]-1);
    const uint32_t b_batch=std::min(i,other._shape[0]-1);

    // Perform matrix multiplication for this batch
    for(uint32_t m=0;m<M;++m)
    {
      for(uint32_t n=0;n<N;++n)
      {
        T sum=0;
        for(uint32_t k=0;k<K;++k)
        {
          sum+=a[a_batch*a_elements_per_batch+m*K+k]*
               b[b_batch*b_elements_per_batch+k*N+n];
        }
        c[i*c_elements_per_batch+m*N+n]=sum;
      }
    }
  }

  return result;
}

template<typename Op>
CAIF_Tensor CAIF_Tensor::ExecuteTypedOperation(const CAIF_Tensor &other,Op op)const
{
  switch(_data_type.Value())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
      return ElementWiseOpWithBroadcast<float>(other,op);
    case CAIF_DataType::CAIF_DataType_e::Float64:
      return ElementWiseOpWithBroadcast<double>(other,op);
    case CAIF_DataType::CAIF_DataType_e::Int32:
      return ElementWiseOpWithBroadcast<int32_t>(other,op);
    case CAIF_DataType::CAIF_DataType_e::UInt32:
      return ElementWiseOpWithBroadcast<uint32_t>(other,op);
    case CAIF_DataType::CAIF_DataType_e::Int8:
      return ElementWiseOpWithBroadcast<int8_t>(other,op);
    case CAIF_DataType::CAIF_DataType_e::UInt8:
      return ElementWiseOpWithBroadcast<uint8_t>(other,op);
    default:
      throw std::runtime_error("Unsupported data type");
  }
  // Fallback to satisfy compiler's control flow analysis in templated context
  throw std::runtime_error("Unsupported data type");
}

template<typename Op>
CAIF_Tensor CAIF_Tensor::ElementWiseOp(const CAIF_Tensor &other,Op op)const
{
  // Validate shapes and get batch size
  const uint32_t batch_result=ValidateBatchOperation(other,"element-wise operation");

  if(_shape!=other._shape)
  {
    throw std::invalid_argument("Tensor shapes must match exactly for element-wise operations");
  }

  if(_data_type!=other._data_type)
  {
    throw std::invalid_argument("Tensor data types must match for element-wise operations");
  }

  CAIF_Tensor result(_framework,_shape,_data_type);
  const size_t num_elements=NumElements();

  switch(_data_type.Value())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
    {
      const float *a=static_cast<const float*>(Data());
      const float *b=static_cast<const float*>(other.Data());
      float *c=static_cast<float*>(result.Data());
      for(size_t i=0;i<num_elements;++i)
      {
        c[i]=op(a[i],b[i]);
      }
      break;
    }
    case CAIF_DataType::CAIF_DataType_e::Float64:
    {
      const double *a=static_cast<const double*>(Data());
      const double *b=static_cast<const double*>(other.Data());
      double *c=static_cast<double*>(result.Data());
      for(size_t i=0;i<num_elements;++i)
      {
        c[i]=op(a[i],b[i]);
      }
      break;
    }
    case CAIF_DataType::CAIF_DataType_e::Int32:
    {
      const int32_t *a=static_cast<const int32_t*>(Data());
      const int32_t *b=static_cast<const int32_t*>(other.Data());
      int32_t *c=static_cast<int32_t*>(result.Data());
      for(size_t i=0;i<num_elements;++i)
      {
        c[i]=op(a[i],b[i]);
      }
      break;
    }
    default:
      throw std::runtime_error("Element-wise operation not implemented for this data type");
  }
  return result;
}

template<typename Op>
CAIF_Tensor CAIF_Tensor::ElementWiseScalarOp(float scalar,Op op)const
{
  CAIF_Tensor result(_framework,_shape,_data_type);
  const size_t num_elements=NumElements();

  switch(_data_type.Value())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
    {
      const float *src=static_cast<const float*>(Data());
      float *dst=static_cast<float*>(result.Data());
      for(size_t i=0;i<num_elements;++i)
      {
        dst[i]=op(src[i],scalar);
      }
      break;
    }
    default:
      throw std::runtime_error("Scalar operation not implemented for this data type");
  }
  return result;
}

template<typename Op,typename InitFunc>
CAIF_Tensor CAIF_Tensor::ReduceAlongAxis(uint32_t axis,Op op,InitFunc init_func)const
{
  if(axis>=_shape.size())
  {
    throw std::out_of_range("Axis out of range");
  }

  // Create new shape with the specified axis removed
  std::vector<uint32_t> new_shape;
  for(size_t i=0;i<_shape.size();++i)
  {
    if(i!=axis)
    {
      new_shape.push_back(_shape[i]);
    }
  }

  // Handle empty result shape
  if(new_shape.empty()==true)
  {
    new_shape.push_back(1);  // Scalar result
  }

  CAIF_Tensor result(_framework,new_shape,_data_type);

  // Calculate strides for both tensors
  const auto src_strides=CalculateStrides(_shape);
  const auto dst_strides=CalculateStrides(new_shape);

  switch(_data_type.Value())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
    {
      const float *src=static_cast<const float*>(Data());
      float *dst=static_cast<float*>(result.Data());
      const size_t result_elements=result.NumElements();

      // Initialize result
      for(size_t i=0;i<result_elements;++i)
      {
        dst[i]=init_func();
      }

      // Iterate through all elements in the source tensor
      const size_t num_elements=NumElements();
      for(size_t src_idx=0;src_idx<num_elements;++src_idx)
      {
        // Get source indices
        const auto src_indices=LinearToMultiIndex(src_idx,_shape,src_strides);

        // Calculate destination index (skip the reduction axis)
        size_t dst_idx=0;
        size_t dst_dim=0;
        for(size_t src_dim=0;src_dim<_shape.size();++src_dim)
        {
          if(src_dim!=axis)
          {
            dst_idx+=src_indices[src_dim]*dst_strides[dst_dim];
            ++dst_dim;
          }
        }

        // Apply reduction operation
        dst[dst_idx]=op(dst[dst_idx],src[src_idx]);
      }
      break;
    }
    default:
      throw std::runtime_error("Reduction operation not implemented for this data type");
  }
  return result;
}

template<typename Op>
CAIF_Tensor CAIF_Tensor::ApplyActivation(Op op)const
{
  CAIF_Tensor result(_framework,_shape,_data_type);
  const size_t num_elements=NumElements();

  switch(_data_type.Value())
  {
    case CAIF_DataType::CAIF_DataType_e::Float32:
    {
      const float *src=static_cast<const float*>(Data());
      float *dst=static_cast<float*>(result.Data());
      for(size_t i=0;i<num_elements;++i)
      {
        dst[i]=op(src[i]);
      }
      break;
    }
    default:
      throw std::runtime_error("Activation function not implemented for this data type");
  }
  return result;
}

}//end instance namespace
