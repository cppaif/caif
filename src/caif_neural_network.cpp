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
 * @file aif_neural_network.cpp
 * @brief Implementation of the CAIF_NeuralNetwork class
 * @author AIF Development Team
 * @version 1.0
 * @date 2024
 */

#include "caif_neural_network.h"
#include "ise_lib/ise_out.h"

#include "caif_layer.h"
#include "caif_optimizer.h"
#include "caif_loss_function.h"
#include "caif_loss_function_bce_logits.h"
#include "caif_dense_layer.h"
#include "caif_dropout_layer.h"
#include "caif_flatten_layer.h"
#include "caif_max_pooling2d_layer.h"
#include "caif_average_pooling2d_layer.h"
#include "caif_convolution2d_layer.h"
#include "caif_batch_normalization_layer.h"
#include "caif_mean_squared_error_loss.h"
#include "caif_cross_entropy_loss.h"
#include "caif_categorical_cross_entropy_loss.h"
#include "caif_sgd_optimizer.h"
#include "caif_adam_optimizer.h"
#include "caif_settings.h"
#include "caif_exception.h"
#include "ise_lib/ise_out.h"
#include <algorithm>
#include <random>
#include <sstream>
#include <cmath>
#include <limits>
#include "caif_reshape_layer.h"

namespace instance
{
CAIF_NeuralNetwork::CAIF_NeuralNetwork():_framework(),
                                       _optimizer_type(CAIF_OptimizerType_e::Adam),
                                       _loss_type(CAIF_LossType_e::MeanSquaredError),
                                       _learning_rate(g_caif_default_learning_rate),
                                       _compiled(false),
                                       _trained(false),
                                       _training_epochs(0)
{
}

CAIF_NeuralNetwork::CAIF_NeuralNetwork(CAIF_Framework &framework):_framework(framework),
                                       _optimizer_type(CAIF_OptimizerType_e::Adam),
                                       _loss_type(CAIF_LossType_e::MeanSquaredError),
                                       _learning_rate(g_caif_default_learning_rate),
                                       _compiled(false),
                                       _trained(false),
                                       _training_epochs(0)
{
}

// Copy constructor - framework reference copied from source
CAIF_NeuralNetwork::CAIF_NeuralNetwork(const CAIF_NeuralNetwork &other):_framework(other._framework),
                                                                     _input_shape(other._input_shape),
                                                                     _output_shape(other._output_shape),
                                                                     _gradients(other._gradients),
                                                                     _optimizer_type(other._optimizer_type),
                                                                     _loss_type(other._loss_type),
                                                                     _learning_rate(other._learning_rate),
                                                                     _compiled(other._compiled),
                                                                     _trained(other._trained),
                                                                     _training_epochs(other._training_epochs)
{
  // Deep copy layers (framework reference is copied from source via copy constructor)
  _layers.reserve(other._layers.size());
  for(const auto &layer:other._layers)
  {
    _layers.push_back(layer->Clone());
  }
  
  // Deep copy optimizer and loss function if they exist (framework reference copied from source)
  if(other._optimizer!=nullptr)
  {
    _optimizer=other._optimizer->Clone();
  }
  if(other._loss_function!=nullptr)
  {
    _loss_function=other._loss_function->Clone();
  }
}

CAIF_NeuralNetwork::CAIF_NeuralNetwork(CAIF_NeuralNetwork &&other):_framework(other._framework),
                                                                _layers(std::move(other._layers)),
                                                                _optimizer(std::move(other._optimizer)),
                                                                _loss_function(std::move(other._loss_function)),
                                                                _input_shape(std::move(other._input_shape)),
                                                                _output_shape(std::move(other._output_shape)),
                                                                _gradients(std::move(other._gradients)),
                                                                _optimizer_type(other._optimizer_type),
                                                                _loss_type(other._loss_type),
                                                                _learning_rate(other._learning_rate),
                                                                _compiled(other._compiled),
                                                                _trained(other._trained),
                                                                _training_epochs(other._training_epochs)
{
  // Reset moved-from object
  other._compiled=false;
  other._trained=false;
  other._training_epochs=0;
}

CAIF_NeuralNetwork &CAIF_NeuralNetwork::operator=(const CAIF_NeuralNetwork &other)
{
  if(this==&other)
  {
    return *this;
  }

  // Copy basic members
  _input_shape=other._input_shape;
  _output_shape=other._output_shape;
  _gradients=other._gradients;
  _optimizer_type=other._optimizer_type;
  _loss_type=other._loss_type;
  _learning_rate=other._learning_rate;
  _compiled=other._compiled;
  _trained=other._trained;
  _training_epochs=other._training_epochs;
  
  // Framework reference cannot be reassigned (it's a reference), but it's already set
  // Note: Framework reference is copied from source (same framework instance)
  
  // Deep copy layers (framework reference copied from source via Clone copy constructor)
  _layers.clear();
  _layers.reserve(other._layers.size());
  for(const auto &layer:other._layers)
  {
    _layers.push_back(layer->Clone());
  }
  
  // Deep copy optimizer and loss function (framework reference copied from source)
  if(other._optimizer!=nullptr)
  {
    _optimizer=other._optimizer->Clone();
  }
  else
  {
    _optimizer.reset();
  }
  
  if(other._loss_function!=nullptr)
  {
    _loss_function=other._loss_function->Clone();
  }
  else
  {
    _loss_function.reset();
  }

  return *this;
}

CAIF_NeuralNetwork &CAIF_NeuralNetwork::operator=(CAIF_NeuralNetwork &&other)
{
  if(this!=&other)
  {
    _layers=std::move(other._layers);
    _optimizer=std::move(other._optimizer);
    _loss_function=std::move(other._loss_function);
    _input_shape=std::move(other._input_shape);
    _output_shape=std::move(other._output_shape);
    _gradients=std::move(other._gradients);
    _optimizer_type=other._optimizer_type;
    _loss_type=other._loss_type;
    _learning_rate=other._learning_rate;
    _compiled=other._compiled;
    _trained=other._trained;
    _training_epochs=other._training_epochs;
    
    // Reset moved-from object
    other._compiled=false;
    other._trained=false;
    other._training_epochs=0;
  }
  return *this;
}

void CAIF_NeuralNetwork::AddDenseLayer(const uint32_t units,
                                                               const CAIF_ActivationType_e activation,
                                                               const bool use_bias
                                                              )
{
  if(_compiled==true)
  {
    THROW_CAIFE("Cannot add layers to compiled network");
  }
  
  if(units==0 || units>g_caif_max_layer_size)
  {
    THROW_CAIFE("Invalid layer size");
  }
  
  // Create actual dense layer implementation
  auto dense_layer=std::make_unique<CAIF_DenseLayer>(_framework,units,activation,use_bias);
  
  // Initialize the layer if we have input shape
  if(_input_shape.empty()==false && _layers.empty()==true)
  {
    dense_layer->Initialize(_input_shape,++_init_seed);
  }
  _layers.push_back(std::move(dense_layer));
  return;
}

void CAIF_NeuralNetwork::AddConvolution2DLayer(const uint32_t filters,
                                              const uint32_t kernel_size,
                                              const uint32_t stride,
                                              const uint32_t padding,
                                              const CAIF_ActivationType_e activation
                                             )
{
  if(_compiled==true)
  {
    THROW_CAIFE("Cannot add layers to compiled network");
  }
  
  if(filters==0 || kernel_size==0 || stride==0)
  {
    THROW_CAIFE("Invalid convolution parameters");
  }
  
  // Create actual convolution layer implementation
  auto conv_layer=std::make_unique<CAIF_Convolution2DLayer>(
    _framework,
    filters,
    kernel_size,
    stride,
    padding,
    activation);
  
  _layers.push_back(std::move(conv_layer));
  return;
}

void CAIF_NeuralNetwork::AddMaxPooling2DLayer(const uint32_t pool_size,
                                             const uint32_t stride
                                            )
{
  if(_compiled==true)
  {
    THROW_CAIFE("Cannot add layers to compiled network");
  }
  
  if(pool_size==0 || stride==0)
  {
    THROW_CAIFE("Invalid pooling parameters");
  }
  
  // Create actual pooling layer implementation
  auto pooling_layer=std::make_unique<CAIF_MaxPooling2DLayer>(_framework,pool_size,stride);
  
  _layers.push_back(std::move(pooling_layer));
  return;
}

void CAIF_NeuralNetwork::AddAveragePooling2DLayer(const uint32_t pool_size,
                                                 const uint32_t stride
                                                )
{
  if(_compiled==true)
  {
    THROW_CAIFE("Cannot add layers to compiled network");
  }
  
  if(pool_size==0 || stride==0)
  {
    THROW_CAIFE("Invalid pooling parameters");
  }
  
  auto pooling_layer=std::make_unique<CAIF_AveragePooling2DLayer>(_framework,pool_size,stride);
  _layers.push_back(std::move(pooling_layer));
  return;
}

void CAIF_NeuralNetwork::AddDropoutLayer(const float rate)
{
  if(_compiled==true)
  {
    THROW_CAIFE("Cannot add layers to compiled network");
  }
  
  if(rate<0.0f || rate>=1.0f)
  {
    THROW_CAIFE("Dropout rate must be between 0.0 and 1.0");
  }
  
  // Create actual dropout layer implementation
  auto dropout_layer=std::make_unique<CAIF_DropoutLayer>(_framework,rate);
  
  // Initialize the layer if we have shape information from previous layers
  if(!_layers.empty())
  {
    auto &previous_layer=_layers.back();
    if(previous_layer->IsInitialized()==true)
    {
      // Use const version of OutputShape() by casting to const
      const CAIF_Layer *const_prev_layer=previous_layer.get();
      const auto &prev_output_shape=const_prev_layer->OutputShape();
      dropout_layer->Initialize(prev_output_shape);
    }
  }
  
  _layers.push_back(std::move(dropout_layer));
  return;
}

void CAIF_NeuralNetwork::AddBatchNormalizationLayer(const float momentum,
                                                   const float epsilon
                                                  )
{
  if(_compiled==true)
  {
    THROW_CAIFE("Cannot add layers to compiled network");
  }
  
  if(momentum<0.0f || momentum>1.0f || epsilon<=0.0f)
  {
    THROW_CAIFE("Invalid batch normalization parameters");
  }
  
  // Create actual batch normalization layer implementation
  auto batch_norm_layer=std::make_unique<CAIF_BatchNormalizationLayer>(_framework,epsilon,momentum);
  
  _layers.push_back(std::move(batch_norm_layer));
  return;
}

void CAIF_NeuralNetwork::AddFlattenLayer()
{
  if(_compiled==true)
  {
    THROW_CAIFE("Cannot add layers to compiled network");
  }
  
  // Create actual flatten layer implementation
  auto flatten_layer=std::make_unique<CAIF_FlattenLayer>(_framework);
  
  _layers.push_back(std::move(flatten_layer));
  return;
}

void CAIF_NeuralNetwork::AddReshapeLayer(const std::vector<uint32_t> &target_shape)
{
  if(_compiled==true)
  {
    THROW_CAIFE("Cannot add layers to compiled network");
  }
  
  if(target_shape.empty()==true)
  {
    THROW_CAIFE("Target shape cannot be empty");
  }
  
  auto reshape_layer=std::make_unique<CAIF_ReshapeLayer>(_framework,target_shape);
  _layers.push_back(std::move(reshape_layer));
  return;
}

void CAIF_NeuralNetwork::Compile(const CAIF_OptimizerType_e optimizer_type,
                                const CAIF_LossType_e loss_type,
                                const float learning_rate
                               )
{
  if(_layers.empty())
  {
    THROW_CAIFE("Cannot compile network with no layers");
  }
  
  _optimizer_type=optimizer_type;
  _loss_type=loss_type;
  _learning_rate=learning_rate;
  
  // Create optimizer instance
  _optimizer=CreateOptimizer(optimizer_type,learning_rate);
  
  // Create loss function instance
  _loss_function=CreateLossFunction(loss_type);
  
  // Initialize layer input shapes
  InitializeLayerShapes();
  
  _compiled=true;
  return;
}

std::vector<CAIF_NeuralNetwork::CAIF_TrainingMetrics_t> CAIF_NeuralNetwork::Train(
                                                                           const CAIF_Tensor &input_data,
                                                                           const CAIF_Tensor &target_data,
                                                                           const TrainingConfig_t &config
                                                                          )
{
  if(_compiled==false)
  {
    THROW_CAIFE("Network must be compiled before training");
  }
  
  if(_optimizer==nullptr || _loss_function==nullptr)
  {
    THROW_CAIFE("Optimizer and loss function must be set before training");
  }
  
  // Validate input data
  if(input_data.Shape().empty()==true || target_data.Shape().empty()==true)
  {
    THROW_CAIFE("Input and target data cannot be empty");
  }
  
  if(input_data.Shape()[0]!=target_data.Shape()[0])
  {
    THROW_CAIFE("Input and target data must have the same batch size");
  }
  
  std::vector<CAIF_TrainingMetrics_t> training_history;
  training_history.reserve(config.epochs);
  
  try
  {
    // Avoid copying large tensors - use const references and slice views per batch
    const CAIF_Tensor &training_inputs=input_data;
    const CAIF_Tensor &training_targets=target_data;

    const uint32_t total_samples=training_inputs.Shape()[0];

    // Compute class prior once for optional bias init
    float computed_bias_val=0.0f;
    bool have_bias_val=false;
    if(_loss_type==CAIF_LossType_e::BinaryCrossEntropyWithLogits)
    {
      const float *td=training_targets.ConstData<float>();
      if(td!=nullptr)
      {
        const size_t n=training_targets.NumElements();
        double sum=0.0;
        for(size_t i=0;i<n;++i)
        {
          sum+=static_cast<double>(td[i]);
        }
        double p=(n>0)?(sum/static_cast<double>(n)):0.5;
        if(p<=0.0)
        {
          p=1e-6;
        }
        if(p>=1.0)
        {
          p=1.0-1e-6;
        }
        computed_bias_val=static_cast<float>(std::log(p/(1.0-p)));
        have_bias_val=true;
        ISE_Out::Out()<<"[TrainDiag] BCE-with-logits prior p="
                      <<static_cast<float>(p)
                      <<" bias_val="
                      <<computed_bias_val
                      <<"\n";
      }
    }

    const uint32_t num_batches=(total_samples+config.batch_size-1)/config.batch_size;
    ISE_Out::Out()<<"Total samples:"
                  <<total_samples
                  <<std::endl
                  <<"Batch Size:"
                  <<config.batch_size
                  <<std::endl
                  <<"Number of Batches:"
                  <<num_batches
                  <<std::endl;
    
    // Ensure optimizer uses training config learning rate
    if(_optimizer!=nullptr)
    {
      // Update internal learning rate to match training config
      _learning_rate=config.learning_rate;
      _optimizer->SetLearningRate(config.learning_rate);
    }
    
    // Enable first-batch trace on all backends to diagnose parity issues
    const bool trace_first_batch=true;
    if(trace_first_batch)
    {
      ISE_Out::Out()<<"[FirstBatchTrace] enabled for backend "
                    <<static_cast<int>(_framework.CurrentBackend())
                    <<"\n";
    }
    for(uint32_t epoch=0;epoch<config.epochs;++epoch)
    {
      CAIF_TrainingMetrics_t metrics;
      metrics.epoch=epoch;
      float epoch_loss=0.0f;
      uint32_t correct_predictions=0;
      // Apply bias just-in-time before the first forward of epoch 0 (only once)
      if(epoch==0 && have_bias_val==true)
      {
        MaybeApplyBinaryCrossEntropyBias(computed_bias_val,have_bias_val);
      }
      
      // Build batch order (optionally shuffled) without copying data
      std::vector<uint32_t> batch_order(num_batches);
      for(uint32_t i=0;i<num_batches;++i)
      {
        batch_order[i]=i;
      }
      if(config.shuffle_data==true)
      {
        std::mt19937 rng(epoch);
        std::shuffle(batch_order.begin(),batch_order.end(),rng);
      }
      
      // Reuse range vectors across batches to reduce allocations
      std::vector<std::pair<uint32_t,uint32_t>> input_ranges;
      std::vector<std::pair<uint32_t,uint32_t>> target_ranges;
      input_ranges.reserve(training_inputs.Shape().size());
      target_ranges.reserve(training_targets.Shape().size());

      for(uint32_t bi=0;bi<num_batches;++bi)
      {
        const uint32_t batch=batch_order[bi];
        // Calculate batch indices
        const uint32_t start_idx=batch*config.batch_size;
        const uint32_t end_idx=std::min(start_idx+config.batch_size,total_samples);
        //const uint32_t actual_batch_size=end_idx-start_idx;
        
        // Extract batch data ranges [start_idx,end_idx) on batch dimension
        input_ranges.clear();
        target_ranges.clear();
        input_ranges.push_back({start_idx,end_idx});
        target_ranges.push_back({start_idx,end_idx});

        //this is the input dimention so rgb images of 256 x 256 would be
        //256 256 3 (channels)  The first dimension is number of batches
        for(size_t dim=1; dim<training_inputs.Shape().size(); ++dim)
        {
          //ISE_Out::Out()<<"Adding input range("
          //              <<dim
          //              <<"):"
          //              <<training_inputs.Shape()[dim]
          //              <<std::endl;
                 
          input_ranges.push_back({0,training_inputs.Shape()[dim]});
        }

        for(size_t dim=1; dim<training_targets.Shape().size(); ++dim)
        {
          //ISE_Out::Out()<<"Adding target range("
          //              <<dim
          //              <<"):"
          //              <<training_targets.Shape()[dim]
          //              <<std::endl;

          target_ranges.push_back({0,training_targets.Shape()[dim]});
        }

        // Avoid copying: create batch views when slicing only along batch dim
        CAIF_Tensor batch_input(_framework);
        CAIF_Tensor batch_target(_framework);
        if(input_ranges.size()>=1 && target_ranges.size()>=1)
        {
          const bool only_batch_slice=(input_ranges.size()==training_inputs.Shape().size() &&
                                       std::all_of(input_ranges.begin()+1,input_ranges.end(),[](const auto &r)
                                                   {
                                                     return r.first==0 && r.second>0 && r.first<r.second;
                                                   }
                                                  ) &&
                                       target_ranges.size()==training_targets.Shape().size() &&
                                       std::all_of(target_ranges.begin()+1,target_ranges.end(),[](const auto &r)
                                                   {
                                                     return r.first==0 && r.second>0 && r.first<r.second;
                                                   }
                                                  )
                                      );
          if(only_batch_slice)
          {
            batch_input=training_inputs.SliceViewBatch({input_ranges[0].first,input_ranges[0].second});
            batch_target=training_targets.SliceViewBatch({target_ranges[0].first,target_ranges[0].second});
          }
          else
          {
            batch_input=training_inputs.Slice(input_ranges);
            batch_target=training_targets.Slice(target_ranges);
          }
          //AIFOut()<<"[Batch] input="<<batch_input.ToString()<<" target="<<batch_target.ToString()<<"\n";
        }
        
        // Forward pass with training=true to store intermediate values for backward pass
        CAIF_Tensor predictions=Forward(batch_input,true);
        
        // First-batch CPU trace: capture simple diagnostics for parity debugging
        std::vector<double> weight_l2_before;
        if(trace_first_batch && epoch==0 && bi==0)
        {
          for(size_t li=0; li<_layers.size(); ++li)
          {
            const auto &layer=_layers[li];
            if(layer->ParameterCount()>0)
            {
              const CAIF_Tensor &w=layer->ParameterRef(0);
              const float *wd=w.ConstData<float>();
              if(wd!=nullptr)
              {
                double acc=0.0;
                const size_t n=w.NumElements();
                for(size_t i=0;i<n;++i){acc+=static_cast<double>(wd[i])*static_cast<double>(wd[i]);}
                weight_l2_before.push_back(std::sqrt(acc/static_cast<double>(n)));
              }
              else
              {
                weight_l2_before.push_back(0.0);
              }
            }
          }
        }
        
        // Compute loss and gradient (use fused path when available)
        auto loss_and_grad=_loss_function->ComputeLossAndGradient(predictions,batch_target);
        
        // Skip per-batch logging/diagnostics to avoid host sync
        CAIF_Tensor &loss_tensor=loss_and_grad.first;
        
        // Accumulate loss on host
        float batch_loss=0.0f;
        {
          const float *loss_data=loss_tensor.ConstData<float>();
          if(loss_data==nullptr)
          {
            THROW_CAIFE("Failed to access loss data");
          }
          const uint32_t nloss=loss_tensor.NumElements();
          double local_sum=0.0;
          #ifdef _OPENMP
          #pragma omp parallel for reduction(+:local_sum)
          #endif
          for(uint32_t i=0;i<nloss;++i)
          {
            local_sum+=loss_data[i];
          }
          batch_loss=static_cast<float>(local_sum);
        }
        epoch_loss+=batch_loss;
        
        if(trace_first_batch && epoch==0 && bi==0)
        {
          double pred_mean_host=0.0;
          double loss_mean_host=0.0;
          size_t pred_count=predictions.NumElements();
          size_t loss_count=loss_tensor.NumElements();
          try
          {
            if(pred_count>0)
            {
              const float *pd=predictions.ConstData<float>();
              for(size_t i=0;i<pred_count;++i){pred_mean_host+=pd[i];}
              pred_mean_host/=static_cast<double>(pred_count);
            }
            if(loss_count>0)
            {
              const float *ld=loss_tensor.ConstData<float>();
              for(size_t i=0;i<loss_count;++i){loss_mean_host+=ld[i];}
              loss_mean_host/=static_cast<double>(loss_count);
            }
          }
          catch(...)
          {
          }
          ISE_Out::Out()<<"[FirstBatchTraceCUDA] pred_mean_host="
                        <<static_cast<float>(pred_mean_host)
                        <<", loss_mean_host="
                        <<static_cast<float>(loss_mean_host)
                        <<"\n";
        }
        
        // Compute gradients (already computed above)
        //AIFOut()<<"[Step] Computing loss gradient"<<"\n";
        CAIF_Tensor &loss_grad=loss_and_grad.second;
        // temp loss_grad debug removed
        //AIFOut()<<"[Step] Loss gradient OK"<<"\n";
        // Diagnostics disabled for performance

        // Backward pass through network
        //AIFOut()<<"[Step] Starting BackwardPass"<<"\n";
        BackwardPass(loss_grad);
        //AIFOut()<<"[Step] BackwardPass OK"<<"\n";
        
        std::vector<double> grad_l2;
        if(trace_first_batch && epoch==0 && bi==0)
        {
          for(size_t li=0; li<_layers.size(); ++li)
          {
            const auto &layer=_layers[li];
            if(layer->ParameterCount()>0)
            {
              const CAIF_Tensor &g=layer->GradientRef(0);
              const float *gd=g.ConstData<float>();
              if(gd!=nullptr)
              {
                double acc=0.0;
                const size_t n=g.NumElements();
                for(size_t i=0;i<n;++i){acc+=static_cast<double>(gd[i])*static_cast<double>(gd[i]);}
                grad_l2.push_back(std::sqrt(acc/static_cast<double>(n)));
              }
              else
              {
                grad_l2.push_back(0.0);
              }
            }
          }
        }
        
        // Update parameters using optimizer
        //AIFOut()<<"[Step] Updating parameters"<<"\n";
        UpdateNetworkParameters();
        //AIFOut()<<"[Step] Parameters updated"<<"\n";

        if(trace_first_batch && epoch==0 && bi==0)
        {
          std::vector<double> weight_l2_after;
          for(size_t li=0; li<_layers.size(); ++li)
          {
            const auto &layer=_layers[li];
            if(layer->ParameterCount()>0)
            {
              const CAIF_Tensor &w=layer->ParameterRef(0);
              const float *wd=w.ConstData<float>();
              if(wd!=nullptr)
              {
                double acc=0.0;
                const size_t n=w.NumElements();
                for(size_t i=0;i<n;++i){acc+=static_cast<double>(wd[i])*static_cast<double>(wd[i]);}
                weight_l2_after.push_back(std::sqrt(acc/static_cast<double>(n)));
              }
              else
              {
                weight_l2_after.push_back(0.0);
              }
            }
          }

          // Log compact summary: loss, mean pred/target, first few layer norms
          double pred_mean=0.0, target_mean=0.0;
          {
            const float *pd=predictions.ConstData<float>();
            const float *td=batch_target.ConstData<float>();
            if(pd!=nullptr && td!=nullptr)
            {
              const size_t n=std::min(predictions.NumElements(), batch_target.NumElements());
              for(size_t i=0;i<n;++i)
              {
                pred_mean+=static_cast<double>(pd[i]);
                target_mean+=static_cast<double>(td[i]);
              }
              if(n>0)
              {
                pred_mean/=static_cast<double>(n);
                target_mean/=static_cast<double>(n);
              }
            }
          }

          ISE_Out::Out()<<"[FirstBatchTrace] loss="
                        <<batch_loss
                        <<", pred_mean="
                        <<static_cast<float>(pred_mean)
                        <<", target_mean="
                        <<static_cast<float>(target_mean)
                        <<"\n";
          const size_t layers_logged=std::min<size_t>(weight_l2_before.size(),weight_l2_after.size());
          const size_t grads_logged=std::min<size_t>(layers_logged,grad_l2.size());
          for(size_t i=0;i<grads_logged;++i)
          {
            ISE_Out::Out()<<"[FirstBatchTrace] layer="
                          <<i
                          <<", w_l2_before="
                          <<static_cast<float>(weight_l2_before[i])
                          <<", grad_l2="
                          <<static_cast<float>(grad_l2[i])
                          <<", w_l2_after="
                          <<static_cast<float>(weight_l2_after[i])
                          <<"\n";
          }
        }
        
        // Skip per-batch accuracy to avoid host reads in CUDA path
      }
      
      // Update metrics
      metrics.loss=epoch_loss/static_cast<float>(num_batches);
      metrics.accuracy=static_cast<float>(correct_predictions)/static_cast<float>(total_samples);
      metrics.val_loss=0.0f;  // TODO: Implement validation
      metrics.val_accuracy=0.0f;  // TODO: Implement validation
      if(ShouldLogEpochSummary(epoch,(epoch+1)==config.epochs)==true)
      {
        LogEpochSummary(metrics);
      }
      
      training_history.push_back(metrics);
    }
    
    _trained=true;
    _training_epochs=config.epochs;
    
    return training_history;
  }
  catch(const std::exception &e)
  {
    THROW_CAIFE((std::string("Training failed: ")+e.what()).c_str());
  }
}

CAIF_Tensor CAIF_NeuralNetwork::Forward(const CAIF_Tensor &input,const bool training)
{
  DbgLog()<<"[DEBUG] CAIF_NeuralNetwork::Forward - Starting forward pass\n";
  DbgLog()<<"[DEBUG] Input tensor: "<<input.ToString()<<"\n";
  {
    const char *training_mode_str;
    if(training==true)
    {
      training_mode_str="true";
    }
    else
    {
      training_mode_str="false";
    }
    DbgLog()<<"[DEBUG] Training mode: "<<training_mode_str<<"\n";
  }
  
  if(_compiled==false)
  {
    ErrorLog()<<"[ERROR] Network must be compiled before forward pass\n";
    THROW_CAIFE("Network must be compiled before forward pass");
  }
  
  if(_layers.empty()==true)
  {
    ErrorLog()<<"[ERROR] Network has no layers\n";
    THROW_CAIFE("Network has no layers");
  }
 
  // Avoid copying the input; pass through as the current tensor
  const CAIF_Tensor &current_input_ref=input;
  CAIF_Tensor current_input=current_input_ref;
  
  // Forward pass through all layers
  for(size_t i=0; i<_layers.size(); ++i)
  {
    auto &layer=_layers[i];
    //AIFOut()<<"[Forward] Layer "
    //         <<i
    //         <<" ("
    //         <<layer->Description()
    //         <<")"
    //         <<"\n";
    DbgLog()<<"[DEBUG] Layer "<<i<<" ("<<layer->Description()<<") - Input: "<<current_input.ToString()<<"\n";
    
    current_input=layer->Forward(current_input,training);
    //AIFOut()<<"[Forward] Layer "
    //         <<i
    //         <<" done"
    //         <<"\n";
    
    DbgLog()<<"[DEBUG] Layer "<<i<<" output: "<<current_input.ToString()<<"\n";
  }
  
  DbgLog()<<"[DEBUG] Forward pass completed successfully\n";
  return current_input;
}

CAIF_Tensor CAIF_NeuralNetwork::Predict(const CAIF_Tensor &input)const
{
  DbgLog()<<"[DEBUG] CAIF_NeuralNetwork::Predict - Starting prediction\n";
  DbgLog()<<"[DEBUG] Input tensor: "<<input.ToString()<<"\n";
  
  if(_compiled==false)
  {
    ErrorLog()<<"[ERROR] Network must be compiled before prediction\n";
    THROW_CAIFE("Network must be compiled before prediction");
  }
  
  try
  {
    // Forward pass through all layers
    CAIF_Tensor current_output=input;
    
    for(size_t i=0; i<_layers.size(); ++i)
    {
      const auto &layer=_layers[i];
      DbgLog()<<"[DEBUG] Layer "<<i<<" ("<<layer->Description()<<") - Input: "<<current_output.ToString()<<"\n";
      
      current_output=layer->Forward(current_output,false);
      DbgLog()<<"[DEBUG] Layer "<<i<<" output: "<<current_output.ToString()<<"\n";
    }
    
    DbgLog()<<"[DEBUG] Prediction completed successfully\n";
    return current_output;
  }
  catch(const std::exception &e)
  {
    ErrorLog()<<"[ERROR] Could not make prediction: "<<e.what()<<"\n";
    THROW_CAIFE((std::string("Could not make prediction: ")+e.what()).c_str());
  }
}

CAIF_NeuralNetwork::CAIF_TrainingMetrics_t CAIF_NeuralNetwork::Evaluate(
                                                                       const CAIF_Tensor &input_data,
                                                                       const CAIF_Tensor &target_data
                                                                      )
{
  if(_compiled==false)
  {
    THROW_CAIFE("Network must be compiled before evaluation");
  }

  if(_loss_function==nullptr)
  {
    THROW_CAIFE("Loss function not configured");
  }

  // Forward pass
  CAIF_Tensor predictions=Forward(input_data,false);

  // Compute loss
  const auto loss_tensor=_loss_function->ComputeLoss(predictions,target_data);
  float loss_value=1.0f;
  if(auto loss_ptr=loss_tensor.ConstData<float>(); loss_ptr!=nullptr)
  {
    if(loss_tensor.NumElements()>0)
    {
      const float *lp=loss_ptr;
      size_t n=loss_tensor.NumElements();
      float sum=0.0f;
      for(size_t i=0;i<n;++i){sum+=lp[i];}
      loss_value=sum/static_cast<float>(n);
    }
  }

  // Compute accuracy if possible (binary or one-hot)
  float accuracy_value=0.0f;
  {
    const auto &p_shape=predictions.Shape();
    const auto &t_shape=target_data.Shape();
    if(p_shape==t_shape || 
       predictions.Type()==CAIF_DataType::CAIF_DataType_e::Float32 || 
       target_data.Type()==CAIF_DataType::CAIF_DataType_e::Float32)
    {
      auto p_ptr=predictions.ConstData<float>();
      auto t_ptr=target_data.ConstData<float>();
      if(p_ptr!=nullptr || t_ptr!=nullptr)
      {
        const float *p=p_ptr;
        const float *t=t_ptr;
        if(p_shape.size()>=2)
        {
          const size_t batch=static_cast<size_t>(p_shape[0]);
          size_t correct=0;
          for(size_t i=0;i<batch;++i)
          {
            const float py=p[i];
            const float y=t[i];
            int pred;
            if(_loss_type==CAIF_LossType_e::BinaryCrossEntropyWithLogits)
            {
              if(py>0.0f)
              {
                pred=1;
              }
              else
              {
                pred=0;
              }
            }
            else
            {
              if(py>0.5f)
              {
                pred=1;
              }
              else
              {
                pred=0;
              }
            }
            int actual;
            if(y>0.5f)
            {
              actual=1;
            }
            else
            {
              actual=0;
            }
            if(pred==actual)
            {
              ++correct;
            }
          }
          if(batch>0)
          {
            accuracy_value=static_cast<float>(correct)*100.0f/static_cast<float>(batch);
          }  
        }
        else if(p_shape.size()>=2)
        {
          const size_t batch=static_cast<size_t>(p_shape[0]);
          const size_t classes=static_cast<size_t>(p_shape.back());
          size_t correct=0;
          for(size_t b=0;b<batch;++b)
          {
            size_t base=b*classes;
            size_t p_arg=0; float p_max=p[base];
            size_t t_arg=0; float t_max=t[base];
            for(size_t c=1;c<classes;++c)
            {
              float pv=p[base+c]; if(pv>p_max){p_max=pv; p_arg=c;}
              float tv=t[base+c]; if(tv>t_max){t_max=tv; t_arg=c;}
            }
            if(p_arg==t_arg){++correct;}
          }
          if(batch>0)
          {
            accuracy_value=static_cast<float>(correct)*100.0f/static_cast<float>(batch);
          }  
        }
      }
    }
  }

  CAIF_TrainingMetrics_t metrics{};
  metrics.loss=loss_value;
  metrics.accuracy=accuracy_value;
  metrics.val_loss=0.0f;
  metrics.val_accuracy=0.0f;
  metrics.epoch=_training_epochs;
  metrics.has_accuracy=true;
  return metrics;
}

CAIF_Tensor CAIF_NeuralNetwork::ExtractFeatures(const CAIF_Tensor &input,const int32_t layer_index)
{
  if(_compiled==false)
  {
    THROW_CAIFE("Network must be compiled before feature extraction");
  }
  
  if(_layers.empty())
  {
    THROW_CAIFE("Network has no layers");
  }
  
  // Determine target layer index
  int32_t target_layer_index;
  if(layer_index<0)
  {
    // Use penultimate layer (second to last)
    if(_layers.size()<2)
    {
      THROW_CAIFE("Network must have at least 2 layers for automatic feature extraction");
    }
    target_layer_index=static_cast<int32_t>(_layers.size())-2;
  }
  else
  {
    target_layer_index=layer_index;
  }
  
  // Validate layer index
  if(target_layer_index<0 || target_layer_index>=static_cast<int32_t>(_layers.size()))
  {
    THROW_CAIFE("Invalid layer index for feature extraction");
  }
  
  CAIF_Tensor current_input=input;
  
  // Forward pass up to target layer (inclusive)
  for(int32_t i=0;i<=target_layer_index;++i)
  {
    current_input=_layers[static_cast<size_t>(i)]->Forward(current_input,false);
  }
  
  return current_input;
}

void CAIF_NeuralNetwork::SaveModel(const std::string &filepath)const
{
  (void)filepath;
  // TODO: Implement SafeTensors serialization for CPU-based CAIF_NeuralNetwork
  THROW_CAIFE("SaveModel not implemented - use CAIF_DeviceNetwork with SafeTensors format");
}

void CAIF_NeuralNetwork::LoadModel(const std::string &filepath)
{
  (void)filepath;
  // TODO: Implement SafeTensors serialization for CPU-based CAIF_NeuralNetwork
  THROW_CAIFE("LoadModel not implemented - use CAIF_DeviceNetwork with SafeTensors format");
}

std::string CAIF_NeuralNetwork::ExportArchitecture()const
{
  std::ostringstream oss;
  oss<<"Neural Network Architecture:\n";
  oss<<"Layers: "<<_layers.size()<<"\n";
  oss<<"Input Shape: [";
  for(size_t i=0;i<_input_shape.size();++i)
  {
    if(i>0)oss<<",";
    oss<<_input_shape[i];
  }
  oss<<"]\n";
  oss<<"Output Shape: [";
  for(size_t i=0;i<_output_shape.size();++i)
  {
    if(i>0)oss<<",";
    oss<<_output_shape[i];
  }
  oss<<"]\n";
  oss<<"Compiled: "<<(_compiled?"Yes":"No")<<"\n";
  oss<<"Trained: "<<(_trained?"Yes":"No")<<"\n";
  
  return oss.str();
}

void CAIF_NeuralNetwork::ApplyRegularization(const float l1_lambda,const float l2_lambda)
{
  if(l1_lambda<0.0f || l2_lambda<0.0f)
  {
    THROW_CAIFE("Regularization lambdas must be non-negative");
  }
  
  // TODO: Implement regularization
  return;
}

void CAIF_NeuralNetwork::ResetWeights(const uint32_t seed)
{
  if(_layers.empty())
  {
    THROW_CAIFE("No layers to reset");
  }
  
  InitializeWeights(seed);
  return;
}

void CAIF_NeuralNetwork::Backward(const CAIF_Tensor &output_gradient)
{
  if(_layers.empty())
  {
    THROW_CAIFE("No layers for backward pass");
  }
  
  CAIF_Tensor current_gradient=output_gradient;
  
  // Backward pass through layers in reverse order
  for(auto it=_layers.rbegin();it!=_layers.rend();++it)
  {
    current_gradient=(*it)->Backward(current_gradient);
  }
  
  return;
}

void CAIF_NeuralNetwork::InitializeWeights(const uint32_t seed)
{
  std::mt19937 rng(seed);
  
  for(auto &layer:_layers)
  {
    if(layer->HasParameters()==true)
    {
      layer->ResetParameters(seed);
    }
  }
  
  return;
}

void CAIF_NeuralNetwork::ValidateArchitecture()const
{
  if(_layers.empty())
  {
    THROW_CAIFE("Network has no layers");
  }
  
  if(_layers.size()>g_caif_max_layers)
  {
    THROW_CAIFE("Too many layers in network");
  }
  
  if(_input_shape.empty())
  {
    THROW_CAIFE("Input shape not set");
  }
  
  return;
}

std::vector<uint32_t> CAIF_NeuralNetwork::CalculateOutputShape()const
{
  if(_layers.empty() || _input_shape.empty())
  {
    THROW_CAIFE("Cannot calculate output shape");
  }
  
  std::vector<uint32_t> current_shape=_input_shape;
  
  for(const auto &layer:_layers)
  {
    current_shape=layer->CalculateOutputShape(current_shape);
  }
  
  return current_shape;
}

void CAIF_NeuralNetwork::ShuffleData(CAIF_Tensor &input_data,CAIF_Tensor &target_data,const uint32_t seed)const
{
  // Validate input dimensions
  if(input_data.Shape().empty() || target_data.Shape().empty())
  {
    THROW_CAIFE("Input and target data cannot be empty");
  }
  
  if(input_data.Shape()[0]!=target_data.Shape()[0])
  {
    THROW_CAIFE("Input and target data must have same batch size");
  }
  
  const uint32_t num_samples=input_data.Shape()[0];
  if(num_samples<=1)
  {
    return;  // Nothing to shuffle
  }
  
  try
  {
    // Build a shuffled index permutation
    std::vector<uint32_t> indices(num_samples);
    for(uint32_t i=0;i<num_samples;++i){indices[i]=i;}
    std::mt19937 rng(seed);
    std::shuffle(indices.begin(),indices.end(),rng);
    
    // Create shuffled copies
    CAIF_Tensor shuffled_inputs=input_data;
    CAIF_Tensor shuffled_targets=target_data;
    
    // Copy one sample at a time according to permutation
    for(uint32_t new_idx=0; new_idx<num_samples; ++new_idx)
    {
      const uint32_t src_idx=indices[new_idx];
      
      // Inputs: slice source and destination
      std::vector<std::pair<uint32_t,uint32_t>> src_in_ranges;
      src_in_ranges.push_back({src_idx,src_idx+1});

      std::vector<std::pair<uint32_t,uint32_t>> dst_in_ranges;
      dst_in_ranges.push_back({new_idx,new_idx+1});

      for(size_t d=1; d<input_data.Shape().size(); ++d)
      {
        src_in_ranges.push_back({0,input_data.Shape()[d]});
        dst_in_ranges.push_back({0,input_data.Shape()[d]});
      }

      CAIF_Tensor src_in_slice=input_data.Slice(src_in_ranges);
      CAIF_Tensor dst_in_slice=shuffled_inputs.Slice(dst_in_ranges);

      const float *src_in_ptr=src_in_slice.ConstData<float>();
      float *dst_in_ptr=dst_in_slice.MutableData<float>();

      if(src_in_ptr==nullptr || dst_in_ptr==nullptr)
      {
        THROW_CAIFE("Failed to access input tensor data during shuffle");
      }
      std::memcpy(dst_in_ptr,src_in_ptr,(dst_in_slice.NumElements())*sizeof(float));
      
      // Targets: slice source and destination
      std::vector<std::pair<uint32_t,uint32_t>> src_t_ranges;
      src_t_ranges.push_back({src_idx,src_idx+1});

      std::vector<std::pair<uint32_t,uint32_t>> dst_t_ranges;
      dst_t_ranges.push_back({new_idx,new_idx+1});

      for(size_t d=1; d<target_data.Shape().size(); ++d)
      {
        src_t_ranges.push_back({0,target_data.Shape()[d]});
        dst_t_ranges.push_back({0,target_data.Shape()[d]});
      }

      CAIF_Tensor src_t_slice=target_data.Slice(src_t_ranges);
      CAIF_Tensor dst_t_slice=shuffled_targets.Slice(dst_t_ranges);

      const float *src_t_ptr=src_t_slice.ConstData<float>();
      float *dst_t_ptr=dst_t_slice.MutableData<float>();

      if(src_t_ptr==nullptr || dst_t_ptr==nullptr)
      {
        THROW_CAIFE("Failed to access target tensor data during shuffle");
      }
      std::memcpy(dst_t_ptr,src_t_ptr,(dst_t_slice.NumElements())*sizeof(float));
    }
    
    // Replace originals
    input_data=shuffled_inputs;
    target_data=shuffled_targets;
    return;
  }
  catch(const std::exception &e)
  {
    THROW_CAIFE((std::string("Data shuffling failed: ")+e.what()).c_str());
  }
}

std::tuple<CAIF_Tensor,CAIF_Tensor,CAIF_Tensor,CAIF_Tensor> CAIF_NeuralNetwork::SplitData(
                                                                              const CAIF_Tensor &input_data,
                                                                              const CAIF_Tensor &target_data,
                                                                              const float validation_split
                                                                                    )const
{
  if(validation_split<0.0f || validation_split>=1.0f)
  {
    THROW_CAIFE("Validation split must be between 0.0 and 1.0");
  }
  
  // Validate input dimensions
  if(input_data.Shape().empty() || target_data.Shape().empty())
  {
    THROW_CAIFE("Input and target data cannot be empty");
  }
  
  if(input_data.Shape()[0]!=target_data.Shape()[0])
  {
    THROW_CAIFE("Input and target data must have same batch size");
  }
  
  const uint32_t total_samples=input_data.Shape()[0];
  const uint32_t validation_samples=static_cast<uint32_t>(total_samples*validation_split);
  const uint32_t training_samples=total_samples-validation_samples;
  
  if(training_samples==0 || validation_samples==0)
  {
    THROW_CAIFE("Split results in empty training or validation set");
  }
  
  // Create slice ranges for training data
  std::vector<std::pair<uint32_t,uint32_t>> train_input_ranges;
  std::vector<std::pair<uint32_t,uint32_t>> train_target_ranges;
  
  // First dimension (batch): 0 to training_samples
  train_input_ranges.push_back({0,training_samples});
  train_target_ranges.push_back({0,training_samples});
  
  // Remaining dimensions: full range
  for(size_t i=1;i<input_data.Shape().size();++i)
  {
    train_input_ranges.push_back({0,input_data.Shape()[i]});
  }

  for(size_t i=1;i<target_data.Shape().size();++i)
  {
    train_target_ranges.push_back({0,target_data.Shape()[i]});
  }
  
  // Create slice ranges for validation data
  std::vector<std::pair<uint32_t,uint32_t>> val_input_ranges;
  std::vector<std::pair<uint32_t,uint32_t>> val_target_ranges;
  
  // First dimension (batch): training_samples to total_samples
  val_input_ranges.push_back({training_samples,total_samples});
  val_target_ranges.push_back({training_samples,total_samples});
  
  // Remaining dimensions: full range
  for(size_t i=1;i<input_data.Shape().size();++i)
  {
    val_input_ranges.push_back({0,input_data.Shape()[i]});
  }
  for(size_t i=1;i<target_data.Shape().size();++i)
  {
    val_target_ranges.push_back({0,target_data.Shape()[i]});
  }
  
  try
  {
    // Slice the data
    CAIF_Tensor train_input=input_data.Slice(train_input_ranges);
    CAIF_Tensor train_target=target_data.Slice(train_target_ranges);
    CAIF_Tensor val_input=input_data.Slice(val_input_ranges);
    CAIF_Tensor val_target=target_data.Slice(val_target_ranges);
    
    return std::make_tuple(std::move(train_input),
                           std::move(train_target),
                           std::move(val_input),
                           std::move(val_target));
  }
  catch(const std::exception &e)
  {
    THROW_CAIFE((std::string("Data splitting failed: ")+e.what()).c_str());
  }
}

std::unique_ptr<CAIF_Optimizer> CAIF_NeuralNetwork::CreateOptimizer(
                                                               const CAIF_OptimizerType_e optimizer_type,
                                                               const float learning_rate
                                                                                           )const
{
  // const_cast needed because optimizer constructors require non-const framework reference
  CAIF_Framework &framework_ref=const_cast<CAIF_Framework&>(_framework);
  switch(optimizer_type)
  {
    case CAIF_OptimizerType_e::SGD:
      return std::make_unique<CAIF_SGDOptimizer>(framework_ref,learning_rate);
    case CAIF_OptimizerType_e::Adam:
      return std::make_unique<CAIF_AdamOptimizer>(framework_ref,learning_rate);
    case CAIF_OptimizerType_e::RMSprop:
      THROW_CAIFE("RMSprop optimizer not yet implemented");
    case CAIF_OptimizerType_e::AdaGrad:
      THROW_CAIFE("AdaGrad optimizer not yet implemented");
    default:
      THROW_CAIFE("Unknown optimizer type");
  }
}

std::unique_ptr<CAIF_LossFunction> CAIF_NeuralNetwork::CreateLossFunction(
                                                                               const CAIF_LossType_e loss_type
                                                                                                 )const
{
  switch(loss_type)
  {
    case CAIF_LossType_e::MeanSquaredError:
      return std::make_unique<CAIF_MeanSquaredErrorLoss>();
    case CAIF_LossType_e::CrossEntropy:
      return std::make_unique<CAIF_CrossEntropyLoss>();
    case CAIF_LossType_e::BinaryCrossEntropy:
      return std::make_unique<CAIF_BinaryCrossEntropyLoss>();
    case CAIF_LossType_e::BinaryCrossEntropyWithLogits:
      return std::make_unique<CAIF_BinaryCrossEntropyWithLogitsLoss>();
    case CAIF_LossType_e::CategoricalCrossEntropy:
      return std::make_unique<CAIF_CategoricalCrossEntropyLoss>();
    default:
      THROW_CAIFE("Unknown loss function type");
  }
}

void CAIF_NeuralNetwork::InitializeLayerShapes()
{
  if(_learning_rate<=0.0f)
  {
    THROW_CAIFE("Learning rate must be positive");
  }
  
  if(_input_shape.empty())
  {
    THROW_CAIFE("Input shape must be set before compilation");
  }
  
  // Initialize all layers with proper shapes
  std::vector<uint32_t> current_shape=_input_shape;
  
  for(auto &layer:_layers)
  {
    // Initialize layer with current shape
    layer->Initialize(current_shape);
    
    //  output shape for next layer
    current_shape=layer->CalculateOutputShape(current_shape);
  }
  
  // Set final output shape
  _output_shape=current_shape;
  
  // Validate architecture
  ValidateArchitecture();
  
  return;
}

void CAIF_NeuralNetwork::MaybeApplyBinaryCrossEntropyBias(const float computed_bias_val,bool &have_bias_val)
{
  if(have_bias_val==false)
  {
    return;
  }
  for(int li=static_cast<int>(_layers.size())-1; li>=0; --li)
  {
    auto &layer=_layers[static_cast<size_t>(li)];
    if(layer->LayerType()==CAIF_LayerType_e::Dense)
    {
      CAIF_DenseLayer *dense=dynamic_cast<CAIF_DenseLayer*>(layer.get());
      if(dense!=nullptr)
      {
        dense->SetBias(computed_bias_val);
        ISE_Out::Out()<<"[TrainDiag] SetBias epoch0 value="
                      <<computed_bias_val
                      <<"\n";
        have_bias_val=false;
      }
      break;
    }
  }
}

bool CAIF_NeuralNetwork::ShouldLogBatchDetails(
  const uint32_t epoch,
  const uint32_t batch_index,
  const bool is_final_epoch)const
{
  if(CAIF_Settings::TrainLog()==false)
  {
    return false;
  }
  if(batch_index!=0)
  {
    return false;
  }
  if(epoch<5)
  {
    return true;
  }
  if((epoch%10)==0)
  {
    return true;
  }
  return is_final_epoch;
}

bool CAIF_NeuralNetwork::ShouldLogEpochSummary(const uint32_t epoch,const bool is_final_epoch)const
{
  if(CAIF_Settings::TrainLog()==false)
  {
    return false;
  }
  if(epoch<5)
  {
    return true;
  }
  if((epoch%10)==0)
  {
    return true;
  }
  return is_final_epoch;
}

void CAIF_NeuralNetwork::LogResidualStats(
  const CAIF_Tensor &predictions,
  const CAIF_Tensor &batch_target,
  const uint32_t epoch)const
{
  const float *pp=predictions.ConstData<float>();
  const float *tt=batch_target.ConstData<float>();
  const size_t pn=predictions.NumElements();
  const size_t tn=batch_target.NumElements();
  if(pp==nullptr||tt==nullptr||pn==0||tn==0)
  {
    return;
  }
  const size_t n=(pn<tn)?pn:tn;
  if(n==0)
  {
    return;
  }
  double abs_sum=0.0;
  double max_abs=0.0;
  for(size_t i=0;i<n;++i)
  {
    const double diff=static_cast<double>(pp[i]-tt[i]);
    const double abs_val=std::fabs(diff);
    abs_sum=abs_sum+abs_val;
    if(abs_val>max_abs)
    {
      max_abs=abs_val;
    }
  }
  const double mean_abs=abs_sum/static_cast<double>(n);
  ISE_Out::Out()<<"[DBG] Residual@epoch "
                <<epoch
                <<" b0 mean|res|="
                <<static_cast<float>(mean_abs)
                <<" max|res|="
                <<static_cast<float>(max_abs)
                <<"\n";
}

void CAIF_NeuralNetwork::LogBatchStatistics(
  const CAIF_Tensor &predictions,
  const CAIF_Tensor &batch_target,
  const uint32_t epoch)const
{
  try
  {
    const float *pp=predictions.ConstData<float>();
    const float *tt=batch_target.ConstData<float>();
    const size_t pn=predictions.NumElements();
    const size_t tn=batch_target.NumElements();
    if(pp==nullptr||tt==nullptr||pn==0||tn==0)
    {
      return;
    }
    double p_sum=0.0;
    double p_sq=0.0;
    for(size_t i=0;i<pn;++i)
    {
      const double v=pp[i];
      p_sum=p_sum+v;
      p_sq=p_sq+v*v;
    }
    const double p_mean=p_sum/static_cast<double>(pn);
    const double p_var=(p_sq/static_cast<double>(pn))-(p_mean*p_mean);
    double t_sum=0.0;
    double t_sq=0.0;
    for(size_t i=0;i<tn;++i)
    {
      const double v=tt[i];
      t_sum=t_sum+v;
      t_sq=t_sq+v*v;
    }
    const double t_mean=t_sum/static_cast<double>(tn);
    const double t_var=(t_sq/static_cast<double>(tn))-(t_mean*t_mean);
    ISE_Out::Out()<<"[TrainLog] epoch="
                  <<epoch
                  <<" batch0 pred_mean="
                  <<static_cast<float>(p_mean)
                  <<" pred_std="
                  <<static_cast<float>(p_var>0.0?std::sqrt(p_var):0.0)
                  <<" targ_mean="
                  <<static_cast<float>(t_mean)
                  <<" targ_std="
                  <<static_cast<float>(t_var>0.0?std::sqrt(t_var):0.0)
                  <<std::endl;
  }
  catch(const std::exception &)
  {
  }
}

double CAIF_NeuralNetwork::ComputeFirstDenseLayerWeightL2()const
{
  if(_layers.empty()==true)
  {
    return 0.0;
  }
  const auto params=_layers[0]->Parameters();
  if(params.empty()==true)
  {
    return 0.0;
  }
  const float *w=params[0].ConstData<float>();
  if(w==nullptr)
  {
    return 0.0;
  }
  double sum=0.0;
  const size_t count=params[0].NumElements();
  for(size_t i=0;i<count;++i)
  {
    const double v=static_cast<double>(w[i]);
    sum=sum+v*v;
  }
  return std::sqrt(sum);
}

void CAIF_NeuralNetwork::LogWeightL2Delta(const double before,const double after,const uint32_t epoch)const
{
  const double delta=after-before;
  ISE_Out::Out()<<"[DBG] WeightL2 layer0 epoch="
                <<epoch
                <<" before="
                <<static_cast<float>(before)
                <<" after="
                <<static_cast<float>(after)
                <<" delta="
                <<static_cast<float>(delta)
                <<"\n";
}

void CAIF_NeuralNetwork::LogEpochSummary(const CAIF_TrainingMetrics_t &metrics)const
{
  try
  {
    float w_mean=0.0f;
    float w_std=0.0f;
    float b_mean=0.0f;
    if(_layers.empty()==false)
    {
      auto params=_layers[0]->Parameters();
      if(params.empty()==false)
      {
        const auto &w=params[0];
        auto sp=w.AsSpan<const float>();
        if(sp.empty()==false)
        {
          double sum=0.0;
          double sq=0.0;
          for(size_t i=0;i<sp.size();++i)
          {
            const double v=sp[i];
            sum=sum+v;
            sq=sq+v*v;
          }
          const double mean=sum/static_cast<double>(sp.size());
          const double var=(sq/static_cast<double>(sp.size()))-(mean*mean);
          w_mean=static_cast<float>(mean);
          w_std=static_cast<float>(var>0.0?std::sqrt(var):0.0);
        }
      }
      if(params.size()>=2)
      {
        const auto &b=params[1];
        auto sb=b.AsSpan<const float>();
        if(sb.empty()==false)
        {
          double sum=0.0;
          for(size_t i=0;i<sb.size();++i)
          {
            sum=sum+static_cast<double>(sb[i]);
          }
          b_mean=static_cast<float>(sum/static_cast<double>(sb.size()));
        }
      }
    }
    float last_bias_mean=0.0f;
    bool has_last_bias=false;
    for(int32_t li=static_cast<int32_t>(_layers.size())-1; li>=0; --li)
    {
      const auto &layer=_layers[static_cast<size_t>(li)];
      if(layer->LayerType()==CAIF_LayerType_e::Dense)
      {
        auto layer_params=layer->Parameters();
        if(layer_params.size()>=2)
        {
          const auto &bias_tensor=layer_params[1];
          auto bias_span=bias_tensor.AsSpan<const float>();
          if(bias_span.empty()==false)
          {
            double sum=0.0;
            for(const float v:bias_span)
            {
              sum=sum+static_cast<double>(v);
            }
            last_bias_mean=static_cast<float>(sum/static_cast<double>(bias_span.size()));
            has_last_bias=true;
          }
        }
        break;
      }
    }
    ISE_Out::Out()<<"[TrainLog] epoch="
                  <<metrics.epoch
                  <<" loss="
                  <<metrics.loss
                  <<" lr="
                  <<_learning_rate
                  <<" w0_mean="
                  <<w_mean
                  <<" w0_std="
                  <<w_std
                  <<" b0_mean="
                  <<b_mean;
    if(has_last_bias==true)
    {
      ISE_Out::Out()<<" last_bias_mean="
                    <<last_bias_mean;
    }
    ISE_Out::Out()<<std::endl;
  }
  catch(const std::exception &)
  {
  }
}

void CAIF_NeuralNetwork::BackwardPass(const CAIF_Tensor &output_gradient)
{
  try
  {
    DbgLog()<<"[DEBUG] CAIF_NeuralNetwork::BackwardPass - Starting backward pass\n";
    DbgLog()<<"[DEBUG] Initial gradient: "<<output_gradient.ToString()<<"\n";
    
    CAIF_Tensor current_gradient=output_gradient;
    
    // Propagate gradient backward through all layers
    for(int i=static_cast<int>(_layers.size())-1;i>=0;--i)
    {
      DbgLog()<<"[DEBUG] Layer "
                <<i
                <<" ("
                <<_layers[i]->Description()
                <<") - Input gradient: "
                <<current_gradient.ToString()
                <<"\n";
      // try
      // {
      //   std::cout<<"[GradDiag] before layer="<<i
      //            <<" type=\""<<_layers[i]->Description()<<"\" "
      //            <<"shape="<<current_gradient.ToString()<<"\n";
      // }
      // catch(const std::exception &)
      // {
      // }
      
      // Gradient diagnostics (min/max/norm/nz)
      try
      {
        auto gptr=current_gradient.ConstData<float>();
        if(gptr!=nullptr)
        {
          const float *gd=gptr;
          const size_t gn=current_gradient.NumElements();
          float gmin=1e30f;
          float gmax=-1e30f;
          double l2=0.0;
          size_t nz=0;
          for(size_t gi=0; gi<gn; ++gi)
          {
            const float v=gd[gi];
            if(v<gmin)
            {
              gmin=v;
            }
            if(v>gmax)
            {
              gmax=v;
            }
            l2+=static_cast<double>(v)*static_cast<double>(v);
            if(v!=0.0f)
            {
              ++nz;
            }
          }
          DbgLog()<<"[DEBUG] Grad stats before layer "<<i
                   <<": min="<<gmin
                   <<" max="<<gmax
                   <<" l2="<<std::sqrt(l2)
                   <<" nz="<<nz<<"/"<<gn<<"\n";
          // std::cout<<"[GradDiag] before layer="<<i
          //          <<" min="<<gmin
          //          <<" max="<<gmax
          //          <<" l2="<<std::sqrt(l2)
          //          <<" nz="<<nz<<"/"<<gn
          //          <<"\n";
        }
      }
      catch(const std::exception &)
      {
      }
      
      current_gradient=_layers[i]->Backward(current_gradient);
      
      DbgLog()<<"[DEBUG] Layer "
                <<i
                <<" output gradient: "
                <<current_gradient.ToString()
                <<"\n";
      try
      {
        auto gptr2=current_gradient.ConstData<float>();
        if(gptr2!=nullptr)
        {
          const float *gd2=gptr2;
          const size_t gn2=current_gradient.NumElements();
          float gmin2=1e30f;
          float gmax2=-1e30f;
          double l22=0.0;
          size_t nz2=0;
          for(size_t gi=0; gi<gn2; ++gi)
          {
            const float v=gd2[gi];
            if(v<gmin2)
            {
              gmin2=v;
            }
            if(v>gmax2)
            {
              gmax2=v;
            }
            l22+=static_cast<double>(v)*static_cast<double>(v);
            if(v!=0.0f)
            {
              ++nz2;
            }
          }
          DbgLog()<<"[DEBUG] Grad stats after layer "<<i
                   <<": min="<<gmin2
                   <<" max="<<gmax2
                   <<" l2="<<std::sqrt(l22)
                   <<" nz="<<nz2<<"/"<<gn2<<"\n";
          // std::cout<<"[GradDiag] after layer="<<i
          //          <<" min="<<gmin2
          //          <<" max="<<gmax2
          //          <<" l2="<<std::sqrt(l22)
          //          <<" nz="<<nz2<<"/"<<gn2<<"\n";
        }
      }
      catch(const std::exception &)
      {
      }
    }
    
    DbgLog()<<"[DEBUG] Backward pass completed successfully\n";
    return;
  }
  catch(const std::exception &e)
  {
    ErrorLog()<<"[ERROR] Backward pass failed: "<<e.what()<<"\n";
    THROW_CAIFE((std::string("Backward pass failed: ")+e.what()).c_str());
  }
}

void CAIF_NeuralNetwork::ComputeGradients(const CAIF_Tensor &output_gradient)
{
  try
  {
    // Assumes a forward has run with training=true so layers cached intermediates
    BackwardPass(output_gradient);
    return;
  }
  catch(const std::exception &e)
  {
    THROW_CAIFE((std::string("ComputeGradients failed: ")+e.what()).c_str());
  }
}

void CAIF_NeuralNetwork::UpdateNetworkParameters()
{
  try
  {
    // Collect parameters and gradients using new direct-access methods
    std::vector<CAIF_Tensor> all_parameters;
    std::vector<CAIF_Tensor> all_gradients;
    
    for(const auto &layer:_layers)
    {
      const size_t param_count=layer->ParameterCount();
      for(size_t i=0;i<param_count;++i)
      {
        all_parameters.push_back(layer->ParameterRef(i));
        all_gradients.push_back(layer->GradientRef(i));
      }
    }
    
    if(all_parameters.empty()==true)
    {
      return;
    }

    // Use optimizer to update parameters
    if(_optimizer->OptimizerType()==CAIF_OptimizerType_e::Adam)
    {
      CAIF_AdamOptimizer *adam=dynamic_cast<CAIF_AdamOptimizer*>(_optimizer.get());
      if(adam==nullptr)
      {
        THROW_CAIFE("Optimizer type mismatch for Adam path");
      }
      adam->ApplyGradients(all_parameters,all_gradients);
    }
    else
    {
      auto updated_parameters=_optimizer->UpdateParameters(all_parameters,all_gradients);
      all_parameters=std::move(updated_parameters);
    }
    
    // Distribute updated values back to layers
    size_t param_idx=0;
    for(auto &layer:_layers)
    {
      const size_t param_count=layer->ParameterCount();
      for(size_t i=0;i<param_count;++i)
      {
        if(param_idx>=all_parameters.size())
        {
          THROW_CAIFE("Parameter index out of range during update");
        }
        
        layer->ParameterRef(i)=all_parameters[param_idx];
        ++param_idx;
      }
    }
    
    return;
  }
  CCAIF_CATCH_BLOCK()
}

uint32_t CAIF_NeuralNetwork::CalculateAccuracy(
                                              const CAIF_Tensor &predictions,
                                              const CAIF_Tensor &targets
                                             )const
{
  try
  {
    const auto &pred_shape=predictions.Shape();
    const auto &target_shape=targets.Shape();
    
    if(pred_shape!=target_shape || pred_shape.empty())
    {
      return 0;  // Shape mismatch
    }
    
    const uint32_t batch_size=pred_shape[0];
    uint32_t correct=0;
    
    //  data pointers
    const float *pred_data=predictions.ConstData<float>();
    const float *target_data=targets.ConstData<float>();
    
    if(pred_data==nullptr || target_data==nullptr)
    {
      return 0;  // Failed to access data
    }
    
    if(pred_shape.size()==2)
    {
      const uint32_t num_classes=pred_shape[1];
      
      if(num_classes==1)
      {
        // Binary classification
        // If using logits (BCE-with-logits), threshold at 0; otherwise threshold at 0.5
        for(uint32_t b=0;b<batch_size;++b)
        {
          const uint32_t idx=b;  // [batch,1]
          int pred_label;
          if(_loss_type==CAIF_LossType_e::BinaryCrossEntropyWithLogits)
          {
            pred_label=(pred_data[idx]>0.0f)?1:0;
          }
          else
          {
            pred_label=(pred_data[idx]>0.5f)?1:0;
          }
          const int true_label=(target_data[idx]>0.5f)?1:0;
          if(pred_label==true_label)
          {
            ++correct;
          }
        }
      }
      else
      {
        // Multi-class classification: argmax over classes
        for(uint32_t b=0;b<batch_size;++b)
        {
          uint32_t pred_class=0;
          uint32_t target_class=0;
          float max_pred=-std::numeric_limits<float>::infinity();
          float max_target=-std::numeric_limits<float>::infinity();
          
          for(uint32_t c=0;c<num_classes;++c)
          {
            const uint32_t idx=b*num_classes+c;
            if(pred_data[idx]>max_pred){max_pred=pred_data[idx]; pred_class=c;}
            if(target_data[idx]>max_target){max_target=target_data[idx]; target_class=c;}
          }
          if(pred_class==target_class){++correct;}
        }
      }
    }
    else
    {
      // Regression case: count as correct if within threshold
      const float threshold=0.1f;  // 10% tolerance
      
      for(uint32_t i=0;i<predictions.NumElements();++i)
      {
        if(std::abs(pred_data[i]-target_data[i])<threshold)
        {
          ++correct;
        }
      }
    }
    
    return correct;
  }
  catch(const std::exception &e)
  {
    return 0;  // Error occurred
  }
}

/**
 * @brief  optimizer state tensors
 * @return Vector of optimizer state tensors
 */
std::vector<CAIF_Tensor> CAIF_NeuralNetwork::OptimizerState()const
{
  if(_optimizer==nullptr)
  {
    return {};
  }
  
  //  optimizer state for training resumption
  return _optimizer->State();
}

/**
 * @brief Set optimizer state tensors
 * @param state Vector of optimizer state tensors
 * @return Expected with void on success or error message
 */
void CAIF_NeuralNetwork::SetOptimizerState(const std::vector<CAIF_Tensor> &state)
{
  if(_optimizer==nullptr)
  {
    THROW_CAIFE("No optimizer configured");
  }
  
  _optimizer->SetState(state);
  return;
}

CAIF_Layer& CAIF_NeuralNetwork::Layer(const uint32_t index)
{
  if(index>=_layers.size())
  {
    THROW_CAIFE("Layer index out of range");
  }
  
  return *_layers[index];
}

const CAIF_Layer& CAIF_NeuralNetwork::Layer(const uint32_t index)const
{
  if(index>=_layers.size())
  {
    THROW_CAIFE("Layer index out of range");
  }
  
  return *_layers[index];
}

void CAIF_NeuralNetwork::ValidateInputShape(const std::vector<uint32_t> &shape)const
{
  if(shape.empty())
  {
    THROW_CAIFE("Input shape cannot be empty");
  }
  
  if(shape.size()<2)
  {
    THROW_CAIFE("Input shape must have at least 2 dimensions [batch_size,features...]");
  }
  
  if(shape[0]==0)
  {
    THROW_CAIFE("Batch size (first dimension) must be greater than 0");
  }
  
  for(size_t i=1; i<shape.size(); ++i)
  {
    if(shape[i]==0)
    {
      THROW_CAIFE("All feature dimensions must be greater than 0");
    }
  }
  
  return;
}

void CAIF_NeuralNetwork::SetInputShape(const std::vector<uint32_t> &shape)
{
  ValidateInputShape(shape);
  
  _input_shape=shape;
  return;
}

}//end instance namespace
