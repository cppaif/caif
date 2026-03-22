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
 * @file aif_serialization_constants.h
 * @brief Constants for model serialization in AIF framework
 * @author AIF Development Team
 * @version 1.0
 * @date 2024
 */

#pragma once

namespace instance
{
  // JSON+Binary format constants
  namespace CAIF_JSONBinaryConstants
  {
    // File extensions
    constexpr const char* g_caif_json_extension=".json";
    constexpr const char* g_caif_binary_extension=".bin";
    
    // Format information
    constexpr const char* g_caif_format_name="JSON+Binary";
    constexpr uint32_t g_caif_format_version=1;
    
    // JSON field names
    constexpr const char* g_caif_field_format_version="format_version";
    constexpr const char* g_caif_field_is_compiled="is_compiled";
    constexpr const char* g_caif_field_is_trained="is_trained";
    constexpr const char* g_caif_field_input_shape="input_shape";
    constexpr const char* g_caif_field_output_shape="output_shape";
    constexpr const char* g_caif_field_learning_rate="learning_rate";
    constexpr const char* g_caif_field_layers="layers";
    constexpr const char* g_caif_field_layer_index="layer_index";
    constexpr const char* g_caif_field_layer_type="layer_type";
    constexpr const char* g_caif_field_config="config";
    
    // Layer type names
    constexpr const char* g_caif_layer_type_dense="dense";
    constexpr const char* g_caif_layer_type_conv2d="conv2d";
    constexpr const char* g_caif_layer_type_maxpool2d="maxpool2d";
    constexpr const char* g_caif_layer_type_flatten="flatten";
    constexpr const char* g_caif_layer_type_dropout="dropout";
    constexpr const char* g_caif_layer_type_batchnorm="batchnorm";
    constexpr const char* g_caif_layer_type_avgpool2d="avgpool2d";
    constexpr const char* g_caif_layer_type_unknown="unknown";
    
    // Layer config field names
    constexpr const char* g_caif_config_units="units";
    constexpr const char* g_caif_config_activation="activation";
    constexpr const char* g_caif_config_use_bias="use_bias";
    constexpr const char* g_caif_config_filters="filters";
    constexpr const char* g_caif_config_kernel_size="kernel_size";
    constexpr const char* g_caif_config_stride="stride";
    constexpr const char* g_caif_config_padding="padding";
    constexpr const char* g_caif_config_pool_size="pool_size";
    constexpr const char* g_caif_config_pool_size_height="pool_size_height";
    constexpr const char* g_caif_config_pool_size_width="pool_size_width";
    constexpr const char* g_caif_config_rate="rate";
    constexpr const char* g_caif_config_momentum="momentum";
    constexpr const char* g_caif_config_epsilon="epsilon";
    
    // Activation type names
    constexpr const char* g_caif_activation_relu="relu";
    constexpr const char* g_caif_activation_sigmoid="sigmoid";
    constexpr const char* g_caif_activation_tanh="tanh";
    constexpr const char* g_caif_activation_softmax="softmax";
    constexpr const char* g_caif_activation_linear="linear";
    constexpr const char* g_caif_activation_leakyrelu="leakyrelu";
    constexpr const char* g_caif_activation_elu="elu";
    constexpr const char* g_caif_activation_gelu="gelu";
    constexpr const char* g_caif_activation_swish="swish";
    constexpr const char* g_caif_activation_unknown="unknown";
    
    // Optimizer type names
    constexpr const char* g_caif_optimizer_sgd="sgd";
    constexpr const char* g_caif_optimizer_adam="adam";
    constexpr const char* g_caif_optimizer_rmsprop="rmsprop";
    constexpr const char* g_caif_optimizer_adagrad="adagrad";
    constexpr const char* g_caif_optimizer_unknown="unknown";
    
    // Loss type names
    constexpr const char* g_caif_loss_mse="mean_squared_error";
    constexpr const char* g_caif_loss_categorical_crossentropy="categorical_cross_entropy";
    constexpr const char* g_caif_loss_binary_crossentropy="binary_cross_entropy";
    constexpr const char* g_caif_loss_binary_crossentropy_logits="binary_cross_entropy_with_logits";
    constexpr const char* g_caif_loss_unknown="unknown";
    
    // Additional JSON field names
    constexpr const char* g_caif_field_optimizer_type="optimizer_type";
    constexpr const char* g_caif_field_loss_type="loss_type";
    constexpr const char* g_caif_field_training_iterations="training_iterations";
    constexpr const char* g_caif_field_metrics_history="metrics_history";
    constexpr const char* g_caif_field_model_version="model_version";
    constexpr const char* g_caif_field_optimizer_state="optimizer_state";
    
    // Feature names
    constexpr const char* g_caif_feature_dense_layers="dense_layers";
    constexpr const char* g_caif_feature_convolution_layers="convolution_layers";
    constexpr const char* g_caif_feature_pooling_layers="pooling_layers";
    constexpr const char* g_caif_feature_dropout_layers="dropout_layers";
    constexpr const char* g_caif_feature_batch_normalization="batch_normalization";
    constexpr const char* g_caif_feature_flatten_layers="flatten_layers";
    constexpr const char* g_caif_feature_weight_export="weight_export";
    constexpr const char* g_caif_feature_architecture_export="architecture_export";
    constexpr const char* g_caif_feature_optimizer_state_export="optimizer_state_export";
    constexpr const char* g_caif_feature_training_history_export="training_history_export";
    constexpr const char* g_caif_feature_model_versioning="model_versioning";
    
    // Error messages
    constexpr const char* g_caif_error_empty_filepath="Empty filepath provided";
    constexpr const char* g_caif_error_json_not_found="JSON architecture file not found: ";
    constexpr const char* g_caif_error_bin_not_found="Binary weights file not found: ";
    constexpr const char* g_caif_error_failed_open_json="Failed to open JSON file for reading: ";
    constexpr const char* g_caif_error_failed_open_bin="Failed to open binary file for reading: ";
    constexpr const char* g_caif_error_failed_write_json="Failed to open file for writing: ";
    constexpr const char* g_caif_error_failed_write_bin="Failed to open binary file for writing: ";
    constexpr const char* g_caif_error_failed_read_json="Failed to open JSON file for reading: ";
    constexpr const char* g_caif_error_missing_format_version="Missing format_version in JSON file";
    constexpr const char* g_caif_error_unsupported_format="Unsupported JSON format version: ";
    constexpr const char* g_caif_error_missing_input_shape="Missing input_shape in JSON file";
    constexpr const char* g_caif_error_layer_count_mismatch="Layer count mismatch: binary file has ";
    constexpr const char* g_caif_error_layers_but_network_has=" layers, but network has ";
    constexpr const char* g_caif_error_failed_to_get_layer="Failed to get layer at index ";
    constexpr const char* g_caif_error_failed_to_set_parameters="Failed to set parameters for layer ";
    constexpr const char* g_caif_error_failed_to_create_directory="Failed to create directory: ";
    constexpr const char* g_caif_error_exception_saving_architecture="Exception while saving architecture: ";
    constexpr const char* g_caif_error_exception_saving_weights="Exception while saving weights: ";
    constexpr const char* g_caif_error_exception_loading_architecture="Exception while loading architecture: ";
    constexpr const char* g_caif_error_exception_loading_weights="Exception while loading weights: ";
    
    // JSON parsing constants
    constexpr const char* g_caif_json_opening_brace="{";
    constexpr const char* g_caif_json_closing_brace="}";
    constexpr const char* g_caif_json_opening_bracket="[";
    constexpr const char* g_caif_json_closing_bracket="]";
    constexpr const char* g_caif_json_colon=":";
    constexpr const char* g_caif_json_comma=",";
    constexpr const char* g_caif_json_quote="\"";
    constexpr const char* g_caif_json_newline="\n";
    constexpr const char* g_caif_json_space=" ";
    constexpr const char* g_caif_json_true="true";
    constexpr const char* g_caif_json_false="false";
    constexpr const char* g_caif_json_null="null";
    
    // JSON field values for metrics parsing
    constexpr const char* g_caif_json_field_epoch="epoch";
    constexpr const char* g_caif_json_field_loss="loss";
    constexpr const char* g_caif_json_field_accuracy="accuracy";
    constexpr const char* g_caif_json_field_val_loss="val_loss";
    constexpr const char* g_caif_json_field_val_accuracy="val_accuracy";
    
    // JSON formatting constants
    constexpr const char* g_caif_json_indent_2="  ";
    constexpr const char* g_caif_json_indent_4="    ";
    constexpr const char* g_caif_json_indent_6="      ";
    constexpr const char* g_caif_json_indent_8="        ";
    
    // Warning messages
    constexpr const char* g_caif_warning_optimizer_mismatch="Warning: Optimizer type mismatch. Saved optimizer: ";
    constexpr const char* g_caif_warning_network_optimizer=", Network optimizer: ";
    constexpr const char* g_caif_warning_failed_restore_state="Warning: Failed to restore optimizer state: ";
  }//end CAIF_JSONBinaryConstants namespace

  // ONNX format constants
  namespace CAIF_ONNXConstants
  {
    // File extensions
    constexpr const char* g_caif_onnx_extension=".onnx";

    // Format information
    constexpr const char* g_caif_format_name="ONNX";

    // Error messages
    constexpr const char* g_caif_error_not_implemented="ONNX serialization not yet implemented";
  }//end CAIF_ONNXConstants namespace
}//end instance namespace
