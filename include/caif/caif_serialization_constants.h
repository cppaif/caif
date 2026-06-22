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
 * @file caif_serialization_constants.h
 * @brief Constants for model serialization in CAIF framework. Every
 * literal that crosses a SafeTensors / JSON+Binary / ONNX boundary or
 * gets concatenated into safetensors metadata layer_descriptions lives
 * here. The g_serial_ prefix identifies the file; no nested namespaces.
 */

#pragma once

#include <cstdint>
#include <string>

namespace instance
{

//------------------------------------------------------------------------------
// JSON+Binary format constants (legacy CAIF native model format)
//------------------------------------------------------------------------------

inline const std::string g_serial_json_extension=".json";
inline const std::string g_serial_binary_extension=".bin";

inline const std::string g_serial_jsonbin_format_name="JSON+Binary";
constexpr uint32_t g_serial_jsonbin_format_version=1;

inline const std::string g_serial_field_format_version="format_version";
inline const std::string g_serial_field_is_compiled="is_compiled";
inline const std::string g_serial_field_is_trained="is_trained";
inline const std::string g_serial_field_input_shape="input_shape";
inline const std::string g_serial_field_output_shape="output_shape";
inline const std::string g_serial_field_learning_rate="learning_rate";
inline const std::string g_serial_field_layers="layers";
inline const std::string g_serial_field_layer_index="layer_index";
inline const std::string g_serial_field_layer_type="layer_type";
inline const std::string g_serial_field_config="config";

inline const std::string g_serial_layer_type_embedding="embedding";
inline const std::string g_serial_layer_type_dense="dense";
inline const std::string g_serial_layer_type_conv2d="conv2d";
inline const std::string g_serial_layer_type_maxpool2d="maxpool2d";
inline const std::string g_serial_layer_type_avgpool2d="avgpool2d";
inline const std::string g_serial_layer_type_batchnorm="batchnorm";
inline const std::string g_serial_layer_type_dropout="dropout";
inline const std::string g_serial_layer_type_flatten="flatten";
inline const std::string g_serial_layer_type_reshape="reshape";
inline const std::string g_serial_layer_type_multi_head_attention="multi_head_attention";
inline const std::string g_serial_layer_type_transformer_encoder="transformer_encoder";
inline const std::string g_serial_layer_type_unknown="unknown";

inline const std::string g_serial_config_units="units";
inline const std::string g_serial_config_activation="activation";
inline const std::string g_serial_config_use_bias="use_bias";
inline const std::string g_serial_config_filters="filters";
inline const std::string g_serial_config_kernel_size="kernel_size";
inline const std::string g_serial_config_stride="stride";
inline const std::string g_serial_config_padding="padding";
inline const std::string g_serial_config_pool_size="pool_size";
inline const std::string g_serial_config_pool_size_height="pool_size_height";
inline const std::string g_serial_config_pool_size_width="pool_size_width";
inline const std::string g_serial_config_rate="rate";
inline const std::string g_serial_config_momentum="momentum";
inline const std::string g_serial_config_epsilon="epsilon";

inline const std::string g_serial_activation_relu="relu";
inline const std::string g_serial_activation_sigmoid="sigmoid";
inline const std::string g_serial_activation_tanh="tanh";
inline const std::string g_serial_activation_softmax="softmax";
inline const std::string g_serial_activation_linear="linear";
inline const std::string g_serial_activation_leakyrelu="leakyrelu";
inline const std::string g_serial_activation_elu="elu";
inline const std::string g_serial_activation_gelu="gelu";
inline const std::string g_serial_activation_swish="swish";
inline const std::string g_serial_activation_unknown="unknown";

// GELU approximation variant names (CAIF_GELUApproximation vocabulary).
inline const std::string g_serial_gelu_approx_tanh="tanh";
inline const std::string g_serial_gelu_approx_exact="exact";

inline const std::string g_serial_optimizer_sgd="sgd";
inline const std::string g_serial_optimizer_adam="adam";
inline const std::string g_serial_optimizer_rmsprop="rmsprop";
inline const std::string g_serial_optimizer_adagrad="adagrad";
inline const std::string g_serial_optimizer_momentum="momentum";
inline const std::string g_serial_optimizer_unknown="unknown";

inline const std::string g_serial_loss_mse="mean_squared_error";
inline const std::string g_serial_loss_cross_entropy="cross_entropy";
inline const std::string g_serial_loss_categorical_crossentropy="categorical_cross_entropy";
inline const std::string g_serial_loss_binary_crossentropy="binary_cross_entropy";
inline const std::string g_serial_loss_binary_crossentropy_logits="binary_cross_entropy_with_logits";
inline const std::string g_serial_loss_huber="huber";
inline const std::string g_serial_loss_mae="mean_absolute_error";
inline const std::string g_serial_loss_unknown="unknown";

inline const std::string g_serial_field_optimizer_type="optimizer_type";
inline const std::string g_serial_field_loss_type="loss_type";
inline const std::string g_serial_field_training_iterations="training_iterations";
inline const std::string g_serial_field_metrics_history="metrics_history";
inline const std::string g_serial_field_model_version="model_version";
inline const std::string g_serial_field_optimizer_state="optimizer_state";

inline const std::string g_serial_feature_dense_layers="dense_layers";
inline const std::string g_serial_feature_convolution_layers="convolution_layers";
inline const std::string g_serial_feature_pooling_layers="pooling_layers";
inline const std::string g_serial_feature_dropout_layers="dropout_layers";
inline const std::string g_serial_feature_batch_normalization="batch_normalization";
inline const std::string g_serial_feature_flatten_layers="flatten_layers";
inline const std::string g_serial_feature_weight_export="weight_export";
inline const std::string g_serial_feature_architecture_export="architecture_export";
inline const std::string g_serial_feature_optimizer_state_export="optimizer_state_export";
inline const std::string g_serial_feature_training_history_export="training_history_export";
inline const std::string g_serial_feature_model_versioning="model_versioning";

inline const std::string g_serial_error_empty_filepath="Empty filepath provided";
inline const std::string g_serial_error_json_not_found="JSON architecture file not found: ";
inline const std::string g_serial_error_bin_not_found="Binary weights file not found: ";
inline const std::string g_serial_error_failed_open_json="Failed to open JSON file for reading: ";
inline const std::string g_serial_error_failed_open_bin="Failed to open binary file for reading: ";
inline const std::string g_serial_error_failed_write_json="Failed to open file for writing: ";
inline const std::string g_serial_error_failed_write_bin="Failed to open binary file for writing: ";
inline const std::string g_serial_error_failed_read_json="Failed to open JSON file for reading: ";
inline const std::string g_serial_error_missing_format_version="Missing format_version in JSON file";
inline const std::string g_serial_error_unsupported_format="Unsupported JSON format version: ";
inline const std::string g_serial_error_missing_input_shape="Missing input_shape in JSON file";
inline const std::string g_serial_error_layer_count_mismatch="Layer count mismatch: binary file has ";
inline const std::string g_serial_error_layers_but_network_has=" layers, but network has ";
inline const std::string g_serial_error_failed_to_get_layer="Failed to get layer at index ";
inline const std::string g_serial_error_failed_to_set_parameters="Failed to set parameters for layer ";
inline const std::string g_serial_error_failed_to_create_directory="Failed to create directory: ";
inline const std::string g_serial_error_exception_saving_architecture="Exception while saving architecture: ";
inline const std::string g_serial_error_exception_saving_weights="Exception while saving weights: ";
inline const std::string g_serial_error_exception_loading_architecture="Exception while loading architecture: ";
inline const std::string g_serial_error_exception_loading_weights="Exception while loading weights: ";

inline const std::string g_serial_json_opening_brace="{";
inline const std::string g_serial_json_closing_brace="}";
inline const std::string g_serial_json_opening_bracket="[";
inline const std::string g_serial_json_closing_bracket="]";
inline const std::string g_serial_json_colon=":";
inline const std::string g_serial_json_comma=",";
inline const std::string g_serial_json_quote="\"";
inline const std::string g_serial_json_newline="\n";
inline const std::string g_serial_json_space=" ";
inline const std::string g_serial_json_true="true";
inline const std::string g_serial_json_false="false";
inline const std::string g_serial_json_null="null";

inline const std::string g_serial_json_field_epoch="epoch";
inline const std::string g_serial_json_field_loss="loss";
inline const std::string g_serial_json_field_accuracy="accuracy";
inline const std::string g_serial_json_field_val_loss="val_loss";
inline const std::string g_serial_json_field_val_accuracy="val_accuracy";

inline const std::string g_serial_json_indent_2="  ";
inline const std::string g_serial_json_indent_4="    ";
inline const std::string g_serial_json_indent_6="      ";
inline const std::string g_serial_json_indent_8="        ";

inline const std::string g_serial_warning_optimizer_mismatch="Warning: Optimizer type mismatch. Saved optimizer: ";
inline const std::string g_serial_warning_network_optimizer=", Network optimizer: ";
inline const std::string g_serial_warning_failed_restore_state="Warning: Failed to restore optimizer state: ";

//------------------------------------------------------------------------------
// ONNX format constants
//------------------------------------------------------------------------------

inline const std::string g_serial_onnx_extension=".onnx";
inline const std::string g_serial_onnx_format_name="ONNX";
inline const std::string g_serial_onnx_error_not_implemented="ONNX serialization not yet implemented";

//------------------------------------------------------------------------------
// SafeTensors format constants — filenames, header JSON keys, dtype names,
// and the CAIF-specific metadata payload keys written into __metadata__.
//------------------------------------------------------------------------------

inline const std::string g_serial_single_shard_name="model.safetensors";
inline const std::string g_serial_single_shard_relative="/model.safetensors";
inline const std::string g_serial_index_relative="/model.safetensors.index.json";
inline const std::string g_serial_dir_separator="/";
inline const std::string g_serial_format_name="SafeTensors";
inline const std::string g_serial_extension=".safetensors";

inline const std::string g_serial_key_dtype="dtype";
inline const std::string g_serial_key_shape="shape";
inline const std::string g_serial_key_data_offsets="data_offsets";
inline const std::string g_serial_key_metadata_outer="__metadata__";
inline const std::string g_serial_key_metadata_inner="metadata";
inline const std::string g_serial_key_weight_map="weight_map";

inline const std::string g_serial_meta_key_format="format";
inline const std::string g_serial_meta_key_layer_count="layer_count";
inline const std::string g_serial_meta_key_layer_descriptions="layer_descriptions";
inline const std::string g_serial_meta_format_value="aif_device_network";

inline const std::string g_serial_dtype_bool="BOOL";
inline const std::string g_serial_dtype_f64="F64";
inline const std::string g_serial_dtype_f32="F32";
inline const std::string g_serial_dtype_f16="F16";
inline const std::string g_serial_dtype_bf16="BF16";
inline const std::string g_serial_dtype_i64="I64";
inline const std::string g_serial_dtype_i32="I32";
inline const std::string g_serial_dtype_i16="I16";
inline const std::string g_serial_dtype_i8="I8";
inline const std::string g_serial_dtype_i4="I4";
inline const std::string g_serial_dtype_u64="U64";
inline const std::string g_serial_dtype_u32="U32";
inline const std::string g_serial_dtype_u16="U16";
inline const std::string g_serial_dtype_u8="U8";

// CAIF-internal lowercase dtype display names. Distinct vocabulary
// from the uppercase safetensors form above; used by CAIF_DataType::Name()
// for human-readable diagnostics and concatenated into safetensors
// layer_descriptions via Description() (e.g. FrozenLinear).
inline const std::string g_serial_dtype_name_fp32="fp32";
inline const std::string g_serial_dtype_name_fp64="fp64";
inline const std::string g_serial_dtype_name_fp16="fp16";
inline const std::string g_serial_dtype_name_bf16="bf16";
inline const std::string g_serial_dtype_name_int4="int4";
inline const std::string g_serial_dtype_name_int8="int8";
inline const std::string g_serial_dtype_name_int16="int16";
inline const std::string g_serial_dtype_name_int32="int32";
inline const std::string g_serial_dtype_name_int64="int64";
inline const std::string g_serial_dtype_name_uint8="uint8";
inline const std::string g_serial_dtype_name_uint16="uint16";
inline const std::string g_serial_dtype_name_uint32="uint32";
inline const std::string g_serial_dtype_name_uint64="uint64";
inline const std::string g_serial_dtype_name_bool="bool";
inline const std::string g_serial_dtype_name_unknown="unknown";

//------------------------------------------------------------------------------
// Description() fragments. Layer Description() output is concatenated with
// `;` into the safetensors metadata layer_descriptions field, so every
// fragment that appears in a Description() override must come from this
// block — external consumers parse exact strings.
//------------------------------------------------------------------------------

inline const std::string g_serial_open_paren="(";
inline const std::string g_serial_close_paren=")";
inline const std::string g_serial_comma=",";

inline const std::string g_serial_kv_dim="dim=";
inline const std::string g_serial_kv_ffn_dim="ffn_dim=";
inline const std::string g_serial_kv_hidden_dim="hidden_dim=";
inline const std::string g_serial_kv_heads="heads=";
inline const std::string g_serial_kv_head_dim="head_dim=";
inline const std::string g_serial_kv_kv_heads="kv_heads=";
inline const std::string g_serial_kv_num_heads="num_heads=";
inline const std::string g_serial_kv_layers="layers=";
inline const std::string g_serial_kv_vocab="vocab=";
inline const std::string g_serial_kv_max_seq="max_seq=";
inline const std::string g_serial_kv_mode="mode=";
inline const std::string g_serial_kv_activation="activation=";
inline const std::string g_serial_kv_causal="causal=";
inline const std::string g_serial_kv_rope="rope=";
inline const std::string g_serial_kv_q_lora="q_lora=";
inline const std::string g_serial_kv_kv_lora="kv_lora=";
inline const std::string g_serial_kv_nope="nope=";
inline const std::string g_serial_kv_bidir="bidir=";
inline const std::string g_serial_kv_buckets="buckets=";
inline const std::string g_serial_kv_max_dist="max_dist=";
inline const std::string g_serial_kv_top_k="top_k=";
inline const std::string g_serial_kv_experts="experts=";
inline const std::string g_serial_kv_shared="shared=";
inline const std::string g_serial_kv_noise="noise=";
inline const std::string g_serial_kv_cap="cap=";
inline const std::string g_serial_kv_alpha="alpha=";
inline const std::string g_serial_kv_rate="rate=";
inline const std::string g_serial_kv_classes="classes=";
inline const std::string g_serial_kv_ch="ch=";
inline const std::string g_serial_kv_patch="patch=";
inline const std::string g_serial_kv_patches="patches=";
inline const std::string g_serial_kv_img="img=";
inline const std::string g_serial_kv_features="features=";
inline const std::string g_serial_kv_freq_bins="freq_bins=";
inline const std::string g_serial_kv_rank="rank=";
inline const std::string g_serial_kv_in="in=";
inline const std::string g_serial_kv_out="out=";
inline const std::string g_serial_kv_v="v=";
inline const std::string g_serial_kv_dtype="dtype=";
inline const std::string g_serial_kv_use_gated="use_gated=";
inline const std::string g_serial_kv_stages="stages=";
inline const std::string g_serial_kv_params="params=";
inline const std::string g_serial_kv_target="target=";
inline const std::string g_serial_kv_loaded="loaded=";

inline const std::string g_serial_flag_projections=",projections";
inline const std::string g_serial_flag_direct_q=",direct_q";
inline const std::string g_serial_flag_cls_true=",cls=true";
inline const std::string g_serial_flag_rope_true=",rope=true";
inline const std::string g_serial_flag_tied_true=",tied=true";
inline const std::string g_serial_flag_frozen_true=",frozen=true";
inline const std::string g_serial_flag_gated=",gated";
inline const std::string g_serial_flag_bias=",bias";
inline const std::string g_serial_flag_qk_norm="QK-norm";

inline const std::string g_serial_suffix_sublayers=" sublayers)";
inline const std::string g_serial_suffix_layers=" layers)";

// LoRA wraps another layer; LoRA Description() is "LoRA(...)+<base>".
inline const std::string g_serial_lora_plus="+";

// ViT Description() prints image dimensions as <H>x<W>.
inline const std::string g_serial_dim_separator="x";

inline const std::string g_serial_tag_ffn="FFN";
inline const std::string g_serial_tag_mla="MLA";
inline const std::string g_serial_tag_multi_head_attention="MultiHeadAttention";
inline const std::string g_serial_tag_cross_attention="CrossAttention";
inline const std::string g_serial_tag_t5_attention="T5Attention";
inline const std::string g_serial_tag_linear_head="LinearHead";
inline const std::string g_serial_tag_layernorm="LayerNorm";
inline const std::string g_serial_tag_rmsnorm="RMSNorm";
inline const std::string g_serial_tag_patch_embedding="PatchEmbedding";
inline const std::string g_serial_tag_tabular_embedding="TabularEmbedding";
inline const std::string g_serial_tag_spectrogram_embedding="SpectrogramEmbedding";
inline const std::string g_serial_tag_positional_encoding="PositionalEncoding";
inline const std::string g_serial_tag_token_embedding="TokenEmbedding";
inline const std::string g_serial_tag_relative_position_bias="RelativePositionBias";
inline const std::string g_serial_tag_lora="LoRA";
inline const std::string g_serial_tag_dense="Dense";
inline const std::string g_serial_tag_dropout="CAIF_DeviceDropout";
inline const std::string g_serial_tag_network="Network";
inline const std::string g_serial_tag_container="Container";
inline const std::string g_serial_tag_pre_norm_block="PreNormBlock";
inline const std::string g_serial_tag_transformer_block="TransformerBlock";
inline const std::string g_serial_tag_transformer_model="TransformerModel";
inline const std::string g_serial_tag_vit="ViT";
inline const std::string g_serial_tag_frozen_linear="FrozenLinear";
inline const std::string g_serial_tag_relu="ReLU";
inline const std::string g_serial_tag_gelu="GELU";
inline const std::string g_serial_tag_gelu_exact="GELUExact";
inline const std::string g_serial_tag_sigmoid="Sigmoid";
inline const std::string g_serial_tag_tanh="Tanh";
inline const std::string g_serial_tag_swish="Swish";
inline const std::string g_serial_tag_leaky_relu="LeakyReLU";
inline const std::string g_serial_tag_elu="ELU";
inline const std::string g_serial_tag_linear="Linear";
inline const std::string g_serial_tag_silu="SiLU";
inline const std::string g_serial_tag_flatten="CAIF_DeviceFlatten";
inline const std::string g_serial_tag_reshape="CAIF_DeviceReshape";
inline const std::string g_serial_tag_average_pooling2d="CAIF_DeviceAveragePooling2D";
inline const std::string g_serial_tag_max_pooling2d="CAIF_DeviceMaxPooling2D";
inline const std::string g_serial_tag_conv2d="CAIF_DeviceConv2D";
inline const std::string g_serial_tag_batch_norm="CAIF_DeviceBatchNorm";

inline const std::string g_serial_gated_swiglu="SwiGLU";
inline const std::string g_serial_gated_geglu="GeGLU";
inline const std::string g_serial_gated_reglu="ReGLU";
inline const std::string g_serial_gated_glu="GLU";
inline const std::string g_serial_gated_bilinear="Bilinear";

inline const std::string g_serial_pe_mode_learned="learned";
inline const std::string g_serial_pe_mode_sinusoidal="sinusoidal";
inline const std::string g_serial_pe_mode_none="none";

}//end instance namespace
