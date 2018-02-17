/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "tc/core/halide2pencil.h"
#include "tc/core/mapping_options.h"
#include "tc/core/utils/dlpack.h"
#include "tc/lang/parser.h"

namespace tc {

/// These checkSizesAndStridesAreCompliant enforce runtime checks on tensor
/// metadata and throw lang::ErrorReport when checks fail.
/// @{
void checkSizesAndStridesAreCompliant(
    const DLTensor* actual,
    const DLTensor* expected,
    const lang::Param& dbg);

void checkSizesAndStridesAreCompliant(
    const std::vector<const DLTensor*>& dlTensors,
    const std::vector<dlutils::DLTensorUPtr>& tensorInfos,
    const lang::ListView<lang::Param>& dbgInfo);

void checkSizesAndStridesAreCompliant(
    const std::vector<DLTensor*>& dlTensors,
    const std::vector<dlutils::DLTensorUPtr>& tensorInfos,
    const lang::ListView<lang::Param>& dbgInfo);
/// @}

/// TcExecutor is an abstract class with target-agnostic support for running
/// the compile/run/uncheckedRun functions.
//
/// It stores a copy of the lang::TreeRef corresponding a single parsed TC
/// function. HalideComponents for the TC are created at construction time and
/// used throughout the execution to first compile and later check the tensor
/// sizes are compliant with expectations.
class TcExecutor {
 public:
  /// Target-agnostic state for executing a compiled kernel.
  /// TODO: break out CUDA-specific data.
  struct TcExecutionInfo {
   public:
    std::string kernelName;
    std::vector<dlutils::DLTensorUPtr> inputsInfo;
    std::vector<dlutils::DLTensorUPtr> outputsInfo;
    std::vector<int> kernelParams;
    std::string kernelSpecializedName;
    std::unique_ptr<tc::MappingOptions> options;
    std::string cudaSource;

    // TODO: move me
    Grid grid{{0, 0, 0}};
    Block block{{0, 0, 0}};
    std::shared_ptr<CudaRTCFunction> rtcFun;
  };

  /// @{ Construct and parse from TC string or already parsed TreeRef
  TcExecutor(
      const std::string& TcDefinition,
      const std::vector<const DLTensor*>& inputsInfo);

  TcExecutor(
      lang::TreeRef TcDefinition,
      const std::vector<const DLTensor*>& inputsInfo);
  /// @}

  /// TODO: should be renamed we do not use Pencil at all
  /// This is the pass that emits naive, readable C which can serve as the
  /// golden standard against which optimized code may be compared.
  HalidePencilState getHalidePencilState(
      const std::vector<const DLTensor*>& inTensorPtrs);

  /// Given a Tc and a list of input tensors that match the definition in the
  /// Tc in positional order, this generates the output tensor infos issued
  /// from forward inference.
  /// The typical flow is to infer output sizes, allocate/resize them within
  /// you favorite ML framework / tensor library and then call compile.
  std::vector<const DLTensor*> inferOutputTensorInfo();

  /// The following methods must be overriden for each backend.
  /// TODO: As we add backends we will want to factor out some of the
  /// implementation and avoid duplications.
  virtual void compile(const std::string& options) = 0;

  virtual Duration run(
      const std::vector<const DLTensor*>& inputs,
      const std::vector<DLTensor*>& outputs,
      bool profile = false) const = 0;

  virtual void uncheckedRun(
      const std::vector<const void*>& inputs,
      const std::vector<void*>& outputs) const = 0;

  // TODO: CUDA specific, should probably move elsewhere
  virtual bool hasRTCFun() const = 0;

 public:
  const static size_t InvalidHandle = std::numeric_limits<size_t>::max();

 protected:
  tc2halide::HalideComponents halideComponents_;
  TcExecutionInfo execInfo_;
  lang::TreeRef tcTree_;
  mutable isl::ctx ctx_;
};

} // namespace tc
