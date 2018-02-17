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
#include "tc/core/tc_executor.h"
#include "tc/core/halide2pencil.h"
#include "tc/core/mapping_options.h"
#include "tc/core/utils/dlpack.h"
#include "tc/lang/parser.h"
#include "tc/lang/sema.h"

using tc::dlutils::DLTensorUPtr;

namespace tc {
const size_t TcExecutor::InvalidHandle;

namespace {
lang::TreeRef parseOneFunction(const std::string& def) {
  lang::Parser parser(def);
  auto r = parser.parseFunction();
  if (parser.L.cur().kind != lang::TK_EOF) {
    throw lang::ErrorReport(parser.L.cur().range)
        << "More than one TCs were passed to TcExecutor.";
  }
  return r;
}

int toTypeToken(DLDataType dtype) {
  return lang::TypeInfo(lang::TypeInfo::Code(dtype.code), dtype.bits)
      .toScalarToken();
}
} // namespace

// TODO: make sure that the empty stride arrays (in DLTensor) are not a problem
void checkSizesAndStridesAreCompliant(
    const DLTensor* actual,
    const DLTensor* expected,
    const lang::Param& dbg) {
  if (actual->ndim != expected->ndim) {
    throw lang::ErrorReport(dbg)
        << "expected " << expected->ndim << " dimensions but found tensor with "
        << actual->ndim << " dimensions";
  }
  auto atype = toTypeToken(actual->dtype);
  auto etype = toTypeToken(expected->dtype);
  if (atype != etype) {
    throw lang::ErrorReport(dbg) << "expected " << lang::kindToString(etype)
                                 << " but found " << lang::kindToString(atype);
  }
  std::vector<int64_t> shapeA(actual->shape, actual->shape + actual->ndim);
  std::vector<int64_t> shapeE(
      expected->shape, expected->shape + expected->ndim);
  for (int i = 0; i < shapeA.size(); ++i) {
    if (shapeA[i] != shapeE[i]) {
      throw lang::ErrorReport(dbg)
          << "expected size " << shapeE[i] << " for dim " << i << " but found "
          << shapeA[i];
    }
  }
}

void checkSizesAndStridesAreCompliant(
    const std::vector<DLTensor*>& dlTensors,
    const std::vector<dlutils::DLTensorUPtr>& tensorInfos,
    const lang::ListView<lang::Param>& dbgInfo) {
  if (tensorInfos.size() != dlTensors.size()) {
    throw lang::ErrorReport(dbgInfo)
        << "expected " << tensorInfos.size() << " values but found "
        << dlTensors.size();
  }
  for (size_t i = 0; i < tensorInfos.size(); ++i) {
    checkSizesAndStridesAreCompliant(
        dlTensors[i], tensorInfos[i].get(), dbgInfo[i]);
  }
}

void checkSizesAndStridesAreCompliant(
    const std::vector<const DLTensor*>& dlTensors,
    const std::vector<dlutils::DLTensorUPtr>& tensorInfos,
    const lang::ListView<lang::Param>& dbgInfo) {
  std::vector<DLTensor*> unconstDlTensors;
  unconstDlTensors.reserve(dlTensors.size());
  for (auto t : dlTensors) {
    unconstDlTensors.push_back(const_cast<DLTensor*>(t));
  }
  checkSizesAndStridesAreCompliant(unconstDlTensors, tensorInfos, dbgInfo);
}

namespace {
void checkInputsCompliant(
    const std::vector<const DLTensor*>& inputsInfo,
    const tc2halide::HalideComponents& halideComponents) {
  if (inputsInfo.size() != halideComponents.inputs.size()) {
    throw lang::ErrorReport(halideComponents.getDef())
        << "expected " << halideComponents.inputs.size() << " inputs but found "
        << inputsInfo.size();
  }
  for (size_t i = 0; i < inputsInfo.size(); ++i) {
    auto dltype_ = inputsInfo[i]->dtype;
    auto htype_ = halideComponents.inputs[i].type();
    // we have three type representations here: (1) halide Type (2) DLTensor
    // type, and (3) the token representing the type in the frontend (e.g.
    // TK_FLOAT) we need to translate to (3) to report user facing errors
    auto dltype =
        lang::TypeInfo(lang::TypeInfo::Code(dltype_.code), dltype_.bits)
            .toScalarToken();
    auto htype =
        lang::TypeInfo(lang::TypeInfo::Code(htype_.code()), htype_.bits())
            .toScalarToken();
    if (dltype != htype) {
      throw lang::ErrorReport(halideComponents.getDef().params()[i])
          << "expected type " << lang::kindToString(htype) << " but found "
          << lang::kindToString(dltype);
    }
    int edim = halideComponents.inputs[i].dimensions();
    int adim = inputsInfo[i]->ndim;
    if (adim != edim) {
      throw lang::ErrorReport(halideComponents.getDef().params()[i])
          << "expected a tensor with " << edim << " dimensions but found "
          << adim << " dimensions.";
    }
  }
}
} // namespace

TcExecutor::TcExecutor(
    const std::string& tcDefinition,
    const std::vector<const DLTensor*>& inputsInfo)
    : TcExecutor(parseOneFunction(tcDefinition), inputsInfo) {}

TcExecutor::TcExecutor(
    lang::TreeRef tcDefinition,
    const std::vector<const DLTensor*>& inputsInfo)
    : tcTree_(tcDefinition), ctx_(isl_ctx_alloc()) {
  // TcExecutor(tcDefinition, isl_ctx_alloc()) {
  execInfo_.kernelName = lang::Def(tcTree_).name().name();
  halideComponents_ = tc2halide::translate(ctx_, tcTree_);
  checkInputsCompliant(inputsInfo, halideComponents_);
  execInfo_.inputsInfo = dlutils::makeDLTensorVector(inputsInfo);
  execInfo_.outputsInfo = getHalidePencilState(inputsInfo).outputsDLT;
}

/// TODO: should be renamed we do not use Pencil at all
HalidePencilState TcExecutor::getHalidePencilState(
    const std::vector<const DLTensor*>& inTensorPtrs) {
  // TODO: check if this is wrong, packed tensors may  have 0 strides stored
  auto halidePencilState = toPencil(
      halideComponents_,
      inTensorPtrs,
      // if execInfo_.options is nullptr then just don't specialize the code
      (execInfo_.options
           ? execInfo_.options->proto.fix_parameters_before_scheduling()
           : false),
      execInfo_.kernelName);
  return halidePencilState;
}

std::vector<const DLTensor*> TcExecutor::inferOutputTensorInfo() {
  return extractRawPtrs(execInfo_.outputsInfo);
}

} // namespace tc
