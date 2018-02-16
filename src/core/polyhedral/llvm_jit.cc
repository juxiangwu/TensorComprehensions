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
#include <stdexcept>

#include "tc/core/polyhedral/llvm_jit.h"

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/Orc/LambdaResolver.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/DynamicLibrary.h"

#include "tc/core/flags.h"
#include "tc/core/polyhedral/codegen_llvm.h"

using namespace llvm;

// Parse through ldconfig to find the path of a particular
// shared library. This is an unfortunate way to have to
// find it, but I couldn't immediately find something in
// imported libraries that would resolve this for us.
std::string find_library_path(std::string library) {
  std::string command = "ldconfig -p | grep " + library;

  FILE* fpipe = popen(command.c_str(), "r");

  if (fpipe == nullptr) {
    throw std::runtime_error("Failed to popen()");
  }

  std::string output;
  char buffer[512];

  while (1) {
    int charactersRead = fread(buffer, 1, sizeof(buffer), fpipe);
    if (charactersRead == 0)
      break;
    output += std::string(buffer, charactersRead);
  }
  pclose(fpipe);

  int idx = output.rfind("=> ");
  if (idx == std::string::npos) {
    throw std::runtime_error("Failed locate library: "+library);
  }
  output = output.substr(idx + 3);
  if (output.length() > 0 && output[output.length() - 1] == '\n') {
    output = output.substr(0, output.length() - 1);
  }
  return output;
}

namespace tc {

Jit::Jit()
    : TM_(EngineBuilder().selectTarget()),
      DL_(TM_->createDataLayout()),
      objectLayer_([]() { return std::make_shared<SectionMemoryManager>(); }),
      compileLayer_(objectLayer_, orc::SimpleCompiler(*TM_)) {
  std::string err;

  auto path = find_library_path("libcilkrts.so");
  sys::DynamicLibrary::LoadLibraryPermanently(path.c_str(), &err);
  if (err != "") {
    throw std::runtime_error("Failed to find cilkrts: " + err);
  }
}

void Jit::codegenScop(
    const std::string& specializedName,
    const polyhedral::Scop& scop) {
  addModule(emitLLVMKernel(
      specializedName, scop, getTargetMachine().createDataLayout()));
}

TargetMachine& Jit::getTargetMachine() {
  return *TM_;
}

Jit::ModuleHandle Jit::addModule(std::unique_ptr<Module> M) {
  M->setTargetTriple(TM_->getTargetTriple().str());
  auto Resolver = orc::createLambdaResolver(
      [&](const std::string& Name) {
        if (auto Sym = compileLayer_.findSymbol(Name, false))
          return Sym;
        return JITSymbol(nullptr);
      },
      [](const std::string& Name) {
        if (auto SymAddr = RTDyldMemoryManager::getSymbolAddressInProcess(Name))
          return JITSymbol(SymAddr, JITSymbolFlags::Exported);
        return JITSymbol(nullptr);
      });

  auto res = compileLayer_.addModule(std::move(M), std::move(Resolver));
  CHECK(res) << "Failed to jit compile.";
  return *res;
}

JITSymbol Jit::findSymbol(const std::string Name) {
  std::string MangledName;
  raw_string_ostream MangledNameStream(MangledName);
  Mangler::getNameWithPrefix(MangledNameStream, Name, DL_);
  return compileLayer_.findSymbol(MangledNameStream.str(), true);
}

JITTargetAddress Jit::getSymbolAddress(const std::string Name) {
  auto res = findSymbol(Name).getAddress();
  CHECK(res) << "Could not find jit-ed symbol";
  return *res;
}

llvm::GenericValue Jit::runFunction(const std::string fn_name,
                                    llvm::ArrayRef<llvm::GenericValue> ArgValues) {
   void *FPtr = (void*)getSymbolAddress(fn_name);
   assert(FPtr && "Pointer to fn's code was null after getPointerToFunction");
   FunctionType *FTy = F->getFunctionType();
   Type *RetTy = FTy->getReturnType();

   assert((FTy->getNumParams() == ArgValues.size() ||
           (FTy->isVarArg() && FTy->getNumParams() <= ArgValues.size())) &&
          "Wrong number of arguments passed into function!");
   assert(FTy->getNumParams() == ArgValues.size() &&
          "This doesn't support passing arguments through varargs (yet)!");

   // Handle some common cases first.  These cases correspond to common `main'
   // prototypes.
   if (RetTy->isIntegerTy(32) || RetTy->isVoidTy()) {
     switch (ArgValues.size()) {
     case 3:
       if (FTy->getParamType(0)->isIntegerTy(32) &&
           FTy->getParamType(1)->isPointerTy() &&
           FTy->getParamType(2)->isPointerTy()) {
         int (*PF)(int, char **, const char **) =
             (int (*)(int, char **, const char **))(intptr_t)FPtr;

         // Call the function.
         GenericValue rv;
         rv.IntVal = APInt(32, PF(ArgValues[0].IntVal.getZExtValue(),
                                  (char **)GVTOP(ArgValues[1]),
                                  (const char **)GVTOP(ArgValues[2])));
         return rv;
       }
       break;
     case 2:
       if (FTy->getParamType(0)->isIntegerTy(32) &&
           FTy->getParamType(1)->isPointerTy()) {
         int (*PF)(int, char **) = (int (*)(int, char **))(intptr_t)FPtr;

         // Call the function.
         GenericValue rv;
         rv.IntVal = APInt(32, PF(ArgValues[0].IntVal.getZExtValue(),
                                  (char **)GVTOP(ArgValues[1])));
         return rv;
       }
       break;
     case 1:
       if (FTy->getNumParams() == 1 && FTy->getParamType(0)->isIntegerTy(32)) {
         GenericValue rv;
         int (*PF)(int) = (int (*)(int))(intptr_t)FPtr;
         rv.IntVal = APInt(32, PF(ArgValues[0].IntVal.getZExtValue()));
         return rv;
       }
       break;
     }
   }

   // Handle cases where no arguments are passed first.
   if (ArgValues.empty()) {
     GenericValue rv;
     switch (RetTy->getTypeID()) {
     default:
       llvm_unreachable("Unknown return type for function call!");
     case Type::IntegerTyID: {
       unsigned BitWidth = cast<IntegerType>(RetTy)->getBitWidth();
       if (BitWidth == 1)
         rv.IntVal = APInt(BitWidth, ((bool (*)())(intptr_t)FPtr)());
       else if (BitWidth <= 8)
         rv.IntVal = APInt(BitWidth, ((char (*)())(intptr_t)FPtr)());
       else if (BitWidth <= 16)
         rv.IntVal = APInt(BitWidth, ((short (*)())(intptr_t)FPtr)());
       else if (BitWidth <= 32)
         rv.IntVal = APInt(BitWidth, ((int (*)())(intptr_t)FPtr)());
       else if (BitWidth <= 64)
         rv.IntVal = APInt(BitWidth, ((int64_t (*)())(intptr_t)FPtr)());
       else
         llvm_unreachable("Integer types > 64 bits not supported");
       return rv;
     }
     case Type::VoidTyID:
       rv.IntVal = APInt(32, ((int (*)())(intptr_t)FPtr)());
       return rv;
     case Type::FloatTyID:
       rv.FloatVal = ((float (*)())(intptr_t)FPtr)();
       return rv;
     case Type::DoubleTyID:
       rv.DoubleVal = ((double (*)())(intptr_t)FPtr)();
       return rv;
     case Type::X86_FP80TyID:
     case Type::FP128TyID:
     case Type::PPC_FP128TyID:
       llvm_unreachable("long double not supported yet");
     case Type::PointerTyID:
       return PTOGV(((void *(*)())(intptr_t)FPtr)());
     }
   }

   llvm_unreachable("Full-featured argument passing not supported yet!");
 }

} // namespace tc
