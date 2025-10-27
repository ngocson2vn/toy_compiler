#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "llvm/Support/CommandLine.h"

#include "common/status.h"
#include "frontend/frontend.h"
#include "middleend/middleend.h"
#include "backend/backend.h"

namespace cl = llvm::cl;
using namespace toy::compiler;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

static cl::opt<std::string> targetArch(cl::Positional,
                                          cl::desc("<target sm arch>"),
                                          cl::init("sm_75"),
                                          cl::value_desc("sm arch"));

int main(int argc, char **argv) {
  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");

  mlir::MLIRContext context;
  auto mref = frontend::getModule(context, inputFilename);
  mlir::ModuleOp mod = mref.get();

  llvm::outs() << "\nOriginal MLIR module:\n";
  mod.dump();

  auto meStatus = middleend::lower(mod);
  if (meStatus.failed()) {
    llvm::errs() << "[middleend] failed to lower mlir::ModuleOp\n";
    return 1;
  }

  llvm::outs() << "\nAfter middleend MLIR module:\n";
  mod.dump();

  llvm::outs() << "\nTarget SM arch: " << targetArch.getValue() << "\n";
  auto beStatus = backend::lower(mod, targetArch.getValue());
  if (beStatus.failed()) {
    llvm::errs() << "[backend] failed to lower mlir::ModuleOp\n";
    return 1;
  }

}
