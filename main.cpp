#include <filesystem>

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"

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
                                          cl::init("-"),
                                          cl::value_desc("sm arch"));

static cl::opt<bool> skipMiddend(cl::Prefix, "skip-middend",
                                          cl::desc("<whether to skip middleend or not>"),
                                          cl::init(false),
                                          cl::value_desc("default to false"));


static cl::opt<bool> skipBackend(cl::Prefix, "skip-backend",
                                          cl::desc("<whether to skip backend or not>"),
                                          cl::init(false),
                                          cl::value_desc("default to false"));

static cl::opt<bool> verbose(cl::Prefix, "verbose",
                                          cl::desc("<whether to verbose or not>"),
                                          cl::init(false),
                                          cl::value_desc("default to false"));
static cl::alias verboseShort("v",
                              cl::desc("Alias for --verbose"),
                              cl::aliasopt(verbose));

int main(int argc, char **argv) {
  // Get binary dir
  std::filesystem::path binPath(argv[0]);
  auto libRuntimePath = binPath.parent_path().parent_path().string();
  libRuntimePath.append("/lib/libruntime.a");

  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");

  mlir::MLIRContext context;

  // 1. frontend
  auto mref = frontend::getModule(context, inputFilename, verbose.getValue());
  mlir::ModuleOp mod = mref.get();

  if (verbose.getValue()) {
    llvm::outs() << "\nOriginal MLIR module:\n";
    mod.dump();
  }


  // 2. middleend
  if (!!skipMiddend) return 0;
  auto meStatus = middleend::lower(mod);
  if (meStatus.failed()) {
    llvm::errs() << "[middleend] failed to lower mlir::ModuleOp\n";
    return 1;
  }

  if (verbose.getValue()) {
    llvm::outs() << "\nAfter middleend MLIR module:\n";
    mod.dump();
  }


  // 3. backend
  if (!!skipBackend) return 0;
  auto beStatus = backend::lower(mod, targetArch.getValue(), libRuntimePath);
  if (beStatus.failed()) {
    llvm::errs() << "[backend] failed to lower mlir::ModuleOp\n";
    return 1;
  }

}
