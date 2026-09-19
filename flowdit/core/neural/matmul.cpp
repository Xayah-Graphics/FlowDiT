module;
#include <cublasLt.h>
#include <flowdit/cuda.h>
module flowdit.neural.matmul;
import std;
namespace flowdit::neural {
    namespace {
        void check(cublasStatus_t status) {
            if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::format("cuBLASLt: {}", cublasGetStatusString(status))};
        }
    } // namespace
    MatmulRuntime::Plan::Plan(cublasLtHandle_t handle, PlanKey k, std::size_t workspace) : key{k} {
        check(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        const auto ta = k.transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N, tb = k.transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
        check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
        check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
        const auto ar = k.transpose_b ? k.reduction : k.columns, ac = k.transpose_b ? k.columns : k.reduction;
        const auto br = k.transpose_a ? k.rows : k.reduction, bc = k.transpose_a ? k.reduction : k.rows;
        check(cublasLtMatrixLayoutCreate(&a, CUDA_R_16BF, ar, ac, ar));
        check(cublasLtMatrixLayoutCreate(&b, CUDA_R_16BF, br, bc, br));
        check(cublasLtMatrixLayoutCreate(&output, k.float_output ? CUDA_R_32F : CUDA_R_16BF, k.columns, k.rows, k.columns));
        cublasLtMatmulPreference_t preference{};
        check(cublasLtMatmulPreferenceCreate(&preference));
        check(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace, sizeof(workspace)));
        const std::uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_NONE;
        check(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &reduction, sizeof(reduction)));
        std::array<cublasLtMatmulHeuristicResult_t, 1> algorithms{};
        int count{};
        const auto status = cublasLtMatmulAlgoGetHeuristic(handle, operation, a, b, output, output, preference, algorithms.size(), algorithms.data(), &count);
        cublasLtMatmulPreferenceDestroy(preference);
        check(status);
        if (count == 0) throw std::runtime_error{std::format("No BF16 GEMM for {} x {} x {}", k.rows, k.columns, k.reduction)};
        check(algorithms.front().state);
        algorithm = algorithms.front().algo;
    }
    MatmulRuntime::Plan::~Plan() {
        cublasLtMatrixLayoutDestroy(output);
        cublasLtMatrixLayoutDestroy(b);
        cublasLtMatrixLayoutDestroy(a);
        cublasLtMatmulDescDestroy(operation);
    }
    MatmulRuntime::MatmulRuntime(::cuda::stream_ref s, MatmulRuntimeConfiguration c) : stream{s}, workspace{stream, ::cuda::device_default_memory_pool(stream.device()), c.workspace_byte_count, ::cuda::no_init} {
        check(cublasLtCreate(&handle));
    }
    MatmulRuntime::~MatmulRuntime() {
        plans.clear();
        cublasLtDestroy(handle);
    }
    void MatmulRuntime::execute(const MatmulRequest& r) {
        const PlanKey key{r.rows, r.columns, r.reduction, r.transpose_a, r.transpose_b, r.float_output};
        auto plan = std::ranges::find(plans, key, &Plan::key);
        if (plan == plans.end()) plan = plans.emplace(plans.end(), handle, key, workspace.size());
        const float one = 1.f;
        check(cublasLtMatmul(handle, plan->operation, &one, r.b, plan->a, r.a, plan->b, &r.beta, r.output, plan->output, r.output, plan->output, &plan->algorithm, workspace.data(), workspace.size(), stream.get()));
    }
} // namespace flowdit::neural
