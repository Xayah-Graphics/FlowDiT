module;
#include <cublasLt.h>
#include <flowdit/cuda.h>
export module flowdit.neural.matmul;
import std;
export namespace flowdit::neural {
    struct MatmulRuntimeConfiguration final {
        std::size_t workspace_byte_count;
    };
    struct MatmulRequest final {
        const std::uint16_t* a;
        const std::uint16_t* b;
        void* output;
        std::uint32_t rows, columns, reduction;
        bool transpose_a{}, transpose_b{}, float_output{};
        float beta{};
    };
    struct MatmulRuntime final {
        struct PlanKey final {
            std::uint32_t rows, columns, reduction;
            bool transpose_a, transpose_b, float_output;
            bool operator==(const PlanKey&) const = default;
        };
        struct Plan final {
            PlanKey key;
            cublasLtMatmulDesc_t operation{};
            cublasLtMatrixLayout_t a{}, b{}, output{};
            cublasLtMatmulAlgo_t algorithm{};
            Plan(cublasLtHandle_t handle, PlanKey key, std::size_t workspace);
            ~Plan();
            Plan(const Plan&)            = delete;
            Plan& operator=(const Plan&) = delete;
        };
        ::cuda::stream_ref stream;
        cublasLtHandle_t handle{};
        ::cuda::device_buffer<std::uint8_t> workspace;
        std::list<Plan> plans;
        MatmulRuntime(::cuda::stream_ref stream, MatmulRuntimeConfiguration configuration);
        ~MatmulRuntime();
        MatmulRuntime(const MatmulRuntime&)            = delete;
        MatmulRuntime& operator=(const MatmulRuntime&) = delete;
        void execute(const MatmulRequest& request);
    };
} // namespace flowdit::neural
