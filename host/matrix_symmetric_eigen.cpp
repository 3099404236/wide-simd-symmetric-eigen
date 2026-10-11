// Host-side Layout Configuration - MatrixSymmetricEigen
#include "interface/operator_registry.hpp"
#include "interface/platform_hardware.hpp"

#include "../device_kernel/matrix_symmetric_eigen_layout.h"
#include "../device_kernel/layout_key_matrix_symmetric_eigen.h"

namespace kernel_layout {
    static hpc_rt::hpc_status_t ComputeLayout(simd_rt::LayoutContext *context) {
        auto platform = target_platform::TargetPlatform(context->GetHardwareProfile());
        int32_t num_simd_cores = platform.GetSimdUnitCount();
        if (num_simd_cores <= 0) num_simd_cores = 24;

        uint32_t n = 1;
        uint32_t batch_size = 1;

        const simd_rt::BufferShape *shape_A_storage = context->GetArrayShape(0);
        if (shape_A_storage != nullptr) {
            const auto &shape_A = shape_A_storage->GetBufferShape();
            size_t dimA = shape_A.GetArrayRank();
            if (dimA >= 2) {
                n = static_cast<uint32_t>(shape_A.GetArrayDimension(dimA - 1));
                int64_t total = shape_A.GetArrayElementCount();
                if (total > 0 && n > 0) {
                    batch_size = static_cast<uint32_t>(total / (n * n));
                }
            }
        }
        if (n == 0) n = 1;
        if (batch_size == 0) batch_size = 1;

        const simd_rt::RuntimeAttrs *attrs = context->GetAttrs();
        const char *uplo_str = (attrs != nullptr) ? attrs->GetStr(0) : "L";
        uint32_t is_upper = (uplo_str != nullptr && (uplo_str[0] == 'U' || uplo_str[0] == 'u')) ? 1 : 0;

        const simd_rt::ArrayRef *mat_A = context->GetMandatoryInput(0);
        hpc_rt::DataType dtype_A = (mat_A != nullptr) ? mat_A->GetDataType() : hpc_rt::DTYPE_F32;
        uint32_t DT_A = static_cast<uint32_t>(dtype_A);
        SIMD_TPL_SELECT_PARAM(context, DT_A);

        MatrixSymmetricEigenLayoutData *layout_cfg = context->GetLayoutConfig<MatrixSymmetricEigenLayoutData>();
        layout_cfg->n = n;
        layout_cfg->batch_size = batch_size;
        layout_cfg->is_upper = is_upper;

        context->SetParallelUnits(num_simd_cores);

        uint32_t n_stride = ((n + 7) / 8) * 8;
        // Per core: the matrix M and the accumulator Wt (n x n_stride each) for n > 128. ProcessVec needs none.
        size_t ws_floats = static_cast<size_t>(num_simd_cores) * 2 * n * n_stride;
        if (ws_floats < static_cast<size_t>(2 * 512 * 512)) ws_floats = static_cast<size_t>(2 * 512 * 512);
        size_t *currentWorkspace = context->GetWorkspaceRequirements(1);
        currentWorkspace[0] = ws_floats * sizeof(float);
        return hpc_rt::HPC_STATUS_SUCCESS;
    }
}  // namespace kernel_layout

namespace hpc_rt {
    static hpc_status_t DeduceShape(simd_rt::ShapeDeduceContext *context) {
        const simd_rt::Shape *shape_A = context->GetArrayShape(0);
        simd_rt::Shape *shape_evals = context->GetResultShape(0);
        simd_rt::Shape *shape_evecs = context->GetResultShape(1);
        if (shape_A == nullptr || shape_evals == nullptr || shape_evecs == nullptr) {
            return HPC_STATUS_SUCCESS;
        }

        size_t dimA = shape_A->GetArrayRank();
        if (dimA < 2) return HPC_STATUS_FAILED;
        int64_t n = shape_A->GetArrayDimension(dimA - 1);

        shape_evals->SetDimNum(dimA - 1);
        for (size_t i = 0; i < dimA - 2; ++i) {
            shape_evals->SetDim(i, shape_A->GetArrayDimension(i));
        }
        shape_evals->SetDim(dimA - 2, n);

        *shape_evecs = *shape_A;
        return HPC_STATUS_SUCCESS;
    }

    static hpc_status_t DeduceDtype(simd_rt::DtypeDeduceContext *context) {
        context->SetResultDtype(0, context->GetArrayDtype(0));
        context->SetResultDtype(1, context->GetArrayDtype(0));
        return hpc_rt::HPC_STATUS_SUCCESS;
    }
}  // namespace hpc_rt

namespace ops {
    class MatrixSymmetricEigen : public OpDescriptor {
    public:
        explicit MatrixSymmetricEigen(const char *name) : OpDescriptor(name) {
            this->Input("A")
                .ArgRequirement(REQ_MANDATORY)
                .DataType({hpc_rt::DTYPE_F32})
                .Format({hpc_rt::LAYOUT_ND_FORMAT});
            this->Output("eigenvalues")
                .ArgRequirement(REQ_MANDATORY)
                .DataType({hpc_rt::DTYPE_F32})
                .Format({hpc_rt::LAYOUT_ND_FORMAT});
            this->Output("eigenvectors")
                .ArgRequirement(REQ_MANDATORY)
                .DataType({hpc_rt::DTYPE_F32})
                .Format({hpc_rt::LAYOUT_ND_FORMAT});
            this->Attr("UPLO").AttrType(REQ_OPTIONAL).String("L");
            this->SetShapeDeduction(hpc_rt::DeduceShape).SetDtypeDeduction(hpc_rt::DeduceDtype);
            this->SimdCore()
                .ConfigureLayout(kernel_layout::ComputeLayout)
                .TargetProfile("simd_target_v2");
        }
    };
    REGISTER_OPERATOR(MatrixSymmetricEigen);
}  // namespace ops
