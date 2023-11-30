# Copyright (c) 2023, Parallel Software and Systems Group, University of 
# Maryland. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import glob

L2_BANKS = 32  # default assuming mi200
XMIN = 0.01
XMAX = 1000


class Roofline:
    def __init__(self, path=None, data=None):
        self.path = path
        self.data = data

    @staticmethod
    def from_ncu(path):
        if path is None or not path.endswith(".csv"):
            raise ValueError("ncu data should be in CSV format.")
        return Roofline(path, pd.read_csv(path))

    @staticmethod
    def from_omniperf(path):
        if not os.path.isdir(path):
            raise ValueError("This function should take a directory.")
        else:
            csv_files = glob.glob(path + "/*")
            pmc_perf = path + "/pmc_perf.csv"
            roofline = path + "/roofline.csv"
            if pmc_perf in csv_files and roofline in csv_files:
                pmc_perf_data = pd.read_csv(pmc_perf)
                roofline_data = pd.read_csv(roofline)
                return Roofline(path, [pmc_perf_data, roofline_data])
            else:
                raise ValueError(
                    "The directory should contain pmc_perf.csv and roofline.csv."
                )

    def analyze_ncu(self, data):
        dataframe = data.copy(deep=True)

        dataframe = dataframe[self.data["Kernel Name"].notna()]

        max_single_sm_per_cycle = dataframe[
            "derived__sm__sass_thread_inst_executed_op_ffma_pred_on_x2"
        ].astype(float)
        cycles_sm_per_sec = dataframe["sm__cycles_elapsed.avg.per_second"].astype(float)
        bytes_dram_per_cycle = dataframe["dram__bytes.sum.peak_sustained"].astype(float)
        cycles_dram_per_sec = dataframe["dram__cycles_elapsed.avg.per_second"].astype(
            float
        )

        single_peakwork = max_single_sm_per_cycle * cycles_sm_per_sec
        single_peaktraffic = bytes_dram_per_cycle * cycles_dram_per_sec

        max_double_sm_per_cycle = dataframe[
            "derived__sm__sass_thread_inst_executed_op_pmc_perf_dataframema_pred_on_x2"
        ].astype(float)

        double_peakwork = max_double_sm_per_cycle * cycles_sm_per_sec
        double_peaktraffic = bytes_dram_per_cycle * cycles_dram_per_sec

        single_fadd_per_cycle = dataframe[
            "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed"
        ].astype(float)
        single_fmul_per_cycle = dataframe[
            "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed"
        ].astype(float)
        single_ffma_per_cycle = dataframe[
            "derived__smsp__sass_thread_inst_executed_op_ffma_pred_on_x2"
        ].astype(float)
        cycles_smsp_per_sec = dataframe["smsp__cycles_elapsed.avg.per_second"].astype(
            float
        )
        cycles_dram_per_sec_sum = dataframe["dram__bytes.sum.per_second"].astype(float)

        dataframe["single_performance"] = (
            single_fadd_per_cycle + single_fmul_per_cycle + single_ffma_per_cycle
        ) * cycles_smsp_per_sec

        double_fadd_per_cycle = dataframe[
            "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum.per_cycle_elapsed"
        ].astype(float)
        double_fmul_per_cycle = dataframe[
            "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum.per_cycle_elapsed"
        ].astype(float)
        double_ffma_per_cycle = dataframe[
            "derived__smsp__sass_thread_inst_executed_op_pmc_perf_dataframema_pred_on_x2"
        ].astype(float)
        cycles_smsp_per_sec = dataframe["smsp__cycles_elapsed.avg.per_second"].astype(
            float
        )
        cycles_dram_per_sec_sum = dataframe["dram__bytes.sum.per_second"].astype(float)

        double_performance = (
            double_fadd_per_cycle + double_fmul_per_cycle + double_ffma_per_cycle
        ) * cycles_smsp_per_sec

        dataframe["achieved_aritmetic_intensity"] = (
            double_performance / cycles_dram_per_sec_sum
        ) / 1000
        dataframe["aritmetic_intensity_double"] = (
            double_peakwork / double_peaktraffic
        ) / 1000
        dataframe["aritmetic_intensity_single"] = (
            single_peakwork / single_peaktraffic
        ) / 1000

        dataframe["achieved_performance"] = double_performance / 1000
        dataframe["peak_performance_double"] = double_peakwork / 1000
        dataframe["peak_performance_single"] = single_peakwork / 1000

        return dataframe[
            [
                "Kernel Name",
                "achieved_aritmetic_intensity",
                "aritmetic_intensity_double",
                "aritmetic_intensity_single",
                "achieved_performance",
                "peak_performance_double",
                "peak_performance_single",
            ]
        ]

    def analyze_omniperf(self, data):
        def calculate_application_performance(pmc_perf_dataframe):
            try:
                pmc_perf_dataframe["total_flops"] = (
                    64
                    * (
                        pmc_perf_dataframe["SQ_INSTS_VALU_ADD_F16"]
                        + pmc_perf_dataframe["SQ_INSTS_VALU_MUL_F16"]
                        + (2 * pmc_perf_dataframe["SQ_INSTS_VALU_FMA_F16"])
                        + pmc_perf_dataframe["SQ_INSTS_VALU_TRANS_F16"]
                    )
                    + (
                        64
                        * (
                            pmc_perf_dataframe["SQ_INSTS_VALU_ADD_F32"]
                            + pmc_perf_dataframe["SQ_INSTS_VALU_MUL_F32"]
                            + (2 * pmc_perf_dataframe["SQ_INSTS_VALU_FMA_F32"])
                            + pmc_perf_dataframe["SQ_INSTS_VALU_TRANS_F32"]
                        )
                    )
                    + (
                        64
                        * (
                            pmc_perf_dataframe["SQ_INSTS_VALU_ADD_F64"]
                            + pmc_perf_dataframe["SQ_INSTS_VALU_MUL_F64"]
                            + (2 * pmc_perf_dataframe["SQ_INSTS_VALU_FMA_F64"])
                            + pmc_perf_dataframe["SQ_INSTS_VALU_TRANS_F64"]
                        )
                    )
                    + (pmc_perf_dataframe["SQ_INSTS_VALU_MFMA_MOPS_F16"] * 512)
                    + (pmc_perf_dataframe["SQ_INSTS_VALU_MFMA_MOPS_BF16"] * 512)
                    + (pmc_perf_dataframe["SQ_INSTS_VALU_MFMA_MOPS_F32"] * 512)
                    + (pmc_perf_dataframe["SQ_INSTS_VALU_MFMA_MOPS_F64"] * 512)
                )
            except KeyError:
                # if verbose >= 3:
                # print("{}: Skipped total_flops at index {}".format(kernelName[:35], idx))
                pass

            try:
                pmc_perf_dataframe["valu_flops"] = (
                    64
                    * (
                        pmc_perf_dataframe["SQ_INSTS_VALU_ADD_F16"]
                        + pmc_perf_dataframe["SQ_INSTS_VALU_MUL_F16"]
                        + (2 * pmc_perf_dataframe["SQ_INSTS_VALU_FMA_F16"])
                        + pmc_perf_dataframe["SQ_INSTS_VALU_TRANS_F16"]
                    )
                    + 64
                    * (
                        pmc_perf_dataframe["SQ_INSTS_VALU_ADD_F32"]
                        + pmc_perf_dataframe["SQ_INSTS_VALU_MUL_F32"]
                        + (2 * pmc_perf_dataframe["SQ_INSTS_VALU_FMA_F32"])
                        + pmc_perf_dataframe["SQ_INSTS_VALU_TRANS_F32"]
                    )
                    + 64
                    * (
                        pmc_perf_dataframe["SQ_INSTS_VALU_ADD_F64"]
                        + pmc_perf_dataframe["SQ_INSTS_VALU_MUL_F64"]
                        + (2 * pmc_perf_dataframe["SQ_INSTS_VALU_FMA_F64"])
                        + pmc_perf_dataframe["SQ_INSTS_VALU_TRANS_F64"]
                    )
                )
            except KeyError:
                # if verbose >= 3:
                #     print("{}: Skipped valu_flops at index {}".format(kernelName[:35], idx))
                pass

            try:
                pmc_perf_dataframe["mfma_flops_f16"] = (
                    pmc_perf_dataframe["SQ_INSTS_VALU_MFMA_MOPS_F16"] * 512
                )
            except KeyError:
                # if verbose >= 3:
                #     print("{}: Skipped mfma ops at index {}".format(kernelName[:35], idx))
                pass
            try:
                pmc_perf_dataframe["mfma_flops_bf16"] = (
                    pmc_perf_dataframe["SQ_INSTS_VALU_MFMA_MOPS_BF16"] * 512
                )
            except KeyError:
                # if verbose >= 3:
                #     print("{}: Skipped mfma ops at index {}".format(kernelName[:35], idx))
                pass
            try:
                pmc_perf_dataframe["mfma_flops_f32"] = (
                    pmc_perf_dataframe["SQ_INSTS_VALU_MFMA_MOPS_F32"] * 512
                )
            except KeyError:
                # if verbose >= 3:
                #     print("{}: Skipped mfma ops at index {}".format(kernelName[:35], idx))
                pass
            try:
                pmc_perf_dataframe["mfma_flops_f64"] = (
                    pmc_perf_dataframe["SQ_INSTS_VALU_MFMA_MOPS_F64"] * 512
                )
            except KeyError:
                # if verbose >= 3:
                #     print("{}: Skipped mfma ops at index {}".format(kernelName[:35], idx))
                pass
            try:
                if "SQ_INSTS_VALU_MFMA_MOPS_I8" in pmc_perf_dataframe.columns:
                    pmc_perf_dataframe["mfma_iops_i8"] = (
                        pmc_perf_dataframe["SQ_INSTS_VALU_MFMA_MOPS_I8"] * 512
                    )
            except KeyError:
                # if verbose >= 3:
                #     print("{}: Skipped mfma ops at index {}".format(kernelName[:35], idx))
                pass

            try:
                lds_data = (
                    (
                        pmc_perf_dataframe["SQ_LDS_IDX_ACTIVE"]
                        - pmc_perf_dataframe["SQ_LDS_BANK_CONFLICT"]
                    )
                    * 4
                    * L2_BANKS
                )
            except KeyError:
                # if verbose >= 3:
                #     print("{}: Skipped lds_data at index {}".format(kernelName[:35], idx))
                pass

            try:
                L1cache_data = pmc_perf_dataframe["TCP_TOTAL_CACHE_ACCESSES_sum"] * 64
            except KeyError:
                # if verbose >= 3:
                #     print("{}: Skipped L1cache_data at index {}".format(kernelName[:35], idx))
                pass

            try:
                L2cache_data = (
                    pmc_perf_dataframe["TCP_TCC_WRITE_REQ_sum"] * 64
                    + pmc_perf_dataframe["TCP_TCC_ATOMIC_WITH_RET_REQ_sum"] * 64
                    + pmc_perf_dataframe["TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum"] * 64
                    + pmc_perf_dataframe["TCP_TCC_READ_REQ_sum"] * 64
                )
            except KeyError:
                # if verbose >= 3:
                #     print("{}: Skipped L2cache_data at index {}".format(kernelName[:35], idx))
                pass

            try:
                hbm_data = (
                    (pmc_perf_dataframe["TCC_EA_RDREQ_32B_sum"] * 32)
                    + (
                        (
                            pmc_perf_dataframe["TCC_EA_RDREQ_sum"]
                            - pmc_perf_dataframe["TCC_EA_RDREQ_32B_sum"]
                        )
                        * 64
                    )
                    + (pmc_perf_dataframe["TCC_EA_WRREQ_64B_sum"] * 64)
                    + (
                        (
                            pmc_perf_dataframe["TCC_EA_WRREQ_sum"]
                            - pmc_perf_dataframe["TCC_EA_WRREQ_64B_sum"]
                        )
                        * 32
                    )
                )
            except KeyError:
                # if verbose >= 3:
                #     print("{}: Skipped hbm_data at index {}".format(kernelName[:35], idx))
                pass

            duration = pmc_perf_dataframe["EndNs"] - pmc_perf_dataframe["BeginNs"]

            pmc_perf_dataframe["ai_l1"] = (
                pmc_perf_dataframe["total_flops"] / L1cache_data
            )
            pmc_perf_dataframe["ai_l2"] = (
                pmc_perf_dataframe["total_flops"] / L2cache_data
            )
            pmc_perf_dataframe["ai_hbm"] = pmc_perf_dataframe["total_flops"] / hbm_data
            pmc_perf_dataframe["ai_hbm"] = pmc_perf_dataframe["total_flops"] / duration

        def calculate_roof(pmc_perf_dataframe, roofline_dataframe):
            cache_hierarchy = ["HBM", "L2", "L1", "LDS"]

            roofline_dataframe = roofline_dataframe.drop(
                columns=roofline_dataframe.filter(like="High").columns
            )
            roofline_dataframe = roofline_dataframe.drop(
                columns=roofline_dataframe.filter(like="Low").columns
            )
            # print(roofline_dataframe)
            peakOps = list(roofline_dataframe.filter(like="Flops").columns)
            peakOps.append("MFMAI8Ops")
            peakBw = roofline_dataframe.filter(like="Bw").columns
            peakMFMA = roofline_dataframe.filter(regex="MFMA.*.Flops").columns

            x1 = float(XMIN)
            y1 = float(XMIN) * roofline_dataframe[peakBw]

            x2 = roofline_dataframe[peakOps] / roofline_dataframe[peakBw]
            y2 = roofline_dataframe[peakOps]

            x2_mfma = roofline_dataframe[peakMFMA] / roofline_dataframe[peakBw]
            y2_mfma = roofline_dataframe[peakMFMA]

            # x0 = XMAX
            # if x2 < x0:
            #     x0 = x2

            # x0_mfma = XMAX
            # if x2_mfma < x0_mfma:
            #     x0_mfma = x2_mfma

        pmc_perf_dataframe = data[0].copy(deep=True)
        roofline_dataframe = data[1].copy(deep=True)

        calculate_application_performance(pmc_perf_dataframe)
        calculate_roof(pmc_perf_dataframe, roofline_dataframe)
