# Copyright (c) 2023, Parallel Software and Systems Group, University of
# Maryland. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import glob
import math

L2_BANKS = 32  # default assuming mi200
XMIN = 0.01
XMAX = 1000


class Roofline:
    def __init__(self, path=None, data=None, rooflines=None, data_format=None):
        self.path = path
        self.data = data
        self.rooflines = rooflines
        self.precisions = []
        self.units = []
        self.data_format = data_format

    @staticmethod
    def from_ncu(path):
        if path is None or not path.endswith(".csv"):
            raise ValueError("ncu data should be in CSV format.")
        return Roofline(path, pd.read_csv(path), data_format="ncu")

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
                return Roofline(
                    path, [pmc_perf_data, roofline_data], data_format="omniperf"
                )
            else:
                raise ValueError(
                    "The directory should contain pmc_perf.csv and roofline.csv."
                )

    def analyze(self):
        if self.data_format == "ncu":
            self.analyze_ncu()
        elif self.data_format == "omniperf":
            self.analyze_omniperf()

    def analyze_ncu(self):
        """
        This function calculates all the required points to create a roofline plot.
        Specifically, it calculates double, single, half precisions and tensor core.
        It also calculates DRAM, L1, and L2 ceilings and achieved values.
        """

        def _calculate_l1(peakwork, achieved_performance, precision):
            # L1 AI
            L1_roof_AI = "{}_l1_roof_AI".format(precision)
            L1_achieved_AI = "{}_l1_achieved_AI".format(precision)

            dataframe[L1_roof_AI] = peakwork / l1_peaktraffic
            dataframe[L1_achieved_AI] = achieved_performance / l1_cycles_per_sec_sum

            columns.extend([L1_roof_AI, L1_achieved_AI])

        def _calculate_l2(peakwork, achieved_performance, precision):
            # L2 AI
            L2_roof_AI = "{}_l2_roof_AI".format(precision)
            L2_achieved_AI = "{}_l2_achieved_AI".format(precision)

            dataframe[L2_roof_AI] = peakwork / l2_peaktraffic
            dataframe[L2_achieved_AI.format(precision)] = (
                achieved_performance / l2_cycles_per_sec_sum
            )

            columns.extend([L2_roof_AI, L2_achieved_AI])

        def _calculate_single():
            max_sm_per_cycle = (
                _obtain_value(
                    dataframe[
                        "sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained"
                    ]
                )
                * 2
            )

            peakwork = max_sm_per_cycle * cycles_sm_per_sec

            # DRAM roof AI
            dataframe["single_dram_roof_AI"] = peakwork / dram_peaktraffic

            # DRAM Achieved
            fadd_per_cycle = _obtain_value(
                dataframe[
                    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed"
                ]
            )
            fmul_per_cycle = _obtain_value(
                dataframe[
                    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed"
                ]
            )
            ffma_per_cycle = (
                _obtain_value(
                    dataframe[
                        "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed"
                    ]
                )
                * 2
            )

            cycles_smsp_per_sec = _obtain_value(
                dataframe["smsp__cycles_elapsed.avg.per_second"]
            )

            # DRAM, L1, and L2 Achieved Work
            achieved_performance = (
                fadd_per_cycle + fmul_per_cycle + ffma_per_cycle
            ) * cycles_smsp_per_sec

            dataframe["single_dram_achieved_AI"] = (
                achieved_performance / dram_cycles_per_sec_sum
            )

            dataframe["single_achieved_performance"] = achieved_performance
            dataframe["single_roof_performance"] = peakwork

            columns.extend(
                [
                    "single_dram_roof_AI",
                    "single_dram_achieved_AI",
                    "single_achieved_performance",
                    "single_roof_performance",
                ]
            )

            if l1:
                _calculate_l1(peakwork, achieved_performance, precision="single")
            if l2:
                _calculate_l2(peakwork, achieved_performance, precision="single")

        def _calculate_double():
            max_sm_per_cycle = (
                _obtain_value(
                    dataframe[
                        "sm__sass_thread_inst_executed_op_dfma_pred_on.sum.peak_sustained"
                    ]
                )
                * 2
            )

            peakwork = max_sm_per_cycle * cycles_sm_per_sec

            # DRAM roof AI
            dataframe["double_dram_roof_AI"] = peakwork / dram_peaktraffic

            # DRAM Achieved
            dadd_per_cycle = _obtain_value(
                dataframe[
                    "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum.per_cycle_elapsed"
                ]
            )

            dmul_per_cycle = _obtain_value(
                dataframe[
                    "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum.per_cycle_elapsed"
                ]
            )

            dfma_per_cycle = (
                _obtain_value(
                    dataframe[
                        "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum.per_cycle_elapsed"
                    ]
                )
                * 2
            )

            cycles_smsp_per_sec = _obtain_value(
                dataframe["smsp__cycles_elapsed.avg.per_second"]
            )

            # DRAM, L1, and L2 Achieved Work
            achieved_performance = (
                dadd_per_cycle + dmul_per_cycle + dfma_per_cycle
            ) * cycles_smsp_per_sec

            dataframe["double_dram_achieved_AI"] = (
                achieved_performance / dram_cycles_per_sec_sum
            )

            dataframe["double_achieved_performance"] = achieved_performance
            dataframe["double_roof_performance"] = peakwork

            columns.extend(
                [
                    "double_dram_roof_AI",
                    "double_dram_achieved_AI",
                    "double_achieved_performance",
                    "double_roof_performance",
                ]
            )

            if l1:
                _calculate_l1(peakwork, achieved_performance, precision="double")
            if l2:
                _calculate_l2(peakwork, achieved_performance, precision="double")

        def _calculate_half():
            max_sm_per_cycle = (
                _obtain_value(
                    dataframe[
                        "sm__sass_thread_inst_executed_op_hfma_pred_on.sum.peak_sustained"
                    ]
                )
                * 2
            )

            peakwork = max_sm_per_cycle * cycles_sm_per_sec

            # DRAM roof AI
            dataframe["half_dram_roof_AI"] = peakwork / dram_peaktraffic

            # DRAM Achieved
            hadd_per_cycle = _obtain_value(
                dataframe[
                    "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum.per_cycle_elapsed"
                ]
            )
            hmul_per_cycle = _obtain_value(
                dataframe[
                    "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum.per_cycle_elapsed"
                ]
            )
            hfma_per_cycle = (
                _obtain_value(
                    dataframe[
                        "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum.per_cycle_elapsed"
                    ]
                )
                * 2
            )

            cycles_smsp_per_sec = _obtain_value(
                dataframe["smsp__cycles_elapsed.avg.per_second"]
            )

            # DRAM, L1, and L2 Achieved Work
            achieved_performance = (
                hadd_per_cycle + hmul_per_cycle + hfma_per_cycle
            ) * cycles_smsp_per_sec

            dataframe["half_dram_achieved_AI"] = (
                achieved_performance / dram_cycles_per_sec_sum
            )

            dataframe["half_achieved_performance"] = achieved_performance
            dataframe["half_roof_performance"] = peakwork

            columns.extend(
                [
                    "half_dram_roof_AI",
                    "half_dram_achieved_AI",
                    "half_achieved_performance",
                    "half_roof_performance",
                ]
            )

            if l1:
                _calculate_l1(peakwork, achieved_performance, precision="half")
            if l2:
                _calculate_l2(peakwork, achieved_performance, precision="half")

        def _calculate_tensor():
            max_sm_per_cycle = (
                _obtain_value(
                    dataframe["sm__inst_executed_pipe_tensor.sum.peak_sustained"]
                )
                * 512
            )

            peakwork = max_sm_per_cycle * cycles_sm_per_sec

            # DRAM roof AI
            dataframe["tensor_dram_roof_AI"] = peakwork / dram_peaktraffic

            # DRAM Achieved
            tensor_per_cycle = (
                _obtain_value(
                    dataframe["smsp__inst_executed_pipe_tensor.sum.per_cycle_elapsed"]
                )
                * 512
            )

            cycles_smsp_per_sec = _obtain_value(
                dataframe["smsp__cycles_elapsed.avg.per_second"]
            )

            # DRAM, L1, and L2 Achieved Work
            achieved_performance = tensor_per_cycle * cycles_smsp_per_sec

            dataframe["tensor_dram_achieved_AI"] = (
                achieved_performance / dram_cycles_per_sec_sum
            )

            dataframe["tensor_achieved_performance"] = achieved_performance
            dataframe["tensor_roof_performance"] = peakwork

            columns.extend(
                [
                    "tensor_dram_roof_AI",
                    "tensor_dram_achieved_AI",
                    "tensor_achieved_performance",
                    "tensor_roof_performance",
                ]
            )

            if l1:
                _calculate_l1(peakwork, achieved_performance, precision="tensor")
            if l2:
                _calculate_l2(peakwork, achieved_performance, precision="tensor")

        def _obtain_value(column):
            """
            Different .csv files can have different unit for
            the same columns.
            Converts everything to byte and seconds.
            """
            units = column.iloc[0]
            units = units.split("/")
            numerator = units[0]
            denominator = units[1]

            assert (
                "second" not in numerator
            ), "{} units are not converted correctly.".format(column)
            assert (
                "byte" not in denominator
            ), "{} units are not converted correctly.".format(column)

            byte_units = {
                "byte": 1,
                "Kbyte": 1e3,
                "Mbyte": 1e6,
                "Gbyte": 1e9,
                "Tbyte": 1e12,
                "Pbyte": 1e15,
            }
            time_units = {"nsecond": 1e9, "usecond": 1e6, "msecond": 1e3, "second": 1}

            column = column.iloc[1:].astype(np.float64)
            if "second" in denominator:
                if denominator not in time_units.keys():
                    raise ValueError(
                        "{} is not converted to second".format(denominator)
                    )

                column *= time_units[denominator]

            if "byte" in numerator:
                if numerator not in byte_units.keys():
                    raise ValueError("{} is not converted to byte".format(numerator))

                column *= byte_units[numerator]

            return column

        dataframe = self.data.copy(deep=True)

        l1 = False
        l2 = False
        if "l1tex__t_bytes.sum.peak_sustained" in dataframe.columns:
            l1 = True
            self.units.append("l1")
        if "lts__t_bytes.sum.peak_sustained" in dataframe.columns:
            l2 = True
            self.units.append("l2")

        cycles_sm_per_sec = _obtain_value(
            dataframe["sm__cycles_elapsed.avg.per_second"]
        )

        # DRAM Roofline
        dram_bytes_per_cycle = _obtain_value(
            dataframe["dram__bytes.sum.peak_sustained"]
        )
        self.units.append("dram")

        dram_cycles_per_sec = _obtain_value(
            dataframe["dram__cycles_elapsed.avg.per_second"]
        )
        dram_peaktraffic = dram_bytes_per_cycle * dram_cycles_per_sec

        # DRAM Achieved Traffic
        dram_cycles_per_sec_sum = _obtain_value(dataframe["dram__bytes.sum.per_second"])

        # L1 Roofline
        if l1:
            l1_bytes_per_cycle = _obtain_value(
                dataframe["l1tex__t_bytes.sum.peak_sustained"]
            )

            l1_cycles_per_sec = _obtain_value(
                dataframe["l1tex__cycles_elapsed.avg.per_second"]
            )

            l1_peaktraffic = l1_bytes_per_cycle * l1_cycles_per_sec

            # L1 Achieved Traffic
            l1_cycles_per_sec_sum = _obtain_value(
                dataframe["l1tex__t_bytes.sum.per_second"]
            )

        # L2 Roofline
        if l2:
            l2_bytes_per_cycle = _obtain_value(
                dataframe["lts__t_bytes.sum.peak_sustained"]
            )

            l2_cycles_per_sec = _obtain_value(
                dataframe["lts__cycles_elapsed.avg.per_second"]
            )

            l2_peaktraffic = l2_bytes_per_cycle * l2_cycles_per_sec

            # L2 Achieved Traffic
            l2_cycles_per_sec_sum = _obtain_value(
                dataframe["lts__t_bytes.sum.per_second"]
            )

        columns = ["Kernel Name"]
        _calculate_single()
        self.precisions.append("single")

        _calculate_double()
        self.precisions.append("double")

        if (
            "sm__sass_thread_inst_executed_op_hfma_pred_on.sum.peak_sustained"
            in dataframe.columns
        ):
            _calculate_half()
            self.precisions.append("half")

        if "sm__inst_executed_pipe_tensor.sum.peak_sustained" in dataframe.columns:
            _calculate_tensor()
            self.precisions.append("tensor")

        dataframe = dataframe[columns]
        self.rooflines = dataframe.iloc[1:]

    # Incomplete
    def analyze_omniperf(self):
        def _calculate_metrics(pmc_perf_dataframe):
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
                pass

            try:
                pmc_perf_dataframe["mfma_flops_f16"] = (
                    pmc_perf_dataframe["SQ_INSTS_VALU_MFMA_MOPS_F16"] * 512
                )
            except KeyError:
                pass

            try:
                pmc_perf_dataframe["mfma_flops_bf16"] = (
                    pmc_perf_dataframe["SQ_INSTS_VALU_MFMA_MOPS_BF16"] * 512
                )
            except KeyError:
                pass

            try:
                pmc_perf_dataframe["mfma_flops_f32"] = (
                    pmc_perf_dataframe["SQ_INSTS_VALU_MFMA_MOPS_F32"] * 512
                )
            except KeyError:
                pass

            try:
                pmc_perf_dataframe["mfma_flops_f64"] = (
                    pmc_perf_dataframe["SQ_INSTS_VALU_MFMA_MOPS_F64"] * 512
                )
            except KeyError:
                pass

            try:
                if "SQ_INSTS_VALU_MFMA_MOPS_I8" in pmc_perf_dataframe.columns:
                    pmc_perf_dataframe["mfma_iops_i8"] = (
                        pmc_perf_dataframe["SQ_INSTS_VALU_MFMA_MOPS_I8"] * 512
                    )
            except KeyError:
                pass

            try:
                pmc_perf_dataframe["total_flops"] = pmc_perf_dataframe["valu_flops"]
                +pmc_perf_dataframe["mfma_flops_f16"]
                +pmc_perf_dataframe["mfma_flops_bf16"]
                +pmc_perf_dataframe["mfma_flops_f32"]
                +pmc_perf_dataframe["mfma_flops_f64"]
            except KeyError:
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
                pass

            try:
                L1cache_data = pmc_perf_dataframe["TCP_TOTAL_CACHE_ACCESSES_sum"] * 64
            except KeyError:
                pass

            try:
                L2cache_data = (
                    pmc_perf_dataframe["TCP_TCC_WRITE_REQ_sum"] * 64
                    + pmc_perf_dataframe["TCP_TCC_ATOMIC_WITH_RET_REQ_sum"] * 64
                    + pmc_perf_dataframe["TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum"] * 64
                    + pmc_perf_dataframe["TCP_TCC_READ_REQ_sum"] * 64
                )
            except KeyError:
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
                pass

            duration = pmc_perf_dataframe["EndNs"] - pmc_perf_dataframe["BeginNs"]

            pmc_perf_dataframe["ai_l1"] = (
                pmc_perf_dataframe["total_flops"] / L1cache_data
            )
            pmc_perf_dataframe["ai_l2"] = (
                pmc_perf_dataframe["total_flops"] / L2cache_data
            )
            pmc_perf_dataframe["ai_hbm"] = pmc_perf_dataframe["total_flops"] / hbm_data

            pmc_perf_dataframe["ai_lds"] = pmc_perf_dataframe["total_flops"] / lds_data

        # TODO: Draft
        def _calculate_roof(pmc_perf_dataframe, roofline_dataframe):
            cache_hierarchy = ["HBM", "L2", "L1", "LDS"]

            roofline_dataframe = roofline_dataframe.drop(
                columns=roofline_dataframe.filter(like="High").columns
            )
            roofline_dataframe = roofline_dataframe.drop(
                columns=roofline_dataframe.filter(like="Low").columns
            )

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

        pmc_perf_dataframe = self.data[0].copy(deep=True)
        roofline_dataframe = self.data[1].copy(deep=True)

        _calculate_metrics(pmc_perf_dataframe)
        _calculate_roof(pmc_perf_dataframe, roofline_dataframe)

        return pmc_perf_dataframe

    def plot_ncu_roofline(self, precisions=None, units=None):
        from bokeh.models import (
            CustomJS,
            CheckboxGroup,
            Button,
            LegendItem,
            HoverTool,
            ColumnDataSource,
            Panel,
            Tabs,
            MultiSelect,
            Div,
        )
        from bokeh.layouts import column, row
        from bokeh.plotting import figure, show, output_notebook

        if precisions is None:
            precisions = self.precisions
        if units is None:
            units = self.units

        roofs = {}
        kernels = {}
        # Group the roof and kernel points.
        for precision in precisions:
            roof_points = {}
            kernel_points = {}
            roof_performance = "{}_roof_performance".format(precision)
            achieved_performance = "{}_achieved_performance".format(precision)
            for unit in units:
                roof_AI = "{}_{}_roof_AI".format(precision, unit)
                achieved_AI = "{}_{}_achieved_AI".format(precision, unit)

                roof_points[unit] = [roof_performance, roof_AI]
                kernel_points[unit] = [achieved_performance, achieved_AI]
            roofs[precision] = roof_points
            kernels[precision] = kernel_points

        starting_point_AI = 0.01
        min_value = self.rooflines.filter(regex="AI")
        min_value = min_value[min_value != 0].min().min()
        if min_value != 0:
            if min_value < 1:
                starting_point_AI = 10 ** (math.floor(math.log10(min_value)) - 1)
            else:
                starting_point_AI = 10 ** (math.ceil(math.log10(min_value)) - 2)

        max_kernel_AI = self.rooflines.filter(regex="achieved_AI").max().max()
        ending_point_AI = 10 ** (len(str(int(max_kernel_AI))) + 2)

        for precision, unit_dict in roofs.items():
            roof_point_perf = (
                self.rooflines.filter(regex="{}_roof_perf*".format(precision, unit))
                .max()
                .max()
            )

            for unit, column_pairs in unit_dict.items():
                roof_point_AI = (
                    self.rooflines.filter(regex="{}_{}_roof_AI".format(precision, unit))
                    .max()
                    .max()
                )
                starting_point_perf = (
                    roof_point_perf / roof_point_AI
                ) * starting_point_AI
                roof_max = 10 ** (len(str(int(roof_point_AI))) + 2)

                if ending_point_AI < roof_max:
                    ending_point_AI = roof_max

                roofs[precision][unit].append([starting_point_AI, roof_point_AI])
                roofs[precision][unit].append([starting_point_perf, roof_point_perf])
                roofs[precision][unit].append([roof_point_AI, ending_point_AI])
                roofs[precision][unit].append([roof_point_perf, roof_point_perf])

        kernel_x = {}
        kernel_y = {}
        kernel_labels = self.rooflines["Kernel Name"].tolist()
        for precision, unit_dict in kernels.items():
            for unit, column_pairs in unit_dict.items():
                kernels[precision][unit].append(
                    self.rooflines[column_pairs[1]].tolist()
                )
                kernels[precision][unit].append(
                    self.rooflines[column_pairs[0]].tolist()
                )

        initial_legend_html_lines = (
            "<ul style='list-style-type: none; margin: 0; padding: 0;'>"
        )
        initial_legend_html_dots = (
            "<ul style='list-style-type: none; margin: 0; padding: 0;'>"
        )

        def _create_line_visibility_callback(
            line_renderers,
            line_legend_items,
            checkbox_group_lines,
            legend_div,
            initial_legend_html_lines,
        ):
            code = """
                for (var i = 0; i < line_renderers.length; i++) {
                    line_renderers[i].visible = cb_obj.active.includes(i);  
                }

                
                for (var i = 0; i < checkbox_group_lines.active.length; i++) {
                    var index = checkbox_group_lines.active[i];
                    var item = line_legend_items[index];  
                    var color = item.renderers[0].glyph.line_color || item.renderers[0].glyph.fill_color;  
                    var label = item.label['value'];  
                    initial_legend_html_lines += "<li><span style='color: " + color + ";'>●</span> " + label + "</li>"; 
                }
                legend_div.text = initial_legend_html_lines;  
            """
            return CustomJS(
                args=dict(
                    line_renderers=line_renderers,
                    line_legend_items=line_legend_items,
                    checkbox_group_lines=checkbox_group_lines,
                    legend_div=legend_div,
                    initial_legend_html_lines=initial_legend_html_lines,
                ),
                code=code,
            )

        def _create_dot_visibility_callback(
            dot_renderers,
            dot_legend_items,
            checkbox_group_labels,
            checkbox_group_dots,
            legend_div,
            initial_legend_html_dots,
        ):
            code = """
            var fullname_to_renderer = {};
            for (var i = 0; i < dot_renderers.length; i++) {
                var renderer = dot_renderers[i];
                var fullname = renderer.data_source.data['full_name'][0]; 
                fullname_to_renderer[fullname] = renderer;
            }

            var active_labels = cb_obj.active.map(function(index) {
                return cb_obj.labels[index];
            });
            console.log("Active labels:", active_labels);

            var active_indices_in_full_list = active_labels.map(function(label) {
                return checkbox_group_labels.indexOf(label); 
            });
            console.log("Active indices in full list:", active_indices_in_full_list);

            for (var i = 0; i < dot_renderers.length; i++) {
                dot_renderers[i].visible = false;
            }

            for (var i = 0; i < active_indices_in_full_list.length; i++) {
                var index = active_indices_in_full_list[i];
                if (index !== -1) { 
                    dot_renderers[index].visible = true;
                    var dotItem = dot_legend_items[index];
                    var dotLabel = dotItem.label['value'];
                    initial_legend_html_dots += "<li><span style='color: red;'>●</span> " + dotLabel + "</li>";
                }
            }
            legend_div.text = initial_legend_html_dots;
            """
            return CustomJS(
                args=dict(
                    dot_renderers=dot_renderers,
                    dot_legend_items=dot_legend_items,
                    checkbox_group_labels=checkbox_group_labels,
                    checkbox_group_dots=checkbox_group_dots,
                    legend_div=legend_div,
                    initial_legend_html_dots=initial_legend_html_dots,
                ),
                code=code,
            )

        output_notebook()

        # Data for the kernels and ceilings
        line_data = {
            "x": [[starting_point_AI, roof_point_AI, ending_point_AI]],
            "y": [[starting_point_perf, roof_point_perf, roof_point_perf]],
        }

        p = figure(
            title="Interactive Kernels and Ceilings",
            x_axis_label="X",
            y_axis_label="Y",
            sizing_mode="scale_both",
            y_axis_type="log",
            x_axis_type="log",
        )

        # Initially active dot indices
        initial_active_dots = [0]

        # Initialize lists for line renderers and legend items
        line_renderers = []
        line_legend_items = []
        line_labels = []
        # Iterate through roofs to create lines and their legends
        for precision, roof_units in roofs.items():
            for unit, data in roof_units.items():
                line_label_precision = "{} Precision ({})".format(
                    precision.upper(), unit
                )
                line_labels.append(line_label_precision)
                line_label_unit = "{} ({})".format(unit.upper(), precision)
                line_labels.append(line_label_unit)

                # inclined
                x_values_1 = data[2]
                y_values_1 = data[3]

                # Create a ColumnDataSource for the line
                line_source_inc = ColumnDataSource(
                    data=dict(
                        x=x_values_1,
                        y=y_values_1,
                        name=[line_label_precision] * len(x_values_1),
                    )
                )
                line_inc = p.line(
                    "x",
                    "y",
                    source=line_source_inc,
                    line_width=2,
                    color="blue",
                    legend_label=line_label_unit,
                    visible=True,
                )
                line_renderers.append(line_inc)
                # Create a legend item for the line
                legend_item_inc = LegendItem(
                    label=line_label_precision, renderers=[line_inc], visible=True
                )
                line_legend_items.append(legend_item_inc)

                # roof
                x_values_roof = data[4]
                y_values_roof = data[5]

                line_source_roof = ColumnDataSource(
                    data=dict(
                        x=x_values_roof,
                        y=y_values_roof,
                        name=[line_label_unit] * len(x_values_roof),
                    )
                )
                line_roof = p.line(
                    "x",
                    "y",
                    source=line_source_roof,
                    line_width=2,
                    color="green",
                    legend_label=line_label_precision,
                    visible=True,
                )
                line_renderers.append(line_roof)
                legend_item_roof = LegendItem(
                    label=line_label_unit, renderers=[line_roof], visible=True
                )
                line_legend_items.append(legend_item_roof)

        initial_active_lines = list(range(len(line_legend_items)))

        # Initialize lists for dot renderers and legend items
        dot_renderers = []
        dot_legend_items = []
        checkbox_labels = []
        full_name_counts = {}
        for precision, unit_dict in kernels.items():
            for unit, column_vals in unit_dict.items():
                dot_precision = precision
                dot_unit = unit
                for i, (dx, dy) in enumerate(zip(column_vals[2], column_vals[3])):
                    dot_name = kernel_labels[i % len(kernel_labels)]
                    base_full_name = f"{dot_name}-{dot_precision}-{dot_unit}"

                    # Check if the base full name exists in the dictionary, initialize if not
                    if base_full_name not in full_name_counts:
                        full_name_counts[base_full_name] = 0

                    # Increment the count for this base full name
                    full_name_counts[base_full_name] += 1

                    # Create a unique full name by appending the count
                    full_name = f"{base_full_name}-{full_name_counts[base_full_name]}"

                    checkbox_labels.append(full_name)

                    dot_source = ColumnDataSource(
                        data={
                            "x": [dx],
                            "y": [dy],
                            "name": [dot_name],
                            "precision": [dot_precision],
                            "unit": [dot_unit],
                            "full_name": [full_name],
                            "identifier": [full_name_counts[base_full_name]],
                        }
                    )

                    dot = p.circle(
                        "x",
                        "y",
                        source=dot_source,
                        size=10,
                        color="red",
                        legend_label=full_name,
                        visible=i in initial_active_dots,
                    )
                    dot_renderers.append(dot)

                    legend_item = LegendItem(
                        label=full_name,
                        renderers=[dot],
                        visible=i in initial_active_dots,
                    )
                    dot_legend_items.append(legend_item)

        # Add HoverTools
        hover_for_lines = HoverTool(
            tooltips=[("Name", "@name"), ("(x, y)", "($x, $y)")],
            renderers=line_renderers,
        )
        hover_for_dots = HoverTool(
            tooltips=[
                ("Name", "@name"),
                ("Precision", "@precision"),
                ("Unit", "@unit"),
                ("Identifier", "@identifier"),
                ("(x, y)", "(@x, @y)"),
            ],
            renderers=dot_renderers,
        )
        p.add_tools(hover_for_lines, hover_for_dots)

        # CheckboxGroups
        checkbox_group_lines = CheckboxGroup(
            labels=line_labels, active=initial_active_lines, inline=False
        )
        checkbox_group_dots = CheckboxGroup(
            labels=checkbox_labels, active=initial_active_dots, inline=False
        )

        # Clear the default legend and add custom legend items
        p.legend.items.clear()
        # Create a custom legend Div
        custom_legend_div_lines = Div(
            text=initial_legend_html_lines,
            width=200,
            height=100,
            style={"overflow-y": "auto", "border": "1px solid #e8e8e8"},
        )
        custom_legend_div_dots = Div(
            text=initial_legend_html_dots,
            width=200,
            height=100,
            style={"overflow-y": "auto", "border": "1px solid #e8e8e8"},
        )

        # Dropdown for filtering by precision, kernel, and unit.
        precision_filter_dropdown = MultiSelect(
            title="Precision Filter:", value=["All"], options=["All"] + precisions
        )
        kernel_filter_dropdown = MultiSelect(
            title="Kernel Filter:", value=["All"], options=["All"] + kernel_labels
        )
        unit_filter_dropdown = MultiSelect(
            title="Unit Filter:", value=["All"], options=["All"] + units
        )

        combined_filter_callback = CustomJS(
            args=dict(
                checkbox_group=checkbox_group_dots,
                all_labels=checkbox_labels,
                unit_filter=unit_filter_dropdown,
                precision_filter=precision_filter_dropdown,
                kernel_filter=kernel_filter_dropdown,
            ),
            code="""
            checkbox_group.labels = [];  

            var filtered_labels = [];
            var precision_selections = precision_filter.value.map(function(selection) { 
                return selection.toLowerCase();  
            });
            var kernel_selections = kernel_filter.value;
            var unit_selections = unit_filter.value.map(function(selection) { 
                return selection.toLowerCase(); 
            });

            for (var i = 0; i < all_labels.length; i++) {
                var label_parts = all_labels[i].split('-').map(function(part) { 
                    return part; 
                });

                var label_kernel = label_parts[0]; 
                var label_precision = label_parts.length > 1 ? label_parts[1] : "";  
                var label_unit = label_parts.length > 2 ? label_parts[2] : "";

                var matchesPrecision = (precision_selections.includes("all") || precision_selections.includes(label_precision));
                var matchesKernel = (kernel_selections.includes("All") || kernel_selections.includes(label_kernel));
                var matchesUnit = (unit_selections.includes("all") || unit_selections.includes(label_unit));

                if (matchesPrecision && matchesKernel & matchesUnit) {
                    filtered_labels.push(all_labels[i]);
                }
            }

            checkbox_group.labels = filtered_labels;  
            checkbox_group.active = []; 
        """,
        )

        # use the combined callback for the MultiSelect widgets
        precision_filter_dropdown.js_on_change("value", combined_filter_callback)
        kernel_filter_dropdown.js_on_change("value", combined_filter_callback)
        unit_filter_dropdown.js_on_change("value", combined_filter_callback)

        # callbacks for CheckboxGroups
        checkbox_group_lines.js_on_change(
            "active",
            _create_line_visibility_callback(
                line_renderers,
                line_legend_items,
                checkbox_group_lines,
                custom_legend_div_lines,
                initial_legend_html_lines,
            ),
        )
        checkbox_group_dots.js_on_change(
            "active",
            _create_dot_visibility_callback(
                dot_renderers,
                dot_legend_items,
                checkbox_labels,
                checkbox_group_dots,
                custom_legend_div_dots,
                initial_legend_html_dots,
            ),
        )

        # buttons to toggle visibility of CheckboxGroups
        toggle_lines_button = Button(label="Rooflines", button_type="primary")
        toggle_dots_button = Button(label="Kernel List", button_type="primary")

        # javascript for buttons
        toggle_lines_button.js_on_click(
            CustomJS(
                args=dict(checkbox_group=checkbox_group_lines),
                code="checkbox_group.visible = !checkbox_group.visible;",
            )
        )
        toggle_dots_button.js_on_click(
            CustomJS(
                args=dict(checkbox_group=checkbox_group_dots),
                code="checkbox_group.visible = !checkbox_group.visible;",
            )
        )

        # layout for the checkboxes and buttons
        controls_layout = column(
            toggle_lines_button,
            checkbox_group_lines,
            toggle_dots_button,
            unit_filter_dropdown,
            precision_filter_dropdown,
            kernel_filter_dropdown,
            checkbox_group_dots,
        )

        # create a panel for the plot and another for the controls
        plot_panel = Panel(child=p, title="Plot")
        controls_panel = Panel(child=controls_layout, title="Controls")

        # combine panels into tabs
        tabs = Tabs(tabs=[plot_panel, controls_panel])

        # add the tabs and legend divs to the layout
        layout = row(tabs, custom_legend_div_lines, custom_legend_div_dots)

        show(layout)
