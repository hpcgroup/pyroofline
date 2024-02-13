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
    def __init__(self, path=None, data=None, data_format=None):
        self.path = path
        self.data = data
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
                return Roofline(path, [pmc_perf_data, roofline_data], data_format="omniperf")
            else:
                raise ValueError(
                    "The directory should contain pmc_perf.csv and roofline.csv."
                )

    def analyze(self):
        if self.data_format == "ncu":
            return self.analyze_ncu()
        elif self.data_format == "omniperf":
            return self.analyze_omniperf()

    def analyze_ncu(self):
        """
        This function calculates all the required points to create a roofline plot.
        Specifically, it calculates double, single, half precisions and tensor core.
        It also calculates DRAM, L1, and L2 ceilings and achieved values.
        """
        dataframe = self.data.copy(deep=True)
        dataframe = dataframe[self.data["Kernel Name"].notna()]

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
                dataframe[
                    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained"
                ].astype(float)
                * 2
            )

            peakwork = max_sm_per_cycle * cycles_sm_per_sec

            # DRAM roof AI
            dataframe["single_dram_roof_AI"] = peakwork / dram_peaktraffic

            # DRAM Achieved
            fadd_per_cycle = dataframe[
                "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed"
            ].astype(float)
            fmul_per_cycle = dataframe[
                "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed"
            ].astype(float)
            ffma_per_cycle = (
                dataframe[
                    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed"
                ].astype(float)
                * 2
            )
            cycles_smsp_per_sec = dataframe[
                "smsp__cycles_elapsed.avg.per_second"
            ].astype(float)

            # DRAM, L1, and L2 Achieved Work
            achieved_performance = (
                fadd_per_cycle + fmul_per_cycle + ffma_per_cycle
            ) * cycles_smsp_per_sec

            dataframe["single_dram_achieved_AI"] = (
                achieved_performance / dram_cycles_per_sec_sum
            )

            dataframe["single_achieved_performance"] = achieved_performance
            dataframe["single_peak_performance"] = peakwork

            columns.extend(
                [
                    "single_dram_roof_AI",
                    "single_dram_achieved_AI",
                    "single_achieved_performance",
                    "single_peak_performance",
                ]
            )

            if l1:
                _calculate_l1(peakwork, achieved_performance, precision="single")
            if l2:
                _calculate_l2(peakwork, achieved_performance, precision="single")

        def _calculate_double():
            max_sm_per_cycle = (
                dataframe[
                    "sm__sass_thread_inst_executed_op_dfma_pred_on.sum.peak_sustained"
                ].astype(float)
                * 2
            )

            peakwork = max_sm_per_cycle * cycles_sm_per_sec

            # DRAM roof AI
            dataframe["double_dram_roof_AI"] = peakwork / dram_peaktraffic

            # DRAM Achieved
            dadd_per_cycle = dataframe[
                "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum.per_cycle_elapsed"
            ].astype(float)
            dmul_per_cycle = dataframe[
                "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum.per_cycle_elapsed"
            ].astype(float)
            dfma_per_cycle = (
                dataframe[
                    "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum.per_cycle_elapsed"
                ].astype(float)
                * 2
            )
            cycles_smsp_per_sec = dataframe[
                "smsp__cycles_elapsed.avg.per_second"
            ].astype(float)

            # DRAM, L1, and L2 Achieved Work
            achieved_performance = (
                dadd_per_cycle + dmul_per_cycle + dfma_per_cycle
            ) * cycles_smsp_per_sec

            dataframe["double_dram_achieved_AI"] = (
                achieved_performance / dram_cycles_per_sec_sum
            )

            dataframe["double_achieved_performance"] = achieved_performance
            dataframe["double_peak_performance"] = peakwork

            columns.extend(
                [
                    "double_dram_roof_AI",
                    "double_dram_achieved_AI",
                    "double_achieved_performance",
                    "double_peak_performance",
                ]
            )

            if l1:
                _calculate_l1(peakwork, achieved_performance, precision="double")
            if l2:
                _calculate_l2(peakwork, achieved_performance, precision="double")

        def _calculate_half():
            max_sm_per_cycle = (
                dataframe[
                    "sm__sass_thread_inst_executed_op_hfma_pred_on.sum.peak_sustained"
                ].astype(float)
                * 2
            )

            peakwork = max_sm_per_cycle * cycles_sm_per_sec

            # DRAM roof AI
            dataframe["half_dram_roof_AI"] = peakwork / dram_peaktraffic

            # DRAM Achieved
            hadd_per_cycle = dataframe[
                "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum.per_cycle_elapsed"
            ].astype(float)
            hmul_per_cycle = dataframe[
                "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum.per_cycle_elapsed"
            ].astype(float)
            hfma_per_cycle = (
                dataframe[
                    "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum.per_cycle_elapsed"
                ].astype(float)
                * 2
            )
            cycles_smsp_per_sec = dataframe[
                "smsp__cycles_elapsed.avg.per_second"
            ].astype(float)

            # DRAM, L1, and L2 Achieved Work
            achieved_performance = (
                hadd_per_cycle + hmul_per_cycle + hfma_per_cycle
            ) * cycles_smsp_per_sec

            dataframe["half_dram_achieved_AI"] = (
                achieved_performance / dram_cycles_per_sec_sum
            )

            dataframe["half_achieved_performance"] = achieved_performance
            dataframe["half_peak_performance"] = peakwork

            columns.extend(
                [
                    "half_dram_roof_AI",
                    "half_dram_achieved_AI",
                    "half_achieved_performance",
                    "half_peak_performance",
                ]
            )

            if l1:
                _calculate_l1(peakwork, achieved_performance, precision="half")
            if l2:
                _calculate_l2(peakwork, achieved_performance, precision="half")

        def _calculate_tensor():
            max_sm_per_cycle = (
                dataframe["sm__inst_executed_pipe_tensor.sum.peak_sustained"].astype(
                    float
                )
                * 512
            )

            peakwork = max_sm_per_cycle * cycles_sm_per_sec

            # DRAM roof AI
            dataframe["tensor_dram_roof_AI"] = peakwork / dram_peaktraffic

            # DRAM Achieved
            tensor_per_cycle = (
                dataframe[
                    "smsp__inst_executed_pipe_tensor.sum.per_cycle_elapsed"
                ].astype(float)
                * 512
            )
            cycles_smsp_per_sec = dataframe[
                "smsp__cycles_elapsed.avg.per_second"
            ].astype(float)

            # DRAM, L1, and L2 Achieved Work
            achieved_performance = tensor_per_cycle * cycles_smsp_per_sec

            dataframe["tensor_dram_achieved_AI"] = (
                achieved_performance / dram_cycles_per_sec_sum
            )

            dataframe["tensor_achieved_performance"] = achieved_performance
            dataframe["tensor_peak_performance"] = peakwork

            columns.extend(
                [
                    "tensor_dram_roof_AI",
                    "tensor_dram_achieved_AI",
                    "tensor_achieved_performance",
                    "tensor_peak_performance",
                ]
            )

            if l1:
                _calculate_l1(peakwork, achieved_performance, precision="tensor")
            if l2:
                _calculate_l2(peakwork, achieved_performance, precision="tensor")

        l1 = False
        l2 = False
        if "l1tex__t_bytes.sum.peak_sustained" in dataframe.columns:
            l1 = True
        if "lts__t_bytes.sum.peak_sustained" in dataframe.columns:
            l2 = True

        cycles_sm_per_sec = dataframe["sm__cycles_elapsed.avg.per_second"].astype(float)

        # DRAM Roofline
        dram_bytes_per_cycle = dataframe["dram__bytes.sum.peak_sustained"].astype(float)
        dram_cycles_per_sec = dataframe["dram__cycles_elapsed.avg.per_second"].astype(
            float
        )

        dram_peaktraffic = dram_bytes_per_cycle * dram_cycles_per_sec

        # DRAM Achieved Traffic
        dram_cycles_per_sec_sum = dataframe["dram__bytes.sum.per_second"].astype(float)

        # L1 Roofline
        if l1:
            l1_bytes_per_cycle = dataframe["l1tex__t_bytes.sum.peak_sustained"].astype(
                float
            )
            l1_cycles_per_sec = dataframe[
                "l1tex__cycles_elapsed.avg.per_second"
            ].astype(float)

            l1_peaktraffic = l1_bytes_per_cycle * l1_cycles_per_sec

            # L1 Achieved Traffic
            l1_cycles_per_sec_sum = dataframe["l1tex__t_bytes.sum.per_second"].astype(
                float
            )

        # L2 Roofline
        if l2:
            l2_bytes_per_cycle = dataframe["lts__t_bytes.sum.peak_sustained"].astype(
                float
            )
            l2_cycles_per_sec = dataframe["lts__cycles_elapsed.avg.per_second"].astype(
                float
            )

            l2_peaktraffic = l2_bytes_per_cycle * l2_cycles_per_sec

            # L2 Achieved Traffic
            l2_cycles_per_sec_sum = dataframe["lts__t_bytes.sum.per_second"].astype(
                float
            )

        columns = ["Kernel Name"]
        _calculate_single()

        _calculate_double()

        if (
            "sm__sass_thread_inst_executed_op_hfma_pred_on.sum.peak_sustained"
            in dataframe.columns
        ):
            _calculate_half()

        if "sm__inst_executed_pipe_tensor.sum.peak_sustained" in dataframe.columns:
            _calculate_tensor()

        dataframe = dataframe[columns]

        return dataframe

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
                + pmc_perf_dataframe["mfma_flops_f16"] 
                + pmc_perf_dataframe["mfma_flops_bf16"] 
                + pmc_perf_dataframe["mfma_flops_f32"] 
                + pmc_perf_dataframe["mfma_flops_f64"]
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

    # TODO: Draft
    def plot_ncu(
        self,
        dataframe,
        precisions=["single", "double"],
        types=["dram"],
        kernels=None,
        summary=False,
    ):
        """
        precision: list of precisions: single, double, half, tensor
        list of hardware options: dram, l1, l2
        """
        start_AI = 0.001

        if summary:
            dataframe = dataframe.groupby("Kernel Name", as_index=False).agg(np.mean)

        columns = list(dataframe.filter(regex="single_dram|double_dram").columns)
        points_achieved = []
        points_peak = []
        max_values = {}
        for precision in precisions:
            achieved_performance = "{}_achieved_performance".format(precision)
            peak_performance = "{}_peak_performance".format(precision)
            max_values[peak_performance] = dataframe[peak_performance].max()
            for type in types:
                achieved_type = "{}_{}_achieved_AI".format(precision, type)
                roof_type = "{}_{}_roof_AI".format(precision, type)
                starting_point = "{}_{}_start_performance".format(precision, type)
                dataframe[starting_point] = (
                    dataframe[peak_performance] / dataframe[roof_type]
                ) * start_AI
                points_achieved.append([achieved_performance, achieved_type])
                points_peak.append([peak_performance, roof_type])

        # for point in points:
        #     dataframe[point].iloc[2]
        return (points_peak, points_achieved)


    # Just an example.
    # We need to update this function and use real roofline values. 
    def plot_roofline(self):
        from bokeh.models import CustomJS, CheckboxGroup, Button, LegendItem, HoverTool, ColumnDataSource
        from bokeh.layouts import column
        from bokeh.plotting import figure, show, output_notebook

        # callback for checkboxGroups to toggle line and dot visibility and update legends
        def _create_visibility_callback(renderers, legend_items):
            code = """
                for (var i = 0; i < renderers.length; i++) {
                    renderers[i].visible = cb_obj.active.includes(i);
                    // Update legend item visibility
                    legend_items[i].visible = cb_obj.active.includes(i);  
                }
            """
            return CustomJS(args=dict(renderers=renderers, legend_items=legend_items), code=code)

        output_notebook()

        # Data for the kernels and ceilings
        # Example: x: [[x1, x2, x3], [y1, y2, y3]]. This will create a single line.
        line_data = {'x': [[0, 10, 30], [0, 5, 30], [0, 15, 30]], 'y': [[0, 10, 10], [0, 5, 5], [0, 15, 15]]}
        # Example: x: [[x1]], y: [[y1]]. This will create a single dot.
        dot_data = {'x': [[5], [6], [12]], 'y': [[3], [2], [8]]}

        line_labels = ["DRAM", "L2", "L1"]
        dot_labels = ["Kernel 1", "Kernel 2", "Kernel 3"]

        p = figure(title="Interactive Kernels and Ceilings", x_axis_label='X', y_axis_label='Y')

        # Add lines and dots with legends to the plot
        line_renderers = []
        dot_renderers = []

        for i, (lx, ly) in enumerate(zip(line_data['x'], line_data['y'])):
            line_source = ColumnDataSource(data=dict(x=lx, y=ly, name=[line_labels[i]]*len(lx)))
            line = p.line('x', 'y', source=line_source, line_width=2, color="blue", legend_label=line_labels[i], visible=True)
            line_renderers.append(line)

        for i, (dx, dy) in enumerate(zip(dot_data['x'], dot_data['y'])):
            dot_source = ColumnDataSource(data=dict(x=dx, y=dy, name=[dot_labels[i]]))
            dot = p.circle('x', 'y', source=dot_source, size=10, color="red", legend_label=dot_labels[i], visible=True)    
            dot_renderers.append(dot)


        # add HoverTool to display name and fixed values for lines and dots
        hover_for_lines = HoverTool(tooltips=[("Name", "@name"), ("(x, y)", "($x, $y)")], renderers=line_renderers)
        hover_for_dots = HoverTool(tooltips=[("Name", "@name"), ("(x, y)", "(@x, @y)")], renderers=dot_renderers)

        p.add_tools(hover_for_lines, hover_for_dots)

        # create legend items for lines and dots to be controlled separately
        line_legend_items = [LegendItem(label=label, renderers=[renderer]) for label, renderer in zip(line_labels, line_renderers)]
        dot_legend_items = [LegendItem(label=label, renderers=[renderer]) for label, renderer in zip(dot_labels, dot_renderers)]

        # clear the default legend and add our custom legend items
        p.legend.items.clear()
        p.legend.items.extend(line_legend_items + dot_legend_items)

        # create checkboxGroup widgets for lines and dots
        checkbox_group_lines = CheckboxGroup(labels=line_labels, active=[0, 1, 2], inline=False)
        checkbox_group_dots = CheckboxGroup(labels=dot_labels, active=[0, 1, 2], inline=False)

        # attach callbacks to CheckboxGroups for controlling visibility and legends
        checkbox_group_lines.js_on_change('active', _create_visibility_callback(line_renderers, line_legend_items))
        checkbox_group_dots.js_on_change('active', _create_visibility_callback(dot_renderers, dot_legend_items))

        # create buttons for the CheckboxGroups
        toggle_lines_button = Button(label="Ceilings", button_type="primary")
        toggle_dots_button = Button(label="Kernels", button_type="primary")

        # click on buttons for the visibility of the CheckboxGroups
        toggle_lines_button.js_on_click(CustomJS(args=dict(checkbox_group=checkbox_group_lines), code="""
            checkbox_group.visible = !checkbox_group.visible;
        """))
        toggle_dots_button.js_on_click(CustomJS(args=dict(checkbox_group=checkbox_group_dots), code="""
            checkbox_group.visible = !checkbox_group.visible;
        """))

        # layout with buttons, CheckboxGroups, and plot
        layout = column(toggle_lines_button, checkbox_group_lines, toggle_dots_button, checkbox_group_dots, p)

        # Display the layout in the notebook
        show(layout)

