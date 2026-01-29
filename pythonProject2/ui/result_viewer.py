"""
结果显示面板模块
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QLabel, QTableWidget, QTableWidgetItem, QTextEdit,
                             QHeaderView, QPushButton, QProgressBar, QSplitter,
                             QTreeWidget, QTreeWidgetItem, QComboBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QBrush
import numpy as np
from datetime import datetime


class ResultViewer(QWidget):
    """结果显示面板"""

    # 信号定义
    result_selected = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.results = []
        self.current_result = None
        self.init_ui()

    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)

        # 标题
        title_label = QLabel("检测结果")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(14)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # 分割器
        splitter = QSplitter(Qt.Vertical)

        # 结果概览表格
        self.table_results = QTableWidget()
        self.table_results.setColumnCount(7)
        self.table_results.setHorizontalHeaderLabels([
            "序号", "时间", "内圆坐标", "外圆坐标", "同心度", "偏心距", "状态"
        ])
        self.table_results.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_results.setSelectionBehavior(QTableWidget.SelectRows)
        self.table_results.itemSelectionChanged.connect(self.on_result_selected)

        # 详细结果展示
        detail_group = QGroupBox("详细结果")
        detail_layout = QVBoxLayout(detail_group)

        # 基本信息
        info_layout = QHBoxLayout()

        self.lbl_status = QLabel()
        self.lbl_status.setStyleSheet("QLabel { font-weight: bold; font-size: 16px; }")
        self.lbl_time = QLabel()
        self.lbl_time.setAlignment(Qt.AlignRight)

        info_layout.addWidget(self.lbl_status)
        info_layout.addWidget(self.lbl_time)
        detail_layout.addLayout(info_layout)

        # 结果表格
        self.table_detail = QTableWidget()
        self.table_detail.setColumnCount(2)
        self.table_detail.setRowCount(15)
        self.table_detail.setHorizontalHeaderLabels(["参数", "值"])
        self.table_detail.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_detail.verticalHeader().setVisible(False)

        # 设置行标签
        row_labels = [
            "内圆坐标 (x, y)",
            "内圆半径",
            "外圆坐标 (x, y)",
            "外圆半径",
            "像素偏心距",
            "实际偏心距 (mm)",
            "同心度 (‰)",
            "基准半径 (mm)",
            "像素到毫米",
            "公差 (mm)",
            "图像大小",
            "检测时间",
            "处理耗时",
            "算法版本",
            "备注"
        ]

        for i, label in enumerate(row_labels):
            item = QTableWidgetItem(label)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.table_detail.setItem(i, 0, item)

        detail_layout.addWidget(self.table_detail)

        # 文本结果
        self.txt_result = QTextEdit()
        self.txt_result.setReadOnly(True)
        self.txt_result.setMaximumHeight(150)
        detail_layout.addWidget(self.txt_result)

        # 统计信息
        stats_group = QGroupBox("统计信息")
        stats_layout = QVBoxLayout(stats_group)

        stats_grid = QHBoxLayout()

        self.lbl_total = QLabel("总检测数: 0")
        self.lbl_passed = QLabel("合格数: 0")
        self.lbl_failed = QLabel("不合格数: 0")
        self.lbl_pass_rate = QLabel("合格率: 0%")
        self.lbl_avg_time = QLabel("平均耗时: 0s")

        for widget in [self.lbl_total, self.lbl_passed, self.lbl_failed,
                       self.lbl_pass_rate, self.lbl_avg_time]:
            stats_grid.addWidget(widget)

        stats_layout.addLayout(stats_grid)

        # 进度条显示合格率
        self.progress_pass_rate = QProgressBar()
        self.progress_pass_rate.setRange(0, 100)
        self.progress_pass_rate.setValue(0)
        self.progress_pass_rate.setTextVisible(True)
        stats_layout.addWidget(self.progress_pass_rate)

        detail_layout.addWidget(stats_group)

        # 控制按钮（仅保留清空结果按钮）
        button_layout = QHBoxLayout()

        self.btn_clear = QPushButton("清空结果")
        self.btn_clear.clicked.connect(self.clear_results)

        button_layout.addWidget(self.btn_clear)
        detail_layout.addLayout(button_layout)

        # 添加到分割器
        splitter.addWidget(self.table_results)
        splitter.addWidget(detail_group)
        splitter.setSizes([300, 500])

        layout.addWidget(splitter)

        # 初始化显示
        self.clear_display()

    def update_results(self, result):
        """更新结果"""
        self.results.append(result)
        self.current_result = result
        self.update_table()
        self.update_detail(result)
        self.update_statistics()

    def update_table(self):
        """更新结果表格"""
        self.table_results.setRowCount(len(self.results))

        for i, result in enumerate(self.results):
            # 序号
            item_no = QTableWidgetItem(str(i + 1))
            item_no.setTextAlignment(Qt.AlignCenter)

            # 时间
            time_str = result.get('detection_time', 'N/A')
            item_time = QTableWidgetItem(time_str)

            # 内圆坐标
            inner_circle = result.get('inner_circle', {})
            inner_coord = f"({inner_circle.get('x', 0):.1f}, {inner_circle.get('y', 0):.1f})"
            item_inner = QTableWidgetItem(inner_coord)

            # 外圆坐标
            outer_circle = result.get('outer_circle', {})
            outer_coord = f"({outer_circle.get('x', 0):.1f}, {outer_circle.get('y', 0):.1f})"
            item_outer = QTableWidgetItem(outer_coord)

            # 同心度
            concentricity = result.get('concentricity', 0)
            item_conc = QTableWidgetItem(f"{concentricity:.3f}‰")

            # 偏心距
            eccentricity = result.get('eccentricity_mm', 0)
            item_ecc = QTableWidgetItem(f"{eccentricity:.3f}mm")

            # 状态
            is_qualified = result.get('is_qualified', False)
            status_text = "合格" if is_qualified else "不合格"
            item_status = QTableWidgetItem(status_text)

            # 设置颜色
            if is_qualified:
                item_status.setForeground(QBrush(QColor(0, 128, 0)))
            else:
                item_status.setForeground(QBrush(QColor(255, 0, 0)))

            # 添加到表格
            for j, item in enumerate([item_no, item_time, item_inner, item_outer,
                                      item_conc, item_ecc, item_status]):
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table_results.setItem(i, j, item)

        # 滚动到最后一行
        if self.results:
            self.table_results.scrollToBottom()

    def update_detail(self, result):
        """更新详细结果"""
        if result is None:
            return

        # 状态标签
        is_qualified = result.get('is_qualified', False)
        status_text = "✅ 合格" if is_qualified else "❌ 不合格"
        status_color = "green" if is_qualified else "red"
        self.lbl_status.setText(f'<font color="{status_color}">{status_text}</font>')

        # 时间标签
        time_str = result.get('detection_time', 'N/A')
        self.lbl_time.setText(f"检测时间: {time_str}")

        # 更新详细表格
        inner_circle = result.get('inner_circle', {})
        outer_circle = result.get('outer_circle', {})

        detail_values = [
            f"({inner_circle.get('x', 0):.1f}, {inner_circle.get('y', 0):.1f})",
            f"{inner_circle.get('radius', 0):.1f} px",
            f"({outer_circle.get('x', 0):.1f}, {outer_circle.get('y', 0):.1f})",
            f"{outer_circle.get('radius', 0):.1f} px",
            f"{result.get('eccentricity_px', 0):.2f} px",
            f"{result.get('eccentricity_mm', 0):.3f} mm",
            f"{result.get('concentricity', 0):.3f} ‰",
            f"{result.get('reference_radius_mm', 50.0):.1f} mm",
            f"{result.get('pixel_to_mm', 0.1):.3f} mm/px",
            f"{result.get('tolerance', 0.2):.2f} mm",
            result.get('image_size', 'N/A'),
            result.get('detection_time', 'N/A'),
            f"{result.get('processing_time', 0):.3f} s",
            result.get('algorithm_version', '1.0'),
            result.get('remark', '无')
        ]

        for i, value in enumerate(detail_values):
            item = QTableWidgetItem(str(value))
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)

            # 设置重要数据的颜色
            if i == 6:  # 同心度
                if is_qualified:
                    item.setForeground(QBrush(QColor(0, 128, 0)))
                else:
                    item.setForeground(QBrush(QColor(255, 0, 0)))
            elif i == 5:  # 偏心距
                if is_qualified:
                    item.setForeground(QBrush(QColor(0, 128, 0)))
                else:
                    item.setForeground(QBrush(QColor(255, 0, 0)))

            self.table_detail.setItem(i, 1, item)

        # 更新文本结果
        self.update_text_result(result)

    def update_text_result(self, result):
        """更新文本结果"""
        text = f"""
检测结果报告
============

检测时间: {result.get('detection_time', 'N/A')}
检测状态: {'合格' if result.get('is_qualified', False) else '不合格'}

几何参数:
--------
内圆坐标: ({result.get('inner_circle', {}).get('x', 0):.1f}, {result.get('inner_circle', {}).get('y', 0):.1f})
内圆半径: {result.get('inner_circle', {}).get('radius', 0):.1f} px
外圆坐标: ({result.get('outer_circle', {}).get('x', 0):.1f}, {result.get('outer_circle', {}).get('y', 0):.1f})
外圆半径: {result.get('outer_circle', {}).get('radius', 0):.1f} px

同心度分析:
----------
像素偏心距: {result.get('eccentricity_px', 0):.2f} px
实际偏心距: {result.get('eccentricity_mm', 0):.3f} mm
同心度值: {result.get('concentricity', 0):.3f} ‰
公差要求: ±{result.get('tolerance', 0.2):.2f} mm
判定结果: {'合格' if result.get('is_qualified', False) else '不合格'}

系统信息:
--------
图像尺寸: {result.get('image_size', 'N/A')}
处理耗时: {result.get('processing_time', 0):.3f} s
算法版本: {result.get('algorithm_version', '1.0')}
        """

        self.txt_result.setText(text)

    def update_statistics(self):
        """更新统计信息"""
        if not self.results:
            return

        total = len(self.results)
        passed = sum(1 for r in self.results if r.get('is_qualified', False))
        failed = total - passed
        pass_rate = (passed / total * 100) if total > 0 else 0

        # 计算平均处理时间
        total_time = sum(r.get('processing_time', 0) for r in self.results)
        avg_time = total_time / total if total > 0 else 0

        self.lbl_total.setText(f"总检测数: {total}")
        self.lbl_passed.setText(f"合格数: {passed}")
        self.lbl_failed.setText(f"不合格数: {failed}")
        self.lbl_pass_rate.setText(f"合格率: {pass_rate:.1f}%")
        self.lbl_avg_time.setText(f"平均耗时: {avg_time:.3f}s")

        # 更新进度条
        self.progress_pass_rate.setValue(int(pass_rate))

    def on_result_selected(self):
        """结果被选中时的处理"""
        selected_rows = self.table_results.selectedItems()
        if selected_rows:
            row = selected_rows[0].row()
            if row < len(self.results):
                self.current_result = self.results[row]
                self.update_detail(self.current_result)
                self.result_selected.emit(self.current_result)

    def clear_results(self):
        """清空结果"""
        self.results.clear()
        self.current_result = None
        self.clear_display()
        self.update_statistics()

    def clear_display(self):
        """清空显示"""
        self.table_results.setRowCount(0)
        self.lbl_status.setText("")
        self.lbl_time.setText("")

        # 清空详细表格
        for i in range(self.table_detail.rowCount()):
            self.table_detail.setItem(i, 1, QTableWidgetItem(""))

        self.txt_result.clear()

    def get_current_result(self):
        """获取当前选中的结果"""
        return self.current_result

    def get_all_results(self):
        """获取所有结果"""
        return self.results.copy()