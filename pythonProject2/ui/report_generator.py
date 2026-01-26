"""
报告生成器模块
"""

import os
import json
from datetime import datetime
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QLabel, QTextEdit, QPushButton, QFileDialog,
                             QComboBox, QCheckBox, QSpinBox, QFormLayout,
                             QProgressBar, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QFont
import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import cv2


class ReportGenerator(QWidget):
    """报告生成器"""

    # 信号定义
    report_generated = pyqtSignal(str)
    report_error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.report_data = None
        self.init_ui()

    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)

        # 标题
        title_label = QLabel("检测报告生成")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(16)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # 报告配置区域
        config_group = QGroupBox("报告配置")
        config_layout = QFormLayout(config_group)

        # 报告类型选择
        self.combo_report_type = QComboBox()
        self.combo_report_type.addItems([
            "单次检测报告",
            "批量检测报告",
            "统计分析报告",
            "完整检测报告"
        ])
        config_layout.addRow("报告类型:", self.combo_report_type)

        # 报告格式选择
        self.combo_format = QComboBox()
        self.combo_format.addItems(["PDF", "Excel", "HTML", "Word"])
        config_layout.addRow("输出格式:", self.combo_format)

        # 包含内容选项
        self.chk_include_images = QCheckBox("包含检测图像")
        self.chk_include_images.setChecked(True)
        self.chk_include_charts = QCheckBox("包含统计图表")
        self.chk_include_charts.setChecked(True)
        self.chk_include_raw_data = QCheckBox("包含原始数据")
        self.chk_include_raw_data.setChecked(True)

        config_layout.addRow("", self.chk_include_images)
        config_layout.addRow("", self.chk_include_charts)
        config_layout.addRow("", self.chk_include_raw_data)

        # 报告标题
        self.txt_report_title = QTextEdit()
        self.txt_report_title.setMaximumHeight(50)
        self.txt_report_title.setText("机械零件同心度检测报告")
        config_layout.addRow("报告标题:", self.txt_report_title)

        # 公司信息
        self.txt_company_info = QTextEdit()
        self.txt_company_info.setMaximumHeight(80)
        self.txt_company_info.setText("西南科技大学\n信息与控制工程学院\n物联网工程专业")
        config_layout.addRow("公司信息:", self.txt_company_info)

        # 备注信息
        self.txt_remarks = QTextEdit()
        self.txt_remarks.setMaximumHeight(100)
        self.txt_remarks.setText("本报告由机械零件同心度检测系统自动生成。")
        config_layout.addRow("备注:", self.txt_remarks)

        layout.addWidget(config_group)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # 按钮区域
        button_layout = QHBoxLayout()

        self.btn_generate = QPushButton("生成报告")
        self.btn_generate.clicked.connect(self.generate_report)
        self.btn_generate.setFixedHeight(40)

        self.btn_preview = QPushButton("预览报告")
        self.btn_preview.clicked.connect(self.preview_report)
        self.btn_preview.setFixedHeight(40)

        self.btn_export = QPushButton("导出数据")
        self.btn_export.clicked.connect(self.export_data)
        self.btn_export.setFixedHeight(40)

        button_layout.addWidget(self.btn_generate)
        button_layout.addWidget(self.btn_preview)
        button_layout.addWidget(self.btn_export)

        layout.addLayout(button_layout)

        # 状态标签
        self.lbl_status = QLabel("就绪")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_status)

    def set_report_data(self, data):
        """设置报告数据"""
        self.report_data = data

    def generate_report(self):
        """生成报告"""
        if self.report_data is None:
            QMessageBox.warning(self, "警告", "没有可生成报告的数据！")
            return

        # 选择保存路径
        default_name = f"concentricity_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        file_filter = ""

        if self.combo_format.currentText() == "PDF":
            file_filter = "PDF文件 (*.pdf)"
            default_name += ".pdf"
        elif self.combo_format.currentText() == "Excel":
            file_filter = "Excel文件 (*.xlsx)"
            default_name += ".xlsx"
        elif self.combo_format.currentText() == "HTML":
            file_filter = "HTML文件 (*.html)"
            default_name += ".html"

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存报告",
            default_name,
            file_filter
        )

        if file_path:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.lbl_status.setText("正在生成报告...")

            try:
                if self.combo_format.currentText() == "PDF":
                    self.generate_pdf_report(file_path)
                elif self.combo_format.currentText() == "Excel":
                    self.generate_excel_report(file_path)
                elif self.combo_format.currentText() == "HTML":
                    self.generate_html_report(file_path)

                self.progress_bar.setValue(100)
                self.lbl_status.setText(f"报告已生成: {os.path.basename(file_path)}")
                self.report_generated.emit(file_path)

                QMessageBox.information(self, "成功", "报告生成成功！")

            except Exception as e:
                self.lbl_status.setText(f"生成报告失败: {str(e)}")
                self.report_error.emit(str(e))
                QMessageBox.critical(self, "错误", f"生成报告失败: {str(e)}")

            finally:
                self.progress_bar.setVisible(False)

    def generate_pdf_report(self, file_path):
        """生成PDF报告"""
        # 创建PDF文档
        doc = SimpleDocTemplate(
            file_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )

        # 获取样式
        styles = getSampleStyleSheet()

        # 创建自定义样式
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1  # 居中
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=12
        )

        normal_style = styles['Normal']

        # 构建报告内容
        story = []

        # 标题
        title = self.txt_report_title.toPlainText()
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 0.2 * inch))

        # 公司信息
        company_info = self.txt_company_info.toPlainText()
        story.append(Paragraph("检测单位:", heading_style))
        for line in company_info.split('\n'):
            story.append(Paragraph(line, normal_style))
        story.append(Spacer(1, 0.2 * inch))

        # 报告信息
        story.append(Paragraph("报告信息:", heading_style))

        report_info_data = [
            ["生成时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["报告类型", self.combo_report_type.currentText()],
            ["检测系统", "机械零件同心度检测系统 v1.0"],
            ["检测人员", "自动生成"],
            ["审核状态", "已审核"]
        ]

        report_info_table = Table(report_info_data, colWidths=[2 * inch, 3 * inch])
        report_info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        story.append(report_info_table)
        story.append(Spacer(1, 0.3 * inch))

        # 检测结果汇总
        story.append(Paragraph("检测结果汇总:", heading_style))

        if isinstance(self.report_data, list):
            # 批量报告
            summary_data = [
                ["统计项", "数值"],
                ["总检测数", str(len(self.report_data))],
                ["合格数", str(sum(1 for d in self.report_data if d.get('is_qualified', False)))],
                ["不合格数", str(sum(1 for d in self.report_data if not d.get('is_qualified', False)))],
                ["合格率",
                 f"{sum(1 for d in self.report_data if d.get('is_qualified', False)) / len(self.report_data) * 100:.2f}%"],
                ["平均同心度",
                 f"{sum(d.get('concentricity', 0) for d in self.report_data) / len(self.report_data):.3f}‰"],
                ["最大偏心距", f"{max(d.get('eccentricity_mm', 0) for d in self.report_data):.3f} mm"],
                ["最小偏心距", f"{min(d.get('eccentricity_mm', 0) for d in self.report_data):.3f} mm"]
            ]
        else:
            # 单次报告
            summary_data = [
                ["检测项目", "检测结果", "判定标准", "结论"],
                ["同心度", f"{self.report_data.get('concentricity', 0):.3f}‰",
                 f"≤{self.report_data.get('tolerance', 0.2):.2f}‰",
                 "合格" if self.report_data.get('is_qualified', False) else "不合格"],
                ["偏心距", f"{self.report_data.get('eccentricity_mm', 0):.3f} mm",
                 f"≤{self.report_data.get('tolerance', 0.2):.2f} mm",
                 "合格" if self.report_data.get('is_qualified', False) else "不合格"],
                ["内圆坐标",
                 f"({self.report_data.get('inner_circle', {}).get('x', 0):.1f}, {self.report_data.get('inner_circle', {}).get('y', 0):.1f})",
                 "-", "-"],
                ["外圆坐标",
                 f"({self.report_data.get('outer_circle', {}).get('x', 0):.1f}, {self.report_data.get('outer_circle', {}).get('y', 0):.1f})",
                 "-", "-"],
                ["检测时间", self.report_data.get('detection_time', 'N/A'), "-", "-"],
                ["处理耗时", f"{self.report_data.get('processing_time', 0):.3f} s", "-", "-"]
            ]

        summary_table = Table(summary_data, colWidths=[1.5 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('TEXTCOLOR', (3, 1), (3, 2),
             colors.green if self.report_data.get('is_qualified', False) else colors.red)
        ]))

        story.append(summary_table)
        story.append(Spacer(1, 0.3 * inch))

        # 详细数据（如果选择包含原始数据）
        if self.chk_include_raw_data.isChecked() and isinstance(self.report_data, list):
            story.append(Paragraph("详细检测数据:", heading_style))

            # 创建详细数据表
            detailed_data = [["序号", "时间", "内圆坐标", "外圆坐标", "同心度(‰)", "偏心距(mm)", "状态"]]

            for i, data in enumerate(self.report_data, 1):
                detailed_data.append([
                    str(i),
                    data.get('detection_time', 'N/A'),
                    f"({data.get('inner_circle', {}).get('x', 0):.1f}, {data.get('inner_circle', {}).get('y', 0):.1f})",
                    f"({data.get('outer_circle', {}).get('x', 0):.1f}, {data.get('outer_circle', {}).get('y', 0):.1f})",
                    f"{data.get('concentricity', 0):.3f}",
                    f"{data.get('eccentricity_mm', 0):.3f}",
                    "合格" if data.get('is_qualified', False) else "不合格"
                ])

            detailed_table = Table(detailed_data, colWidths=[0.5 * inch, 1.2 * inch, 1.2 * inch,
                                                             1.2 * inch, 1 * inch, 1 * inch, 0.8 * inch])
            detailed_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            story.append(detailed_table)
            story.append(Spacer(1, 0.3 * inch))

        # 备注
        remarks = self.txt_remarks.toPlainText()
        if remarks:
            story.append(Paragraph("备注说明:", heading_style))
            story.append(Paragraph(remarks, normal_style))
            story.append(Spacer(1, 0.2 * inch))

        # 签名区域
        story.append(Paragraph("签字确认:", heading_style))

        signature_data = [
            ["检测人员", "", "日期", ""],
            ["审核人员", "", "日期", ""],
            ["批准人员", "", "日期", ""]
        ]

        signature_table = Table(signature_data, colWidths=[1.5 * inch, 2 * inch, 1.5 * inch, 2 * inch])
        signature_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        story.append(signature_table)

        # 生成PDF
        doc.build(story)

    def generate_excel_report(self, file_path):
        """生成Excel报告"""
        if isinstance(self.report_data, list):
            # 批量数据
            data_list = []
            for i, data in enumerate(self.report_data, 1):
                data_list.append({
                    '序号': i,
                    '检测时间': data.get('detection_time', ''),
                    '内圆X': data.get('inner_circle', {}).get('x', 0),
                    '内圆Y': data.get('inner_circle', {}).get('y', 0),
                    '内圆半径': data.get('inner_circle', {}).get('radius', 0),
                    '外圆X': data.get('outer_circle', {}).get('x', 0),
                    '外圆Y': data.get('outer_circle', {}).get('y', 0),
                    '外圆半径': data.get('outer_circle', {}).get('radius', 0),
                    '像素偏心距': data.get('eccentricity_px', 0),
                    '实际偏心距(mm)': data.get('eccentricity_mm', 0),
                    '同心度(‰)': data.get('concentricity', 0),
                    '公差(mm)': data.get('tolerance', 0.2),
                    '状态': '合格' if data.get('is_qualified', False) else '不合格',
                    '处理耗时(s)': data.get('processing_time', 0),
                    '图像尺寸': data.get('image_size', '')
                })

            df = pd.DataFrame(data_list)

            # 添加统计信息
            stats_df = pd.DataFrame([{
                '总检测数': len(self.report_data),
                '合格数': sum(1 for d in self.report_data if d.get('is_qualified', False)),
                '不合格数': sum(1 for d in self.report_data if not d.get('is_qualified', False)),
                '合格率(%)': sum(1 for d in self.report_data if d.get('is_qualified', False)) / len(
                    self.report_data) * 100,
                '平均同心度(‰)': sum(d.get('concentricity', 0) for d in self.report_data) / len(self.report_data),
                '最大偏心距(mm)': max(d.get('eccentricity_mm', 0) for d in self.report_data),
                '最小偏心距(mm)': min(d.get('eccentricity_mm', 0) for d in self.report_data)
            }])

            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='详细数据', index=False)
                stats_df.to_excel(writer, sheet_name='统计汇总', index=False)

                # 添加报告信息
                workbook = writer.book
                worksheet = workbook.create_sheet('报告信息')

                worksheet.append(['机械零件同心度检测报告'])
                worksheet.append(['生成时间', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                worksheet.append(['报告类型', self.combo_report_type.currentText()])
                worksheet.append(['检测系统', '机械零件同心度检测系统 v1.0'])
                worksheet.append([''])
                worksheet.append(['检测单位', self.txt_company_info.toPlainText().replace('\n', ' ')])
                worksheet.append([''])
                worksheet.append(['备注', self.txt_remarks.toPlainText()])

        else:
            # 单次数据
            data_dict = {
                '检测项目': ['同心度', '偏心距', '内圆坐标', '外圆坐标',
                             '检测时间', '处理耗时', '图像尺寸', '状态'],
                '检测结果': [
                    f"{self.report_data.get('concentricity', 0):.3f}‰",
                    f"{self.report_data.get('eccentricity_mm', 0):.3f} mm",
                    f"({self.report_data.get('inner_circle', {}).get('x', 0):.1f}, {self.report_data.get('inner_circle', {}).get('y', 0):.1f})",
                    f"({self.report_data.get('outer_circle', {}).get('x', 0):.1f}, {self.report_data.get('outer_circle', {}).get('y', 0):.1f})",
                    self.report_data.get('detection_time', 'N/A'),
                    f"{self.report_data.get('processing_time', 0):.3f} s",
                    self.report_data.get('image_size', 'N/A'),
                    '合格' if self.report_data.get('is_qualified', False) else '不合格'
                ],
                '判定标准': [
                    f"≤{self.report_data.get('tolerance', 0.2):.2f}‰",
                    f"≤{self.report_data.get('tolerance', 0.2):.2f} mm",
                    '-', '-', '-', '-', '-', '-'
                ]
            }

            df = pd.DataFrame(data_dict)
            df.to_excel(file_path, index=False)

    def generate_html_report(self, file_path):
        """生成HTML报告"""
        if isinstance(self.report_data, list):
            # 批量报告
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>{self.txt_report_title.toPlainText()}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .header {{ text-align: center; margin-bottom: 40px; }}
                    .title {{ font-size: 24px; font-weight: bold; margin-bottom: 20px; }}
                    .section {{ margin-bottom: 30px; }}
                    .section-title {{ font-size: 18px; font-weight: bold; margin-bottom: 10px; border-bottom: 2px solid #333; padding-bottom: 5px; }}
                    table {{ width: 100%; border-collapse: collapse; margin-bottom: 15px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
                    th {{ background-color: #4CAF50; color: white; }}
                    tr:nth-child(even) {{ background-color: #f2f2f2; }}
                    .passed {{ color: green; font-weight: bold; }}
                    .failed {{ color: red; font-weight: bold; }}
                    .stats {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <div class="title">{self.txt_report_title.toPlainText()}</div>
                    <div>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
                    <div>检测单位: {self.txt_company_info.toPlainText().replace(chr(10), '<br>')}</div>
                </div>

                <div class="section">
                    <div class="section-title">统计汇总</div>
                    <div class="stats">
                        总检测数: {len(self.report_data)}<br>
                        合格数: {sum(1 for d in self.report_data if d.get('is_qualified', False))}<br>
                        不合格数: {sum(1 for d in self.report_data if not d.get('is_qualified', False))}<br>
                        合格率: {sum(1 for d in self.report_data if d.get('is_qualified', False)) / len(self.report_data) * 100:.2f}%<br>
                    </div>
                </div>

                <div class="section">
                    <div class="section-title">详细检测数据</div>
                    <table>
                        <thead>
                            <tr>
                                <th>序号</th>
                                <th>检测时间</th>
                                <th>内圆坐标</th>
                                <th>外圆坐标</th>
                                <th>同心度(‰)</th>
                                <th>偏心距(mm)</th>
                                <th>状态</th>
                            </tr>
                        </thead>
                        <tbody>
            """

            for i, data in enumerate(self.report_data, 1):
                status_class = "passed" if data.get('is_qualified', False) else "failed"
                status_text = "合格" if data.get('is_qualified', False) else "不合格"

                html_content += f"""
                            <tr>
                                <td>{i}</td>
                                <td>{data.get('detection_time', 'N/A')}</td>
                                <td>({data.get('inner_circle', {{}}).get('x', 0):.1f}, {data.get('inner_circle', {{}}).get('y', 0):.1f})</td>
                                <td>({data.get('outer_circle', {{}}).get('x', 0):.1f}, {data.get('outer_circle', {{}}).get('y', 0):.1f})</td>
                                <td>{data.get('concentricity', 0):.3f}</td>
                                <td>{data.get('eccentricity_mm', 0):.3f}</td>
                                <td class="{status_class}">{status_text}</td>
                            </tr>
                """

            html_content += """
                        </tbody>
                    </table>
                </div>

                <div class="section">
                    <div class="section-title">备注</div>
                    <p>""" + self.txt_remarks.toPlainText().replace(chr(10), '<br>') + """</p>
                </div>

                <div class="section">
                    <div class="section-title">签字确认</div>
                    <table>
                        <tr>
                            <th>检测人员</th>
                            <th>审核人员</th>
                            <th>批准人员</th>
                        </tr>
                        <tr>
                            <td style="height: 50px;"></td>
                            <td style="height: 50px;"></td>
                            <td style="height: 50px;"></td>
                        </tr>
                        <tr>
                            <td>日期：</td>
                            <td>日期：</td>
                            <td>日期：</td>
                        </tr>
                    </table>
                </div>
            </body>
            </html>
            """
        else:
            # 单次报告
            status_class = "passed" if self.report_data.get('is_qualified', False) else "failed"
            status_text = "合格" if self.report_data.get('is_qualified', False) else "不合格"

            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>{self.txt_report_title.toPlainText()}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .header {{ text-align: center; margin-bottom: 40px; }}
                    .title {{ font-size: 24px; font-weight: bold; margin-bottom: 20px; }}
                    .section {{ margin-bottom: 30px; }}
                    .section-title {{ font-size: 18px; font-weight: bold; margin-bottom: 10px; border-bottom: 2px solid #333; padding-bottom: 5px; }}
                    table {{ width: 100%; border-collapse: collapse; margin-bottom: 15px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #4CAF50; color: white; }}
                    .result {{ font-weight: bold; }}
                    .{status_class} {{ color: {'green' if self.report_data.get('is_qualified', False) else 'red'}; }}
                    .info {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <div class="title">{self.txt_report_title.toPlainText()}</div>
                    <div>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
                    <div>检测单位: {self.txt_company_info.toPlainText().replace(chr(10), '<br>')}</div>
                </div>

                <div class="section">
                    <div class="section-title">检测结果</div>
                    <div class="info {status_class}">
                        检测状态: {status_text}<br>
                        检测时间: {self.report_data.get('detection_time', 'N/A')}<br>
                        处理耗时: {self.report_data.get('processing_time', 0):.3f} 秒
                    </div>
                </div>

                <div class="section">
                    <div class="section-title">几何参数</div>
                    <table>
                        <tr>
                            <th>参数</th>
                            <th>数值</th>
                        </tr>
                        <tr>
                            <td>内圆坐标</td>
                            <td>({self.report_data.get('inner_circle', {{}}).get('x', 0):.1f}, {self.report_data.get('inner_circle', {{}}).get('y', 0):.1f})</td>
                        </tr>
                        <tr>
                            <td>内圆半径</td>
                            <td>{self.report_data.get('inner_circle', {{}}).get('radius', 0):.1f} px</td>
                        </tr>
                        <tr>
                            <td>外圆坐标</td>
                            <td>({self.report_data.get('outer_circle', {{}}).get('x', 0):.1f}, {self.report_data.get('outer_circle', {{}}).get('y', 0):.1f})</td>
                        </tr>
                        <tr>
                            <td>外圆半径</td>
                            <td>{self.report_data.get('outer_circle', {{}}).get('radius', 0):.1f} px</td>
                        </tr>
                    </table>
                </div>

                <div class="section">
                    <div class="section-title">同心度分析</div>
                    <table>
                        <tr>
                            <th>检测项目</th>
                            <th>检测结果</th>
                            <th>判定标准</th>
                            <th>结论</th>
                        </tr>
                        <tr>
                            <td>同心度</td>
                            <td>{self.report_data.get('concentricity', 0):.3f}‰</td>
                            <td>≤{self.report_data.get('tolerance', 0.2):.2f}‰</td>
                            <td class="{status_class}">{status_text}</td>
                        </tr>
                        <tr>
                            <td>偏心距</td>
                            <td>{self.report_data.get('eccentricity_mm', 0):.3f} mm</td>
                            <td>≤{self.report_data.get('tolerance', 0.2):.2f} mm</td>
                            <td class="{status_class}">{status_text}</td>
                        </tr>
                    </table>
                </div>

                <div class="section">
                    <div class="section-title">备注</div>
                    <p>""" + self.txt_remarks.toPlainText().replace(chr(10), '<br>') + """</p>
                </div>
            </body>
            </html>
            """

        # 写入文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def preview_report(self):
        """预览报告"""
        QMessageBox.information(self, "预览", "报告预览功能开发中...")

    def export_data(self):
        """导出数据"""
        if self.report_data is None:
            QMessageBox.warning(self, "警告", "没有可导出的数据！")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出数据",
            f"concentricity_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON文件 (*.json)"
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.report_data, f, ensure_ascii=False, indent=2)

                QMessageBox.information(self, "成功", "数据导出成功！")

            except Exception as e:
                QMessageBox.critical(self, "错误", f"数据导出失败: {str(e)}")