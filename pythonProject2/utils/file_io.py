"""
文件读写工具模块
处理图像、参数、结果的读写操作
"""

import os
import json
import csv
import yaml
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import cv2
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as ReportImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
import logging


class DataExporter:
    """数据导出器，支持多种格式导出"""

    def __init__(self, output_dir: str = "results"):
        """
        初始化数据导出器

        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        self.ensure_directory_exists(output_dir)

    @staticmethod
    def ensure_directory_exists(directory: str):
        """确保目录存在"""
        os.makedirs(directory, exist_ok=True)

    def export_csv(self, data: List[Dict[str, Any]], filename: str = None) -> str:
        """
        导出数据为CSV文件

        Args:
            data: 数据列表，每个元素是一个字典
            filename: 文件名，如果为None则自动生成

        Returns:
            导出的文件路径
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{timestamp}.csv"

        filepath = os.path.join(self.output_dir, filename)

        try:
            # 转换数据为DataFrame
            df = pd.DataFrame(data)

            # 保存为CSV
            df.to_csv(filepath, index=False, encoding='utf-8-sig')

            logging.info(f"数据已导出到CSV文件: {filepath}")
            return filepath

        except Exception as e:
            logging.error(f"导出CSV文件失败: {str(e)}")
            raise

    def export_excel(self, data: List[Dict[str, Any]], filename: str = None) -> str:
        """
        导出数据为Excel文件

        Args:
            data: 数据列表
            filename: 文件名

        Returns:
            导出的文件路径
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{timestamp}.xlsx"

        filepath = os.path.join(self.output_dir, filename)

        try:
            df = pd.DataFrame(data)

            # 使用ExcelWriter保存多个sheet
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='检测结果', index=False)

                # 添加统计信息sheet
                stats_df = self._calculate_statistics(data)
                stats_df.to_excel(writer, sheet_name='统计信息', index=False)

            logging.info(f"数据已导出到Excel文件: {filepath}")
            return filepath

        except Exception as e:
            logging.error(f"导出Excel文件失败: {str(e)}")
            raise

    def export_json(self, data: Dict[str, Any], filename: str = None) -> str:
        """
        导出数据为JSON文件

        Args:
            data: 数据字典
            filename: 文件名

        Returns:
            导出的文件路径
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{timestamp}.json"

        filepath = os.path.join(self.output_dir, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logging.info(f"数据已导出到JSON文件: {filepath}")
            return filepath

        except Exception as e:
            logging.error(f"导出JSON文件失败: {str(e)}")
            raise

    def export_pickle(self, data: Any, filename: str = None) -> str:
        """
        导出数据为pickle文件

        Args:
            data: 任意Python对象
            filename: 文件名

        Returns:
            导出的文件路径
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data_{timestamp}.pkl"

        filepath = os.path.join(self.output_dir, filename)

        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)

            logging.info(f"数据已导出到pickle文件: {filepath}")
            return filepath

        except Exception as e:
            logging.error(f"导出pickle文件失败: {str(e)}")
            raise

    def _calculate_statistics(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """计算统计信息"""
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        stats = []

        # 数值型列的统计
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            stats.append({
                '指标': f'{col}_平均值',
                '值': f"{df[col].mean():.4f}",
                '单位': self._get_unit(col)
            })
            stats.append({
                '指标': f'{col}_标准差',
                '值': f"{df[col].std():.4f}",
                '单位': self._get_unit(col)
            })
            stats.append({
                '指标': f'{col}_最小值',
                '值': f"{df[col].min():.4f}",
                '单位': self._get_unit(col)
            })
            stats.append({
                '指标': f'{col}_最大值',
                '值': f"{df[col].max():.4f}",
                '单位': self._get_unit(col)
            })

        # 合格率统计（如果有is_qualified列）
        if 'is_qualified' in df.columns:
            total = len(df)
            passed = df['is_qualified'].sum()
            failed = total - passed
            pass_rate = (passed / total * 100) if total > 0 else 0

            stats.append({
                '指标': '总检测数',
                '值': str(total),
                '单位': '个'
            })
            stats.append({
                '指标': '合格数',
                '值': str(passed),
                '单位': '个'
            })
            stats.append({
                '指标': '不合格数',
                '值': str(failed),
                '单位': '个'
            })
            stats.append({
                '指标': '合格率',
                '值': f"{pass_rate:.2f}",
                '单位': '%'
            })

        return pd.DataFrame(stats)

    @staticmethod
    def _get_unit(column_name: str) -> str:
        """根据列名获取单位"""
        unit_map = {
            'concentricity': '‰',
            'eccentricity_mm': 'mm',
            'eccentricity_px': 'px',
            'radius': 'px',
            'processing_time': 's',
            'tolerance': 'mm',
            'x': 'px',
            'y': 'px'
        }

        for key, unit in unit_map.items():
            if key in column_name.lower():
                return unit

        return ''


# ====================== 图像文件处理 ======================

def load_image(filepath: str, color_mode: str = 'rgb') -> np.ndarray:
    """
    加载图像文件

    Args:
        filepath: 图像文件路径
        color_mode: 颜色模式，'rgb'或'bgr'

    Returns:
        numpy数组表示的图像
    """
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"图像文件不存在: {filepath}")

        # 使用OpenCV加载图像
        image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

        if image is None:
            raise ValueError(f"无法读取图像文件: {filepath}")

        # 转换为RGB格式
        if color_mode.lower() == 'rgb':
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

        return image

    except Exception as e:
        logging.error(f"加载图像失败: {str(e)}")
        raise


def save_image(image: np.ndarray, filepath: str,
               quality: int = 95, create_dir: bool = True) -> str:
    """
    保存图像文件

    Args:
        image: numpy数组表示的图像
        filepath: 保存路径
        quality: JPEG质量（1-100）
        create_dir: 是否创建目录

    Returns:
        保存的文件路径
    """
    try:
        # 确保目录存在
        if create_dir:
            directory = os.path.dirname(filepath)
            if directory:
                os.makedirs(directory, exist_ok=True)

        # 确保图像是uint8类型
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # 根据扩展名确定保存格式
        ext = os.path.splitext(filepath)[1].lower()

        if ext in ['.jpg', '.jpeg']:
            # JPEG格式
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            # 如果是RGBA，转换为RGB
            if len(image.shape) == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            cv2.imwrite(filepath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR), params)

        elif ext == '.png':
            # PNG格式
            params = [cv2.IMWRITE_PNG_COMPRESSION, 9 - int(quality / 100 * 9)]
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
            cv2.imwrite(filepath, image, params)

        else:
            # 其他格式
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, image)

        logging.info(f"图像已保存: {filepath}")
        return filepath

    except Exception as e:
        logging.error(f"保存图像失败: {str(e)}")
        raise


def save_image_with_annotations(image: np.ndarray, filepath: str,
                                annotations: Dict[str, Any] = None,
                                draw_circles: bool = True,
                                draw_text: bool = True) -> str:
    """
    保存带标注的图像

    Args:
        image: 原始图像
        filepath: 保存路径
        annotations: 标注信息
        draw_circles: 是否绘制圆
        draw_text: 是否绘制文本

    Returns:
        保存的文件路径
    """
    try:
        # 创建图像副本
        annotated_image = image.copy()

        if annotations and (draw_circles or draw_text):
            # 绘制标注
            if draw_circles and 'circles' in annotations:
                circles = annotations['circles']
                if isinstance(circles, list) and len(circles) >= 2:
                    # 绘制内圆
                    inner_circle = circles[0]
                    if len(inner_circle) == 3:
                        x, y, r = map(int, inner_circle)
                        cv2.circle(annotated_image, (x, y), r, (0, 255, 0), 2)
                        cv2.circle(annotated_image, (x, y), 3, (0, 255, 0), -1)

                    # 绘制外圆
                    outer_circle = circles[1]
                    if len(outer_circle) == 3:
                        x, y, r = map(int, outer_circle)
                        cv2.circle(annotated_image, (x, y), r, (255, 0, 0), 2)
                        cv2.circle(annotated_image, (x, y), 3, (255, 0, 0), -1)

                    # 绘制中心连线
                    if len(circles[0]) == 3 and len(circles[1]) == 3:
                        x1, y1, _ = map(int, circles[0])
                        x2, y2, _ = map(int, circles[1])
                        cv2.line(annotated_image, (x1, y1), (x2, y2), (255, 255, 0), 2)

            if draw_text and 'results' in annotations:
                results = annotations['results']
                font = cv2.FONT_HERSHEY_SIMPLEX
                y_offset = 30

                for key, value in results.items():
                    if isinstance(value, (int, float)):
                        text = f"{key}: {value:.3f}"
                    else:
                        text = f"{key}: {value}"

                    cv2.putText(annotated_image, text, (10, y_offset),
                                font, 0.7, (255, 255, 255), 2)
                    y_offset += 25

        # 保存图像
        return save_image(annotated_image, filepath)

    except Exception as e:
        logging.error(f"保存标注图像失败: {str(e)}")
        raise


# ====================== 配置文件处理 ======================

def save_parameters_to_yaml(parameters: Dict[str, Any],
                            filepath: str,
                            create_dir: bool = True) -> str:
    """
    保存参数到YAML文件

    Args:
        parameters: 参数字典
        filepath: YAML文件路径
        create_dir: 是否创建目录

    Returns:
        保存的文件路径
    """
    try:
        if create_dir:
            directory = os.path.dirname(filepath)
            if directory:
                os.makedirs(directory, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(parameters, f, default_flow_style=False, allow_unicode=True)

        logging.info(f"参数已保存到YAML文件: {filepath}")
        return filepath

    except Exception as e:
        logging.error(f"保存YAML文件失败: {str(e)}")
        raise


def load_parameters_from_yaml(filepath: str) -> Dict[str, Any]:
    """
    从YAML文件加载参数

    Args:
        filepath: YAML文件路径

    Returns:
        参数字典
    """
    try:
        if not os.path.exists(filepath):
            logging.warning(f"YAML文件不存在: {filepath}")
            return {}

        with open(filepath, 'r', encoding='utf-8') as f:
            parameters = yaml.safe_load(f)

        return parameters or {}

    except Exception as e:
        logging.error(f"加载YAML文件失败: {str(e)}")
        raise


# ====================== 结果文件处理 ======================

def save_results_to_csv(results: Union[Dict[str, Any], List[Dict[str, Any]]],
                        filepath: str,
                        mode: str = 'w') -> str:
    """
    保存结果到CSV文件

    Args:
        results: 单个结果字典或结果列表
        filepath: CSV文件路径
        mode: 写入模式，'w'为覆盖，'a'为追加

    Returns:
        保存的文件路径
    """
    try:
        # 确保目录存在
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        # 转换结果为列表
        if isinstance(results, dict):
            results_list = [results]
        else:
            results_list = results

        # 写入CSV文件
        with open(filepath, mode, newline='', encoding='utf-8-sig') as f:
            if results_list:
                # 获取所有字段
                fieldnames = set()
                for result in results_list:
                    fieldnames.update(result.keys())
                fieldnames = sorted(fieldnames)

                writer = csv.DictWriter(f, fieldnames=fieldnames)

                # 写入表头（如果是覆盖模式）
                if mode == 'w':
                    writer.writeheader()

                # 写入数据
                for result in results_list:
                    writer.writerow(result)

        logging.info(f"结果已保存到CSV文件: {filepath}")
        return filepath

    except Exception as e:
        logging.error(f"保存CSV文件失败: {str(e)}")
        raise


def save_results_to_excel(results: Union[Dict[str, Any], List[Dict[str, Any]]],
                          filepath: str) -> str:
    """
    保存结果到Excel文件

    Args:
        results: 单个结果字典或结果列表
        filepath: Excel文件路径

    Returns:
        保存的文件路径
    """
    try:
        # 使用DataExporter
        exporter = DataExporter(os.path.dirname(filepath))

        if isinstance(results, dict):
            results_list = [results]
        else:
            results_list = results

        filename = os.path.basename(filepath)
        return exporter.export_excel(results_list, filename)

    except Exception as e:
        logging.error(f"保存Excel文件失败: {str(e)}")
        raise


# ====================== 报告生成 ======================

def export_report_to_pdf(data: Dict[str, Any], filepath: str,
                         title: str = "同心度检测报告",
                         company: str = "西南科技大学",
                         author: str = "同心度检测系统") -> str:
    """
    导出报告为PDF文件

    Args:
        data: 报告数据
        filepath: PDF文件路径
        title: 报告标题
        company: 公司/单位名称
        author: 作者

    Returns:
        导出的文件路径
    """
    try:
        # 确保目录存在
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        # 创建PDF文档
        doc = SimpleDocTemplate(filepath, pagesize=A4,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)

        styles = getSampleStyleSheet()
        story = []

        # 标题
        title_style = styles['Heading1']
        title_style.alignment = 1  # 居中
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 12))

        # 基本信息
        story.append(Paragraph(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Paragraph(f"生成单位: {company}", styles['Normal']))
        story.append(Paragraph(f"生成系统: {author}", styles['Normal']))
        story.append(Spacer(1, 20))

        # 检测结果
        if 'results' in data:
            story.append(Paragraph("检测结果", styles['Heading2']))

            results_data = []
            results_data.append(['参数', '数值', '单位'])

            for key, value in data['results'].items():
                if isinstance(value, (int, float)):
                    value_str = f"{value:.4f}"
                else:
                    value_str = str(value)

                # 确定单位
                unit = ''
                if 'concentricity' in key.lower():
                    unit = '‰'
                elif 'eccentricity' in key.lower():
                    unit = 'mm'
                elif 'radius' in key.lower():
                    unit = 'mm'
                elif 'time' in key.lower():
                    unit = 's'

                results_data.append([key, value_str, unit])

            # 创建表格
            table = Table(results_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            story.append(table)
            story.append(Spacer(1, 20))

        # 检测图像（如果有）
        if 'image_path' in data and os.path.exists(data['image_path']):
            story.append(Paragraph("检测图像", styles['Heading2']))

            try:
                # 添加图像（缩放到合适大小）
                img = ReportImage(data['image_path'], width=300, height=200)
                story.append(img)
                story.append(Spacer(1, 20))
            except:
                story.append(Paragraph("无法加载检测图像", styles['Normal']))

        # 备注
        if 'remarks' in data:
            story.append(Paragraph("备注", styles['Heading2']))
            story.append(Paragraph(data['remarks'], styles['Normal']))

        # 生成PDF
        doc.build(story)

        logging.info(f"报告已导出为PDF: {filepath}")
        return filepath

    except Exception as e:
        logging.error(f"导出PDF报告失败: {str(e)}")
        raise


def export_report_to_word(data: Dict[str, Any], filepath: str,
                          title: str = "同心度检测报告") -> str:
    """
    导出报告为Word文件

    Args:
        data: 报告数据
        filepath: Word文件路径
        title: 报告标题

    Returns:
        导出的文件路径
    """
    try:
        # 确保目录存在
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        # 创建Word文档
        doc = Document()

        # 标题
        title_para = doc.add_paragraph()
        title_run = title_para.add_run(title)
        title_run.bold = True
        title_run.font.size = Pt(16)
        title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER

        doc.add_paragraph()  # 空行

        # 基本信息
        doc.add_paragraph(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        doc.add_paragraph(f"生成系统: 同心度检测系统 v1.0")

        doc.add_paragraph()  # 空行

        # 检测结果
        if 'results' in data:
            doc.add_paragraph("检测结果").bold = True

            # 创建表格
            table = doc.add_table(rows=1, cols=3)
            table.style = 'Light Grid'

            # 表头
            header_cells = table.rows[0].cells
            header_cells[0].text = "参数"
            header_cells[1].text = "数值"
            header_cells[2].text = "单位"

            # 数据行
            for key, value in data['results'].items():
                row_cells = table.add_row().cells
                row_cells[0].text = key

                if isinstance(value, (int, float)):
                    row_cells[1].text = f"{value:.4f}"
                else:
                    row_cells[1].text = str(value)

                # 确定单位
                unit = ''
                if 'concentricity' in key.lower():
                    unit = '‰'
                elif 'eccentricity' in key.lower():
                    unit = 'mm'
                elif 'radius' in key.lower():
                    unit = 'mm'
                elif 'time' in key.lower():
                    unit = 's'

                row_cells[2].text = unit

        doc.add_paragraph()  # 空行

        # 备注
        if 'remarks' in data:
            doc.add_paragraph("备注").bold = True
            doc.add_paragraph(data['remarks'])

        # 保存文档
        doc.save(filepath)

        logging.info(f"报告已导出为Word: {filepath}")
        return filepath

    except Exception as e:
        logging.error(f"导出Word报告失败: {str(e)}")
        raise


# ====================== 其他工具函数 ======================

def create_directory(directory: str) -> str:
    """
    创建目录（如果不存在）

    Args:
        directory: 目录路径

    Returns:
        创建的目录路径
    """
    try:
        os.makedirs(directory, exist_ok=True)
        return directory
    except Exception as e:
        logging.error(f"创建目录失败: {str(e)}")
        raise


def get_unique_filename(base_name: str, extension: str,
                        directory: str = ".") -> str:
    """
    获取唯一的文件名

    Args:
        base_name: 基础文件名
        extension: 文件扩展名（带点）
        directory: 目录路径

    Returns:
        唯一的文件路径
    """
    try:
        create_directory(directory)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 尝试带时间戳的文件名
        filename = f"{base_name}_{timestamp}{extension}"
        filepath = os.path.join(directory, filename)

        if not os.path.exists(filepath):
            return filepath

        # 如果存在，添加序号
        counter = 1
        while True:
            filename = f"{base_name}_{timestamp}_{counter:03d}{extension}"
            filepath = os.path.join(directory, filename)

            if not os.path.exists(filepath):
                return filepath

            counter += 1

    except Exception as e:
        logging.error(f"生成唯一文件名失败: {str(e)}")
        raise