"""
同心度计算模块 - 偏心距计算、同心度转换、容差判断等
"""
import numpy as np
import math
from typing import Tuple, Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class ConcentricityCalculator:
    """同心度计算器类"""

    def __init__(self, pixel_to_mm_ratio: float = 0.1):
        """
        初始化同心度计算器

        Args:
            pixel_to_mm_ratio: 像素到毫米的转换比例 (mm/pixel)
        """
        self.pixel_to_mm_ratio = pixel_to_mm_ratio
        self.results_history = []

        # 容差设置
        self.tolerance_settings = {
            'concentricity_tolerance': 0.2,  # mm 同心度容差
            'eccentricity_tolerance': 0.5,  # mm 偏心距容差
            'radius_tolerance': 0.1,  # mm 半径容差
        }

        # 单位设置
        self.unit_settings = {
            'length_unit': 'mm',  # 长度单位
            'concentricity_unit': '‰',  # 同心度单位
            'angle_unit': 'degree',  # 角度单位
        }

    def calculate_concentricity(self,
                                inner_circle: Dict,
                                outer_circle: Dict,
                                reference_radius: float = None) -> Dict:
        """
        计算同心度
        """
        logger.info(f"=== 开始calculate_concentricity方法 ===")
        logger.info(f"inner_circle: {inner_circle}")
        logger.info(f"outer_circle: {outer_circle}")
        logger.info(f"reference_radius: {reference_radius}")
        logger.info(f"当前pixel_to_mm_ratio: {self.pixel_to_mm_ratio}")

        try:
            # 提取圆心坐标
            x1, y1 = inner_circle['x'], inner_circle['y']
            x2, y2 = outer_circle['x'], outer_circle['y']
            logger.info(f"内圆坐标: ({x1}, {y1}), 外圆坐标: ({x2}, {y2})")

            # 计算像素偏心距
            pixel_eccentricity = self._calculate_eccentricity_pixel(x1, y1, x2, y2)
            logger.info(f"像素偏心距: {pixel_eccentricity}")

            # 转换为实际尺寸
            eccentricity_mm = pixel_eccentricity * self.pixel_to_mm_ratio
            logger.info(f"偏心距(mm): {eccentricity_mm}")

            # 获取参考半径
            if reference_radius is None:
                reference_radius = inner_circle.get('radius', 0)
                logger.info(f"未提供reference_radius，使用内圆半径: {reference_radius}")

            logger.info(f"参考半径(像素): {reference_radius}")

            # 转换为实际半径
            reference_radius_mm = reference_radius * self.pixel_to_mm_ratio
            logger.info(f"参考半径(mm): {reference_radius_mm}")

            # 计算同心度（千分比）- 添加详细检查
            logger.info(f"检查参考半径: {reference_radius_mm} (类型: {type(reference_radius_mm)})")

            # 添加更严格的检查
            if reference_radius_mm is not None and abs(reference_radius_mm) > 1e-10:  # 使用很小的阈值而不是0
                logger.info(f"使用参考半径计算同心度")
                try:
                    concentricity_permille = (eccentricity_mm / reference_radius_mm) * 1000
                    logger.info(f"计算出的同心度: {concentricity_permille}‰")
                except ZeroDivisionError:
                    logger.error(
                        f"ZeroDivisionError: eccentricity_mm={eccentricity_mm}, reference_radius_mm={reference_radius_mm}")
                    concentricity_permille = float('inf')
            else:
                logger.warning(f"参考半径太小或为0: {reference_radius_mm}")
                # 使用内圆半径作为替代
                inner_radius_mm = inner_circle.get('radius', 0) * self.pixel_to_mm_ratio
                logger.info(f"尝试使用内圆半径: {inner_radius_mm}")

                if inner_radius_mm is not None and abs(inner_radius_mm) > 1e-10:
                    try:
                        concentricity_permille = (eccentricity_mm / inner_radius_mm) * 1000
                        logger.info(f"使用内圆半径计算同心度: {concentricity_permille}‰")
                    except ZeroDivisionError:
                        logger.error(
                            f"ZeroDivisionError with inner radius: eccentricity_mm={eccentricity_mm}, inner_radius_mm={inner_radius_mm}")
                        concentricity_permille = float('inf')
                else:
                    concentricity_permille = float('inf')
                    logger.error("内圆半径也太小或为0，无法计算同心度")

            # 计算偏心方向
            eccentric_direction = self._calculate_eccentric_direction(x1, y1, x2, y2)

            # 容差判断
            tolerance_result = self._check_tolerance(eccentricity_mm, concentricity_permille)

            # 创建结果字典
            result = {
                'inner_circle': inner_circle,
                'outer_circle': outer_circle,
                'pixel_eccentricity': pixel_eccentricity,
                'eccentricity_mm': eccentricity_mm,
                'reference_radius_mm': reference_radius_mm,
                'concentricity_permille': concentricity_permille,
                'eccentric_direction_rad': eccentric_direction,
                'eccentric_direction_deg': math.degrees(eccentric_direction),
                'tolerance_check': tolerance_result,
                'is_within_tolerance': tolerance_result['is_within_tolerance'],
                'timestamp': np.datetime64('now'),
                'pixel_to_mm_ratio': self.pixel_to_mm_ratio,
            }

            # 添加到历史记录
            self.results_history.append(result)

            logger.info(f"同心度计算完成: {concentricity_permille:.2f}‰, "
                        f"偏心距: {eccentricity_mm:.3f}mm, "
                        f"方向: {math.degrees(eccentric_direction):.1f}°")

            logger.info(f"=== 结束calculate_concentricity方法 ===")

            return result

        except Exception as e:
            logger.error(f"同心度计算失败: {e}")
            return {
                'error': str(e),
                'is_within_tolerance': False,
                'concentricity_permille': float('inf')
            }

    def _calculate_eccentricity_pixel(self, x1: float, y1: float,
                                      x2: float, y2: float) -> float:
        """
        计算像素偏心距

        Args:
            x1, y1: 基准圆心坐标
            x2, y2: 被测圆心坐标

        Returns:
            float: 偏心距（像素）
        """
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def _calculate_eccentric_direction(self, x1: float, y1: float,
                                       x2: float, y2: float) -> float:
        """
        计算偏心方向

        Args:
            x1, y1: 基准圆心坐标
            x2, y2: 被测圆心坐标

        Returns:
            float: 偏心方向（弧度，0表示正右方，π/2表示正上方）
        """
        dx = x2 - x1
        dy = y2 - y1

        if dx == 0 and dy == 0:
            return 0.0

        angle = math.atan2(dy, dx)

        # 确保角度在0到2π之间
        if angle < 0:
            angle += 2 * math.pi

        return angle

    def _check_tolerance(self, eccentricity_mm: float,
                         concentricity_permille: float) -> Dict:
        """
        检查是否在容差范围内

        Args:
            eccentricity_mm: 偏心距 (mm)
            concentricity_permille: 同心度 (‰)

        Returns:
            Dict: 容差检查结果
        """
        # 计算允许的偏心距
        allowed_eccentricity = self.tolerance_settings['concentricity_tolerance']

        # 检查是否在容差范围内
        is_within_eccentricity = eccentricity_mm <= allowed_eccentricity

        # 计算偏差百分比
        if allowed_eccentricity > 0:
            eccentricity_percentage = (eccentricity_mm / allowed_eccentricity) * 100
        else:
            eccentricity_percentage = 100 if eccentricity_mm > 0 else 0

        result = {
            'is_within_tolerance': is_within_eccentricity,
            'allowed_eccentricity_mm': allowed_eccentricity,
            'actual_eccentricity_mm': eccentricity_mm,
            'eccentricity_deviation_mm': eccentricity_mm - allowed_eccentricity,
            'eccentricity_percentage': eccentricity_percentage,
            'concentricity_permille': concentricity_permille,
            'tolerance_type': 'concentricity',
            'tolerance_value': self.tolerance_settings['concentricity_tolerance'],
        }

        return result

    def calculate_batch_concentricity(self,
                                      inner_circles: List[Dict],
                                      outer_circles: List[Dict]) -> List[Dict]:
        """
        批量计算同心度

        Args:
            inner_circles: 内圆列表
            outer_circles: 外圆列表

        Returns:
            List[Dict]: 批量计算结果
        """
        if len(inner_circles) != len(outer_circles):
            logger.error(f"内圆数量({len(inner_circles)})和外圆数量({len(outer_circles)})不匹配")
            return []

        results = []
        for i in range(len(inner_circles)):
            result = self.calculate_concentricity(
                inner_circles[i], outer_circles[i]
            )
            results.append(result)

        # 统计信息
        if results:
            stats = self._calculate_statistics(results)
            logger.info(f"批量计算完成，共 {len(results)} 个零件，合格率: {stats['pass_rate']:.1f}%")

        return results

    def _calculate_statistics(self, results: List[Dict]) -> Dict:
        """
        计算统计信息

        Args:
            results: 计算结果列表

        Returns:
            Dict: 统计信息
        """
        if not results:
            return {}

        concentricities = [r.get('concentricity_permille', 0) for r in results]
        eccentricities = [r.get('eccentricity_mm', 0) for r in results]
        within_tolerance = [r.get('is_within_tolerance', False) for r in results]

        valid_concentricities = [c for c in concentricities if c != float('inf')]
        valid_eccentricities = [e for e in eccentricities if e != float('inf')]

        stats = {
            'total_count': len(results),
            'pass_count': sum(within_tolerance),
            'fail_count': len(results) - sum(within_tolerance),
            'pass_rate': (sum(within_tolerance) / len(results)) * 100,
        }

        if valid_concentricities:
            stats.update({
                'avg_concentricity': np.mean(valid_concentricities),
                'min_concentricity': np.min(valid_concentricities),
                'max_concentricity': np.max(valid_concentricities),
                'std_concentricity': np.std(valid_concentricities),
            })

        if valid_eccentricities:
            stats.update({
                'avg_eccentricity': np.mean(valid_eccentricities),
                'min_eccentricity': np.min(valid_eccentricities),
                'max_eccentricity': np.max(valid_eccentricities),
                'std_eccentricity': np.std(valid_eccentricities),
            })

        return stats

    def calibrate_pixel_to_mm(self, known_length_mm: float,
                              measured_length_pixel: float) -> float:
        """
        标定像素到毫米的转换比例

        Args:
            known_length_mm: 已知实际长度 (mm)
            measured_length_pixel: 测量的像素长度

        Returns:
            float: 新的转换比例 (mm/pixel)
        """
        if measured_length_pixel <= 0:
            logger.error("测量的像素长度必须大于0")
            return self.pixel_to_mm_ratio

        new_ratio = known_length_mm / measured_length_pixel

        # 平滑更新
        self.pixel_to_mm_ratio = 0.7 * self.pixel_to_mm_ratio + 0.3 * new_ratio

        logger.info(f"像素到毫米比例已更新: {self.pixel_to_mm_ratio:.6f} mm/pixel "
                    f"(已知长度: {known_length_mm}mm, 像素长度: {measured_length_pixel})")

        return self.pixel_to_mm_ratio

    def set_tolerance(self, concentricity_tolerance: float = None,
                      eccentricity_tolerance: float = None,
                      radius_tolerance: float = None):
        """
        设置容差参数

        Args:
            concentricity_tolerance: 同心度容差 (mm)
            eccentricity_tolerance: 偏心距容差 (mm)
            radius_tolerance: 半径容差 (mm)
        """
        if concentricity_tolerance is not None:
            self.tolerance_settings['concentricity_tolerance'] = concentricity_tolerance

        if eccentricity_tolerance is not None:
            self.tolerance_settings['eccentricity_tolerance'] = eccentricity_tolerance

        if radius_tolerance is not None:
            self.tolerance_settings['radius_tolerance'] = radius_tolerance

        logger.info(f"容差设置已更新: {self.tolerance_settings}")

    def set_units(self, length_unit: str = None,
                  concentricity_unit: str = None,
                  angle_unit: str = None):
        """
        设置单位

        Args:
            length_unit: 长度单位 ('mm', 'cm', 'inch')
            concentricity_unit: 同心度单位 ('‰', '%', 'mm')
            angle_unit: 角度单位 ('degree', 'radian')
        """
        if length_unit in ['mm', 'cm', 'inch']:
            self.unit_settings['length_unit'] = length_unit

        if concentricity_unit in ['‰', '%', 'mm']:
            self.unit_settings['concentricity_unit'] = concentricity_unit

        if angle_unit in ['degree', 'radian']:
            self.unit_settings['angle_unit'] = angle_unit

        logger.info(f"单位设置已更新: {self.unit_settings}")

    def convert_units(self, value: float, from_unit: str,
                      to_unit: str) -> float:
        """
        单位转换

        Args:
            value: 原始值
            from_unit: 原始单位
            to_unit: 目标单位

        Returns:
            float: 转换后的值
        """
        # 长度单位转换
        length_conversions = {
            ('mm', 'cm'): 0.1,
            ('cm', 'mm'): 10.0,
            ('mm', 'inch'): 1 / 25.4,
            ('inch', 'mm'): 25.4,
            ('cm', 'inch'): 1 / 2.54,
            ('inch', 'cm'): 2.54,
        }

        # 角度单位转换
        angle_conversions = {
            ('radian', 'degree'): 180 / math.pi,
            ('degree', 'radian'): math.pi / 180,
        }

        # 同心度单位转换
        concentricity_conversions = {
            ('‰', '%'): 0.1,
            ('%', '‰'): 10.0,
            ('‰', 'mm'): None,  # 需要额外参数
            ('mm', '‰'): None,  # 需要额外参数
        }

        # 相同单位不需要转换
        if from_unit == to_unit:
            return value

        # 检查并执行转换
        conversion_key = (from_unit, to_unit)

        if conversion_key in length_conversions:
            return value * length_conversions[conversion_key]

        elif conversion_key in angle_conversions:
            return value * angle_conversions[conversion_key]

        elif conversion_key in concentricity_conversions:
            factor = concentricity_conversions[conversion_key]
            if factor is not None:
                return value * factor
            else:
                logger.warning(f"单位转换 {from_unit}->{to_unit} 需要额外参数")
                return value

        else:
            logger.warning(f"不支持的单位转换: {from_unit}->{to_unit}")
            return value

    def export_results(self, format: str = 'dict') -> Dict:
        """
        导出计算结果

        Args:
            format: 导出格式 ('dict', 'summary', 'statistics')

        Returns:
            Dict: 导出结果
        """
        if not self.results_history:
            return {"message": "无计算结果"}

        if format == 'dict':
            return {
                'results': self.results_history,
                'total_count': len(self.results_history),
                'settings': {
                    'pixel_to_mm_ratio': self.pixel_to_mm_ratio,
                    'tolerance_settings': self.tolerance_settings,
                    'unit_settings': self.unit_settings,
                }
            }

        elif format == 'summary':
            # 生成摘要
            latest_result = self.results_history[-1]
            stats = self._calculate_statistics(self.results_history)

            return {
                'latest_result': {
                    'concentricity': latest_result.get('concentricity_permille', 0),
                    'eccentricity_mm': latest_result.get('eccentricity_mm', 0),
                    'is_within_tolerance': latest_result.get('is_within_tolerance', False),
                    'direction_deg': latest_result.get('eccentric_direction_deg', 0),
                },
                'statistics': stats,
                'timestamp': np.datetime64('now').astype(str),
            }

        elif format == 'statistics':
            return self._calculate_statistics(self.results_history)

        else:
            return {"error": f"不支持的格式: {format}"}

    def clear_history(self):
        """清空历史记录"""
        self.results_history.clear()
        logger.info("历史记录已清空")

    # 在 core/concentricity_calc.py 的 ConcentricityCalculator 类中添加：

    def calculate(self, inner_circle, outer_circle, **kwargs):
        """
        兼容旧接口的calculate方法
        """
        logger.info(f"=== 开始calculate方法 ===")
        logger.info(f"inner_circle: {inner_circle}")
        logger.info(f"outer_circle: {outer_circle}")
        logger.info(f"kwargs: {kwargs}")

        # 更新像素到毫米比例
        if 'pixel_to_mm' in kwargs:
            new_ratio = kwargs['pixel_to_mm']
            logger.info(f"传入的pixel_to_mm参数: {new_ratio}")
            if new_ratio > 0:
                self.pixel_to_mm_ratio = new_ratio
            else:
                logger.warning(f"无效的像素到毫米比例: {new_ratio}，使用当前值: {self.pixel_to_mm_ratio}")

        logger.info(f"当前pixel_to_mm_ratio: {self.pixel_to_mm_ratio}")

        # 获取参考半径
        reference_radius = kwargs.get('reference_radius_mm', None)
        logger.info(f"传入的reference_radius_mm: {reference_radius}")

        if reference_radius is not None:
            # 转换为像素单位 - 添加零值检查
            if self.pixel_to_mm_ratio > 0:
                reference_radius_pixel = reference_radius / self.pixel_to_mm_ratio
                logger.info(f"转换后的reference_radius_pixel: {reference_radius_pixel}")
            else:
                logger.error(f"像素到毫米比例为0，无法转换参考半径")
                reference_radius_pixel = None
        else:
            reference_radius_pixel = None

        logger.info(f"传递给calculate_concentricity的reference_radius_pixel: {reference_radius_pixel}")

        # 调用实际的同心度计算方法
        try:
            result = self.calculate_concentricity(
                inner_circle=inner_circle,
                outer_circle=outer_circle,
                reference_radius=reference_radius_pixel
            )
        except Exception as e:
            import traceback
            logger.error(f"calculate_concentricity调用失败: {e}\n{traceback.format_exc()}")
            raise

        # 确保返回字典包含ui/main_window.py期望的键
        result['concentricity'] = result.get('concentricity_permille', 0)
        result['is_qualified'] = result.get('is_within_tolerance', False)

        logger.info(f"通过calculate方法计算同心度: {result['concentricity']:.2f}‰")
        logger.info(f"=== 结束calculate方法 ===")

        return result