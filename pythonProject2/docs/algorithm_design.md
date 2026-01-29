# 算法设计说明

## 1. 算法总体架构

本系统采用模块化算法设计，主要包含五大核心模块：图像采集、图像预处理、圆心检测、畸变矫正和同心度计算。各模块协同工作，形成完整的检测流水线。

### 1.1 算法流程总览
```
开始
  ↓
图像采集（相机/文件）
  ↓
图像预处理（去噪、反光抑制、边缘增强）
  ↓
透视畸变矫正（相机标定）
  ↓
双圆心检测（霍夫变换+最小二乘法）
  ↓
同心度计算（偏心距→同心度）
  ↓
结果输出与可视化
  ↓
结束
```

## 2. 核心算法模块详解

### 2.1 图像预处理模块 (`preprocess.py`)

#### 2.1.1 反光抑制算法
**问题**：金属零件表面强反光导致图像局部过曝，影响边缘检测精度。

**解决方案**：采用改进的Retinex算法
```python
def adaptive_retinex_enhancement(image, scales=[2, 4, 8], gain=1.5):
    """
    自适应Retinex光照补偿
    参数：
        image: 输入图像 (BGR格式)
        scales: 多尺度高斯核半径列表
        gain: 增益系数，控制增强程度
    返回：
        增强后的图像
    """
    # 1. 转换为HSV空间，处理亮度通道
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2].astype(np.float32) / 255.0
    
    # 2. 多尺度Retinex处理
    enhanced_V = np.zeros_like(V)
    for scale in scales:
        # 高斯滤波模拟光照分量
        gaussian = cv2.GaussianBlur(V, (0, 0), scale)
        # 计算反射分量：R = log(V) - log(G)
        R = np.log(V + 1e-6) - np.log(gaussian + 1e-6)
        enhanced_V += R
    
    # 3. 均值化并增益调整
    enhanced_V = enhanced_V / len(scales)
    enhanced_V = (enhanced_V - np.min(enhanced_V)) / (np.max(enhanced_V) - np.min(enhanced_V) + 1e-6)
    enhanced_V = np.clip(enhanced_V * gain, 0, 1)
    
    # 4. 合并回HSV空间
    hsv[:, :, 2] = (enhanced_V * 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
```

#### 2.1.2 噪声滤波处理
```python
def adaptive_filtering(image, filter_type='median', kernel_size=5, sigma=1.0):
    """
    自适应滤波处理
    参数：
        filter_type: 'median'或'gaussian'
        kernel_size: 滤波核大小（奇数）
        sigma: 高斯滤波的标准差
    返回：
        滤波后的图像
    """
    if filter_type == 'median':
        # 中值滤波：去除椒盐噪声
        return cv2.medianBlur(image, kernel_size)
    elif filter_type == 'gaussian':
        # 高斯滤波：平滑图像，保留边缘
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
```

#### 2.1.3 边缘增强与形态学处理
```python
def edge_enhancement(image, method='canny', low_threshold=50, high_threshold=150):
    """
    边缘增强处理
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if method == 'canny':
        edges = cv2.Canny(gray, low_threshold, high_threshold)
    elif method == 'sobel':
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobel_x**2 + sobel_y**2)
        edges = np.uint8(edges / np.max(edges) * 255)
    
    # 形态学操作：闭运算填充边缘间断
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    return edges
```

### 2.2 圆心检测模块 (`circle_detection.py`)

#### 2.2.1 改进的霍夫变换算法
**传统霍夫变换问题**：参数空间维度高，计算量大，对噪声敏感。

**改进方案**：多尺度检测+参数空间优化
```python
def improved_hough_circle_detection(edges, min_radius=20, max_radius=100, 
                                    param1=50, param2=30, scale_factor=0.8):
    """
    改进的霍夫圆检测算法
    参数：
        edges: 边缘图像
        min_radius, max_radius: 半径检测范围
        param1: Canny边缘检测高阈值
        param2: 圆心检测累加器阈值
        scale_factor: 图像缩放因子，加速计算
    返回：
        检测到的圆列表 [(x, y, radius), ...]
    """
    # 1. 图像金字塔多尺度检测
    circles_list = []
    
    for scale in [1.0, 0.8, 1.2]:  # 三个尺度
        scaled_edges = cv2.resize(edges, None, fx=scale, fy=scale, 
                                 interpolation=cv2.INTER_LINEAR)
        
        # 2. 自适应参数调整
        scaled_min_r = int(min_radius * scale)
        scaled_max_r = int(max_radius * scale)
        
        # 3. 霍夫圆检测
        circles = cv2.HoughCircles(
            scaled_edges,
            cv2.HOUGH_GRADIENT,
            dp=1,  # 累加器分辨率
            minDist=int(30 * scale),  # 最小圆心距离
            param1=param1,
            param2=int(param2 * scale),
            minRadius=scaled_min_r,
            maxRadius=scaled_max_r
        )
        
        if circles is not None:
            circles = circles[0] / scale  # 缩放回原始尺寸
            circles_list.extend(circles)
    
    # 4. 圆聚类与去重
    if circles_list:
        circles_array = np.array(circles_list)
        # 使用DBSCAN聚类去除重复检测
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=10, min_samples=1).fit(circles_array[:, :2])
        
        unique_circles = []
        for label in set(clustering.labels_):
            cluster_points = circles_array[clustering.labels_ == label]
            # 取聚类中心作为最终圆
            center = np.mean(cluster_points[:, :2], axis=0)
            radius = np.mean(cluster_points[:, 2])
            unique_circles.append([center[0], center[1], radius])
        
        return np.array(unique_circles)
    return None
```

#### 2.2.2 最小二乘法圆拟合优化
```python
def least_squares_circle_fitting(points, initial_center, initial_radius):
    """
    最小二乘法圆拟合优化
    参数：
        points: 边缘点坐标数组 (N×2)
        initial_center: 初始圆心估计
        initial_radius: 初始半径估计
    返回：
        优化后的圆心和半径
    """
    def circle_residuals(params, x, y):
        """
        圆方程残差计算
        (x - a)² + (y - b)² - r² = 0
        """
        a, b, r = params
        return (x - a)**2 + (y - b)**2 - r**2
    
    from scipy.optimize import least_squares
    
    x = points[:, 0]
    y = points[:, 1]
    
    # 初始参数
    initial_params = [initial_center[0], initial_center[1], initial_radius]
    
    # 最小二乘优化
    result = least_squares(
        circle_residuals,
        initial_params,
        args=(x, y),
        method='lm',  # Levenberg-Marquardt算法
        ftol=1e-8,
        xtol=1e-8,
        max_nfev=1000
    )
    
    a_opt, b_opt, r_opt = result.x
    return (a_opt, b_opt), r_opt
```

### 2.3 透视畸变矫正模块 (`distortion_correction.py`)

#### 2.3.1 相机标定（张正友标定法）
```python
def camera_calibration(calibration_images, pattern_size=(9, 6), square_size=25):
    """
    相机标定，获取内参矩阵和畸变系数
    参数：
        calibration_images: 标定板图像列表
        pattern_size: 棋盘格角点数量 (cols, rows)
        square_size: 棋盘格方格实际尺寸 (mm)
    返回：
        camera_matrix: 相机内参矩阵
        dist_coeffs: 畸变系数
    """
    # 准备三维世界坐标点
    objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    # 存储所有图像的点对应关系
    objpoints = []  # 三维世界坐标
    imgpoints = []  # 二维图像坐标
    
    for img in calibration_images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if ret:
            # 亚像素级角点精确化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            objpoints.append(objp)
            imgpoints.append(corners_refined)
    
    # 相机标定
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    
    return camera_matrix, dist_coeffs
```

#### 2.3.2 实时畸变矫正
```python
def real_time_distortion_correction(image, camera_matrix, dist_coeffs):
    """
    实时透视畸变矫正
    参数：
        image: 输入图像
        camera_matrix: 相机内参矩阵
        dist_coeffs: 畸变系数
    返回：
        矫正后的图像
    """
    h, w = image.shape[:2]
    
    # 计算最优的新相机矩阵
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    
    # 畸变矫正
    corrected = cv2.undistort(
        image, camera_matrix, dist_coeffs,
        None, new_camera_matrix
    )
    
    # 裁剪有效区域
    x, y, w_roi, h_roi = roi
    if w_roi > 0 and h_roi > 0:
        corrected = corrected[y:y+h_roi, x:x+w_roi]
    
    return corrected
```

### 2.4 同心度计算模块 (`concentricity_calc.py`)

#### 2.4.1 像素到物理尺寸转换
```python
def pixel_to_physical_conversion(pixel_coords, pixel_size_mm=0.05):
    """
    像素坐标转换为物理尺寸
    参数：
        pixel_coords: 像素坐标 (x, y)
        pixel_size_mm: 每个像素对应的物理尺寸 (mm/像素)
    返回：
        物理坐标 (mm)
    """
    return pixel_coords[0] * pixel_size_mm, pixel_coords[1] * pixel_size_mm
```

#### 2.4.2 同心度计算模型
```python
def calculate_concentricity(inner_center, outer_center, reference_radius_mm):
    """
    计算同心度
    参数：
        inner_center: 内圆圆心坐标 (mm)
        outer_center: 外圆圆心坐标 (mm)
        reference_radius_mm: 基准圆半径 (mm)
    返回：
        eccentricity: 偏心距 (mm)
        concentricity: 同心度 (‰)
    """
    # 计算偏心距
    dx = outer_center[0] - inner_center[0]
    dy = outer_center[1] - inner_center[1]
    eccentricity = np.sqrt(dx**2 + dy**2)
    
    # 计算同心度 (千分比)
    concentricity = (eccentricity / reference_radius_mm) * 1000
    
    return eccentricity, concentricity
```

#### 2.4.3 误差分析与精度评估
```python
def error_analysis(ground_truth, measured_values):
    """
    误差分析
    参数：
        ground_truth: 真实值列表
        measured_values: 测量值列表
    返回：
        误差统计信息
    """
    errors = np.array(measured_values) - np.array(ground_truth)
    
    stats = {
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'max_error': np.max(np.abs(errors)),
        'min_error': np.min(np.abs(errors)),
        'rmse': np.sqrt(np.mean(errors**2)),  # 均方根误差
        'mae': np.mean(np.abs(errors)),       # 平均绝对误差
    }
    
    # 计算置信区间 (95%)
    from scipy import stats
    confidence = 0.95
    n = len(errors)
    se = stats.sem(errors)  # 标准误差
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    stats['confidence_interval'] = (stats['mean_error'] - h, stats['mean_error'] + h)
    
    return stats
```

## 3. 算法性能优化策略

### 3.1 计算效率优化
1. **图像金字塔加速**：多尺度检测减少计算量
2. **区域限制检测**：基于零件位置先验信息限制检测区域
3. **并行计算优化**：利用多线程处理批量检测

### 3.2 精度提升策略
1. **亚像素级定位**：边缘检测后亚像素级拟合
2. **多帧平均**：连续多帧检测取平均值减少随机误差
3. **温度补偿**：考虑相机和零件温度变化对尺寸的影响

### 3.3 鲁棒性增强
1. **自适应阈值**：根据图像质量自动调整检测参数
2. **异常值剔除**：统计方法去除异常检测结果
3. **多算法融合**：霍夫变换与最小二乘法结果加权融合

## 4. 实验验证方案

### 4.1 标准件标定实验
```python
def calibration_experiment():
    """
    标准件标定实验设计
    """
    # 1. 准备不同直径的标准件
    standards = [10.0, 20.0, 30.0, 40.0, 50.0]  # mm
    
    results = []
    for true_diameter in standards:
        # 2. 多次测量取平均
        measurements = []
        for _ in range(10):
            # 图像采集与处理
            measured_diameter = detect_and_measure()
            measurements.append(measured_diameter)
        
        # 3. 计算误差
        avg_measured = np.mean(measurements)
        error = avg_measured - true_diameter
        
        results.append({
            'true_diameter': true_diameter,
            'measured_diameter': avg_measured,
            'error': error,
            'relative_error': (error / true_diameter) * 100
        })
    
    return results
```

### 4.2 批量测试实验
```python
def batch_test_experiment(sample_size=100):
    """
    批量测试实验设计
    """
    detection_times = []
    concentricity_results = []
    accuracy_records = []
    
    for i in range(sample_size):
        start_time = time.time()
        
        # 单次检测
        result = single_detection()
        
        end_time = time.time()
        detection_time = end_time - start_time
        
        # 记录结果
        detection_times.append(detection_time)
        concentricity_results.append(result['concentricity'])
        
        if result['is_standard']:  # 如果是标准件
            accuracy = 1 if abs(result['error']) <= 0.2 else 0
            accuracy_records.append(accuracy)
    
    # 统计结果
    stats = {
        'avg_detection_time': np.mean(detection_times),
        'max_detection_time': np.max(detection_times),
        'min_detection_time': np.min(detection_times),
        'throughput': 1 / np.mean(detection_times),  # 检测通量 (个/秒)
        'accuracy_rate': np.mean(accuracy_records) * 100 if accuracy_records else None
    }
    
    return stats
```

## 5. 预期性能指标

| 指标 | 目标值 | 测试方法 |
|------|--------|----------|
| 检测精度 | ≤ ±0.2 mm | 标准件重复测量 |
| 检测速度 | ≥ 3 fps | 批量零件连续检测 |
| 重复精度 | ≤ 0.1 mm | 同一零件多次测量 |
| 系统稳定性 | 连续运行8小时无异常 | 长时间稳定性测试 |
| 误检率 | ≤ 1% | 包含干扰的复杂场景测试 |
| 漏检率 | ≤ 0.5% | 标准件全覆盖测试 |

## 6. 算法创新点总结

1. **多尺度霍夫变换优化**：结合图像金字塔和参数空间优化，提升检测效率
2. **自适应Retinex反光抑制**：针对金属表面强反光的专用预处理算法
3. **实时畸变矫正集成**：将相机标定结果实时应用于检测流程
4. **双算法结果融合**：霍夫变换粗定位 + 最小二乘法精拟合
5. **全流程误差控制**：从图像采集到结果输出的全链路误差分析与补偿

## 7. 答辩要点提示

1. **算法原理清晰**：重点讲解霍夫变换、最小二乘法、Retinex算法的基本原理
2. **创新点突出**：强调算法改进的部分和实际效果提升
3. **实验数据支撑**：准备充分的实验数据和对比结果
4. **系统集成展示**：演示完整的检测流程和可视化界面
5. **实际应用价值**：说明系统在工业检测中的实际应用场景和价值

---

**注**：本算法设计旨在满足本科毕业设计要求，平衡算法复杂度与实现可行性，确保在规定时间内完成系统开发和实验验证。