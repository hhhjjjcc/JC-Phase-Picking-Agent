"""
峰值检测模块
"""
import numpy as np

def detect_peaks(x, threshold=0.5, min_distance=50):
    """
    检测信号中的峰值
    
    Args:
        x: 输入信号
        threshold: 峰值阈值
        min_distance: 最小峰值距离（采样点数）
        
    Returns:
        numpy.ndarray: 峰值位置索引
    """
    # 找到超过阈值的点
    peaks = np.where(x > threshold)[0]
    
    if len(peaks) == 0:
        return np.array([])
        
    # 找到局部最大值
    local_max = []
    for i in range(len(peaks)):
        if i == 0 or i == len(peaks) - 1:
            continue
            
        if x[peaks[i]] > x[peaks[i-1]] and x[peaks[i]] > x[peaks[i+1]]:
            local_max.append(peaks[i])
            
    if len(local_max) == 0:
        return np.array([])
        
    # 按距离过滤峰值
    filtered_peaks = [local_max[0]]
    for peak in local_max[1:]:
        if peak - filtered_peaks[-1] >= min_distance:
            filtered_peaks.append(peak)
            
    return np.array(filtered_peaks)