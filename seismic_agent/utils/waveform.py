from typing import Dict, Tuple, Optional
import numpy as np
from obspy import Stream, Trace, read
from obspy.signal.filter import bandpass
import logging
import obspy

logger = logging.getLogger(__name__)

class WaveformProcessor:
    """波形处理工具类"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.default_filters = {
            'highpass': {'freq': 1.0},
            'lowpass': {'freq': 20.0},
            'bandpass': {'freqmin': 1.0, 'freqmax': 20.0}
        }
    
    def get_waveform(self, file_path):
        """
        从文件路径获取波形数据
        
        Args:
            file_path (str): 波形文件路径
            
        Returns:
            obspy.Stream: 波形数据
        """
        try:
            stream = obspy.read(file_path)
            logger.info(f"成功从 {file_path} 读取波形数据")
            return stream
        except Exception as e:
            logger.error(f"读取波形文件失败: {file_path}, 错误: {str(e)}")
            raise
    
    def preprocess(self, stream: Stream) -> Stream:
        """
        波形预处理
        
        Args:
            stream: 输入波形数据流
            
        Returns:
            Stream: 预处理后的波形数据流
        """
        try:
            # 1. 去趋势
            stream.detrend('linear')
            
            # 2. 去均值
            stream.detrend('demean')
            
            # 3. 带通滤波
            stream.filter('bandpass', 
                         freqmin=self.default_filters['bandpass']['freqmin'],
                         freqmax=self.default_filters['bandpass']['freqmax'])
            
            self.logger.info("波形预处理完成")
            return stream
            
        except Exception as e:
            self.logger.error(f"波形预处理失败: {str(e)}")
            raise
    
    def calculate_snr(self, st: obspy.Stream) -> float:
        """
        计算波形的信噪比（SNR）。
        噪声窗口取前15秒，信号窗口取15-30秒。
        
        Args:
            st: ObsPy Stream 对象
            
        Returns:
            float: 信噪比 (SNR)
        """
        if not st or len(st) == 0:
            logger.warning("波形数据为空，无法计算信噪比。")
            return 0.0

        tr = st[0]
        data = tr.data
        sampling_rate = tr.stats.sampling_rate

        if sampling_rate == 0:
            logger.error("采样率为0，无法计算信噪比。")
            return 0.0

        # 定义噪声和信号窗口（以秒为单位）
        noise_window_duration = 15.0  # 噪声窗口长度
        signal_window_start_time = 15.0 # 信号窗口开始时间（相对于波形开始）
        signal_window_duration = 15.0 # 信号窗口长度 (15s到30s，即15s时长)

        # 转换为采样点
        noise_samples = int(noise_window_duration * sampling_rate)
        signal_start_sample = int(signal_window_start_time * sampling_rate)
        signal_end_sample = int((signal_window_start_time + signal_window_duration) * sampling_rate)

        # 确保波形数据足够长
        required_samples = int(max(noise_window_duration, signal_window_start_time + signal_window_duration) * sampling_rate)
        if len(data) < required_samples:
            logger.warning(f"波形数据 ({len(data)}个采样点) 太短，至少需要 {required_samples} 个采样点才能计算信噪比，返回默认值0.0")
            return 0.0

        # 提取噪声和信号数据
        noise_data = data[0:noise_samples]
        signal_data = data[signal_start_sample:signal_end_sample]

        if len(noise_data) == 0 or len(signal_data) == 0:
            logger.warning("噪声或信号窗口数据不足，无法计算信噪比。")
            return 0.0

        # 计算RMS (均方根)
        rms_noise = np.sqrt(np.mean(noise_data**2))
        rms_signal = np.sqrt(np.mean(signal_data**2))

        # 避免除以零
        if rms_noise == 0:
            return float('inf') # 如果噪声为零，SNR为无穷大
        
        snr = rms_signal / rms_noise
        logger.info(f"SNR计算完成: 噪声RMS={rms_noise:.4f}, 信号RMS={rms_signal:.4f}, SNR={snr:.2f}")
        return snr
    
    def validate_waveform(self, stream: Stream) -> Dict:
        """
        验证波形数据质量
        
        Args:
            stream: 波形数据流
            
        Returns:
            Dict: 验证结果
        """
        validation_results = {
            'is_valid': True,
            'issues': [],
            'recommendations': []
        }
        
        try:
            for trace in stream:
                # 检查采样率
                if trace.stats.sampling_rate < 20:
                    validation_results['issues'].append("采样率过低")
                    validation_results['recommendations'].append("建议使用采样率≥50Hz的数据")
                
                # 检查数据长度
                if len(trace.data) < 1000:
                    validation_results['issues'].append("数据长度过短")
                    validation_results['recommendations'].append("建议使用至少20秒的波形数据")
                
                # 检查数据缺失
                nan_ratio = np.isnan(trace.data).sum() / len(trace.data)
                if nan_ratio > 0.1:
                    validation_results['issues'].append(f"数据缺失严重({nan_ratio:.1%})")
                    validation_results['recommendations'].append("建议预处理填补缺失数据")
            
            if validation_results['issues']:
                validation_results['is_valid'] = False
                
            return validation_results
            
        except Exception as e:
            self.logger.error(f"波形验证失败: {str(e)}")
            raise 
    
    def calculate_stalta_ratio(self, stream: Stream, sta_window: float = 0.3, lta_window: float = 16.0) -> Dict:
        """
        计算STA/LTA比值，用于辅助确定合适的参数
        
        Args:
            stream: 波形数据流
            sta_window: 短时窗口长度（秒），默认0.3秒
            lta_window: 长时窗口长度（秒），默认16.0秒
            
        Returns:
            Dict: 包含STA/LTA比值和时间序列的字典
        """
        try:
            if not stream or len(stream) == 0:
                return {'status': 'error', 'message': '输入波形为空'}
                
            # 获取第一个分量
            trace = stream[0]
            data = trace.data
            sampling_rate = trace.stats.sampling_rate
            
            # 计算窗口长度（采样点数）
            sta_len = int(sta_window * sampling_rate)
            lta_len = int(lta_window * sampling_rate)
            
            # 计算STA和LTA
            sta = np.zeros_like(data)
            lta = np.zeros_like(data)
            
            # 使用滑动窗口计算STA
            for i in range(len(data)):
                start = max(0, i - sta_len + 1)
                sta[i] = np.mean(np.abs(data[start:i+1]))
                
            # 使用滑动窗口计算LTA
            for i in range(len(data)):
                start = max(0, i - lta_len + 1)
                lta[i] = np.mean(np.abs(data[start:i+1]))
                
            # 避免除以零
            lta[lta == 0] = 1e-10
            
            # 计算STA/LTA比值
            stalta = sta / lta
            
            # 计算时间序列
            times = np.arange(len(data)) / sampling_rate
            
            # 计算一些统计信息
            max_ratio = np.max(stalta)
            mean_ratio = np.mean(stalta)
            std_ratio = np.std(stalta)
            
            # 找出超过不同阈值的点
            thresholds = [1.5, 2.0, 2.5, 3.0]
            trigger_points = {}
            for threshold in thresholds:
                trigger_points[threshold] = np.where(stalta > threshold)[0]
            
            return {
                'status': 'success',
                'data': {
                    'times': times.tolist(),
                    'stalta': stalta.tolist(),
                    'statistics': {
                        'max_ratio': float(max_ratio),
                        'mean_ratio': float(mean_ratio),
                        'std_ratio': float(std_ratio)
                    },
                    'trigger_points': {
                        str(thresh): points.tolist() 
                        for thresh, points in trigger_points.items()
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"计算STA/LTA比值失败: {str(e)}")
            return {'status': 'error', 'message': f'计算STA/LTA比值失败: {str(e)}'} 

    def stream_to_json(self, stream: Stream) -> Dict:
        """
        将ObsPy Stream对象转换为JSON可序列化的字典格式。
        
        Args:
            stream (obspy.Stream): 要转换的ObsPy Stream对象。
            
        Returns:
            Dict: 包含波形数据的JSON可序列化字典。
        """
        json_data = {
            "traces": []
        }

        for trace in stream:
            trace_data = {
                "channel": trace.stats.channel,
                "network": trace.stats.network,
                "station": trace.stats.station,
                "location": trace.stats.location,
                "sampling_rate": trace.stats.sampling_rate,
                "start_time": trace.stats.starttime.isoformat(),
                "delta": trace.stats.delta,
                "npts": trace.stats.npts,
                "data": trace.data.tolist() # 将numpy数组转换为Python列表
            }
            json_data["traces"].append(trace_data)

        self.logger.info("ObsPy Stream 成功转换为 JSON 格式。")
        return json_data 