"""
STA/LTA震相拾取器
"""
import numpy as np
import obspy
import logging
from .base import BasePicker

logger = logging.getLogger(__name__)

class STALTAPicker(BasePicker):
    """STA/LTA震相拾取器 - 支持P波和S波差异化检测"""
    
    def __init__(self, mode='dual', p_wave_params=None, s_wave_params=None, 
                 sta_window=1.0, lta_window=8.0, threshold=1.2):
        """
        初始化STA/LTA拾取器
        
        Args:
            mode: 检测模式 ('single', 'dual', 'p_only', 's_only')
            p_wave_params: P波检测参数字典
            s_wave_params: S波检测参数字典
            sta_window, lta_window, threshold: 单一模式下的参数（兼容性）
        """
        super().__init__()
        self.mode = mode
        
        # 基于我们分析的优化参数
        self.p_wave_params = p_wave_params or {
            'sta_window': 0.1,      # 短窗口捕捉P波初动 (从0.3优化)
            'lta_window': 12.0,     # 长窗口稳定背景 (从16.0优化)
            'threshold': 1.05,      # 略高于观测到的P波最大值1.1616 (从1.12优化)
            'detrigger_threshold': 1.05,
            'min_duration': 1.0,    # P波最小持续时间 (从1.5优化)
            'max_triggers': 2,      # 新增：限制最大检测数量
        }
        
        self.s_wave_params = s_wave_params or {
            'sta_window': 0.8,      # 适中窗口适应S波特征 (与agent.py一致)
            'lta_window': 8.0,      # 标准LTA窗口 (与agent.py一致)
            'threshold': 1.5,       # S波触发阈值 (与agent.py一致)
            'detrigger_threshold': 1.3, # S波去触发阈值 (与agent.py一致)
            'min_duration': 1.0,    # S波最小持续时间 (与agent.py一致)
            'frequency_band': [0.5, 10],
            'search_window': 30.0   # 在P波后30秒内搜索
        }
        
        # 兼容性：单一模式参数
        if mode == 'single':
            self.sta_window = sta_window
            self.lta_window = lta_window
            self.threshold = threshold
            self.detrigger_threshold = 1.0
            
        logger.info(f"STALTAPicker 初始化: 模式={mode}")
        
    def _calculate_stalta(self, data, sampling_rate, params):
        """
        改进的STA/LTA计算 - 支持参数化
        
        Args:
            data: 波形数据
            sampling_rate: 采样率
            params: 参数字典，包含sta_window, lta_window等
            
        Returns:
            numpy.ndarray: STA/LTA比值
        """
        sta_len = int(params['sta_window'] * sampling_rate)
        lta_len = int(params['lta_window'] * sampling_rate)
        
        # 数据长度检查
        if len(data) < lta_len:
            logger.error(f"数据长度({len(data)})小于LTA窗口长度({lta_len})")
            return np.zeros(len(data))
        
        stalta_ratio = np.zeros(len(data))
        
        # 使用滑动窗口计算（更高效）
        # 预计算平方
        data_squared = data ** 2
        
        # 计算累积和以提高效率
        cumsum = np.cumsum(np.concatenate(([0], data_squared)))
        
        for i in range(lta_len, len(data)):
            # 使用累积和快速计算窗口平均值
            sta_val = (cumsum[i] - cumsum[i-sta_len]) / sta_len
            lta_val = (cumsum[i] - cumsum[i-lta_len]) / lta_len
            
            # 避免除零，增加数值稳定性
            if lta_val > 1e-12:
                stalta_ratio[i] = sta_val / lta_val
            else:
                stalta_ratio[i] = 0.0
        
        logger.debug(f"STA/LTA计算完成。参数: STA={params['sta_window']}s, LTA={params['lta_window']}s")
        logger.debug(f"最大STA/LTA比值: {np.max(stalta_ratio):.4f}")
        
        return stalta_ratio
        
    def _find_triggers(self, stalta, threshold_on, threshold_off):
        """
        寻找触发和去触发点，返回P波到时索引
        """
        triggers = []
        in_trigger = False
        
        for i, ratio in enumerate(stalta):
            if not in_trigger and ratio > threshold_on:
                triggers.append(i)
                in_trigger = True
            elif in_trigger and ratio < threshold_off:
                in_trigger = False
        
        return triggers

    def _find_triggers_advanced(self, stalta, params, sampling_rate):
        """
        增强的触发检测，支持持续时间过滤和质量评估
        """
        threshold_on = params['threshold']
        threshold_off = params['detrigger_threshold']
        min_duration = params.get('min_duration', 0.5)
        min_duration_samples = int(min_duration * sampling_rate)
        
        triggers = []
        in_trigger = False
        trigger_start = None
        trigger_values = []
        
        for i, ratio in enumerate(stalta):
            if not in_trigger and ratio > threshold_on:
                in_trigger = True
                trigger_start = i
                trigger_values = [ratio]
                
            elif in_trigger:
                if ratio > threshold_off:
                    trigger_values.append(ratio)
                else:
                    # 检查持续时间
                    if len(trigger_values) >= min_duration_samples:
                        # 找到峰值位置
                        max_ratio = max(trigger_values)
                        peak_offset = trigger_values.index(max_ratio)
                        peak_index = trigger_start + peak_offset
                        
                        # 计算置信度
                        confidence = min(1.0, max_ratio / (threshold_on * 1.5))
                        
                        triggers.append({
                            'index': peak_index,
                            'ratio': max_ratio,
                            'confidence': confidence,
                            'duration': len(trigger_values) / sampling_rate,
                            'start_index': trigger_start,
                            'end_index': trigger_start + len(trigger_values)
                        })
                    
                    in_trigger = False
                    trigger_values = []
        
        return triggers

    def _empty_result(self, message: str, algorithm: str = 'STA/LTA', status: str = 'error') -> dict:
        """
        返回一个标准化的空拾取结果字典
        """
        return {
            'p_picks': [],
            's_picks': [],
            'algorithm': algorithm,
            'status': status,
            'message': message
        }

    def pick(self, st, **kwargs):
        """
        执行震相拾取 - 支持P波和S波差异化检测
        """
        try:
            if not st or len(st) == 0:
                logger.warning("输入波形为空")
                return self._empty_result("输入波形为空")

            tr = st[0]
            logger.info(f"开始震相拾取: {tr.id}, 模式: {self.mode}")
            
            if tr.stats.sampling_rate == 0:
                logger.error("采样率为0，无法进行震相拾取")
                return self._empty_result("采样率为0")
            
            if self.mode == 'single':
                return self._single_mode_pick(tr)
            elif self.mode == 'dual':
                return self._dual_mode_pick(tr)
            elif self.mode == 'p_only':
                return self._p_wave_pick(tr)
            elif self.mode == 's_only':
                return self._s_wave_pick(tr) # 补充S波单独拾取方法
            else:
                raise ValueError(f"不支持的模式: {self.mode}")
                
        except Exception as e:
            logger.error(f"震相拾取失败: {str(e)}")
            return self._empty_result(f"拾取失败: {str(e)}")

    # --- 辅助拾取方法 --- #

    def _single_mode_pick(self, tr):
        """
        单一模式拾取（兼容旧逻辑）
        """
        stalta = self._calculate_stalta(tr.data, tr.stats.sampling_rate, 
                                       {'sta_window': self.sta_window, 
                                        'lta_window': self.lta_window})
        
        # 触发器阈值使用兼容模式下的threshold
        p_pick_indices = self._find_triggers(stalta, self.threshold, self.detrigger_threshold) # 使用旧的_find_triggers
        
        # 为日志准备触发点信息
        log_triggers = []
        for idx in p_pick_indices:
            confidence = min(1.0, stalta[idx] / (self.threshold * 2))
            log_triggers.append({
                'index': idx,
                'ratio': stalta[idx],
                'confidence': confidence,
                'duration': 0.0, # 旧方法不计算持续时间
                'start_index': idx,
                'end_index': idx # 简化处理
            })

        # 记录STA/LTA分析报告
        single_mode_params = {
            'sta_window': self.sta_window,
            'lta_window': self.lta_window,
            'threshold': self.threshold,
            'detrigger_threshold': self.detrigger_threshold
        }
        self._log_stalta_analysis(stalta, tr, log_triggers, single_mode_params)

        if not p_pick_indices:
            logger.info(f"单一模式STA/LTA未检测到震相。最大STA/LTA比值: {np.max(stalta):.2f}, 触发阈值: {self.threshold}, 去触发阈值: {self.detrigger_threshold}")
            return self._empty_result('未检测到任何震相', status='success')
            
        p_picks = []
        for pick_idx in p_pick_indices:
            pick_time = tr.stats.starttime + pick_idx / tr.stats.sampling_rate
            confidence = min(1.0, stalta[pick_idx] / (self.threshold * 2))
            p_picks.append({
                'time': pick_time.isoformat(),
                'confidence': float(confidence)
            })
        
        logger.info(f"单一模式STA/LTA成功拾取到 {len(p_picks)} 个P波震相。详细拾取信息: {p_picks}")
        return {
            'p_picks': p_picks,
            's_picks': [],
            'algorithm': 'STA/LTA-Single',
            'status': 'success',
            'message': f'成功检测到{len(p_picks)}个P波震相'
        }

    def _dual_mode_pick(self, tr):
        """双模式拾取：先检测P波，再检测S波"""
        results = {'p_picks': [], 's_picks': [], 'algorithm': 'STA/LTA-Dual', 'status': 'success'}
        
        # 1. 检测P波
        p_results = self._p_wave_pick(tr)
        results['p_picks'] = p_results['p_picks']
        
        # 2. 如果检测到P波，在其后搜索S波
        if results['p_picks']:
            p_time_str = results['p_picks'][0]['time']
            p_time = obspy.UTCDateTime(p_time_str)
            p_index = int((p_time - tr.stats.starttime) * tr.stats.sampling_rate)
            
            # 在P波后5-35秒内搜索S波 (使用s_wave_params中的search_window)
            search_window_duration = self.s_wave_params.get('search_window', 30.0) # 默认30秒
            search_start_offset_s = 5.0 # P波后多少秒开始搜索S波

            search_start_idx = p_index + int(search_start_offset_s * tr.stats.sampling_rate)
            search_end_idx = min(len(tr.data), p_index + int((search_start_offset_s + search_window_duration) * tr.stats.sampling_rate))
            
            if search_end_idx > search_start_idx:
                s_results = self._s_wave_pick_in_window(tr, search_start_idx, search_end_idx)
                results['s_picks'] = s_results['s_picks']
        
        # 3. 验证P-S时差的合理性
        if results['p_picks'] and results['s_picks']:
            p_time = obspy.UTCDateTime(results['p_picks'][0]['time'])
            s_time = obspy.UTCDateTime(results['s_picks'][0]['time'])
            ps_diff = s_time - p_time
            
            if ps_diff < 2 or ps_diff > 60:  # P-S时差应在2-60秒之间
                logger.warning(f"P-S时差异常: {ps_diff:.2f}秒，移除S波检测")
                results['s_picks'] = []
        
        message = f"P波: {len(results['p_picks'])}个, S波: {len(results['s_picks'])}个"
        results['message'] = message
        logger.info(f"双模式拾取完成: {message}")
        
        return results

    def _p_wave_pick(self, tr):
        """
P波专用检测"""
        # 预处理：根据P波参数进行滤波
        tr_copy = tr.copy()
        if 'frequency_band' in self.p_wave_params and self.p_wave_params['frequency_band']:
            freq_band = self.p_wave_params['frequency_band']
            tr_copy.filter('bandpass', freqmin=freq_band[0], freqmax=freq_band[1])
        
        # 计算STA/LTA
        stalta = self._calculate_stalta(tr_copy.data, tr_copy.stats.sampling_rate, self.p_wave_params)
        
        # 寻找触发点
        triggers = self._find_triggers_advanced(stalta, self.p_wave_params, tr_copy.stats.sampling_rate)
        
        # 应用P波后处理过滤
        filtered_triggers = self._filter_p_wave_picks(triggers, tr_copy.stats.sampling_rate)
        
        # 根据max_triggers限制数量
        max_triggers = self.p_wave_params.get('max_triggers', None)
        if max_triggers is not None and len(filtered_triggers) > max_triggers:
            filtered_triggers = filtered_triggers[:max_triggers]

        # 记录STA/LTA分析报告
        self._log_stalta_analysis(stalta, tr, filtered_triggers, self.p_wave_params)

        # 转换为标准格式
        p_picks = []
        for trigger in filtered_triggers:
            pick_time = tr.stats.starttime + trigger['index'] / tr.stats.sampling_rate
            p_picks.append({
                'time': pick_time.isoformat(),
                'confidence': float(trigger['confidence']),
                'stalta_ratio': float(trigger['ratio']),
                'duration': float(trigger['duration']),
                'phase_type': 'P'
            })
        
        logger.info(f"P波检测完成: {len(p_picks)}个")
        return {
            'p_picks': p_picks,
            's_picks': [],
            'algorithm': 'STA/LTA-P',
            'status': 'success',
            'message': f'检测到{len(p_picks)}个P波'
        }

    def _s_wave_pick(self, tr):
        """
        S波专用检测 (在整个波形上搜索)
        """
        # 预处理：根据S波参数进行滤波
        tr_copy = tr.copy()
        if 'frequency_band' in self.s_wave_params and self.s_wave_params['frequency_band']:
            freq_band = self.s_wave_params['frequency_band']
            tr_copy.filter('bandpass', freqmin=freq_band[0], freqmax=freq_band[1])

        # 计算STA/LTA
        stalta = self._calculate_stalta(tr_copy.data, tr_copy.stats.sampling_rate, self.s_wave_params)

        # 寻找触发点
        triggers = self._find_triggers_advanced(stalta, self.s_wave_params, tr_copy.stats.sampling_rate)

        # 记录STA/LTA分析报告
        self._log_stalta_analysis(stalta, tr, triggers, self.s_wave_params)

        # 转换为标准格式
        s_picks = []
        for trigger in triggers:
            pick_time = tr.stats.starttime + trigger['index'] / tr.stats.sampling_rate
            s_picks.append({
                'time': pick_time.isoformat(),
                'confidence': float(trigger['confidence']),
                'stalta_ratio': float(trigger['ratio']),
                'duration': float(trigger['duration']),
                'phase_type': 'S'
            })
            
        logger.info(f"S波检测完成: {len(s_picks)}个")
        return {
            'p_picks': [],
            's_picks': s_picks,
            'algorithm': 'STA/LTA-S',
            'status': 'success',
            'message': f'检测到{len(s_picks)}个S波'
        }

    def _s_wave_pick_in_window(self, tr, start_sample, end_sample):
        """
        在指定窗口内检测S波
        """
        logger.debug(f"S波搜索窗口: {start_sample/tr.stats.sampling_rate:.1f}s - {end_sample/tr.stats.sampling_rate:.1f}s (绝对时间)")

        windowed_data = tr.data[start_sample:end_sample]
        # 如果窗口数据为空，则直接返回空结果
        if len(windowed_data) == 0:
            logger.warning("S波搜索窗口数据为空")
            return {'p_picks': [], 's_picks': [], 'algorithm': 'STA/LTA-S-Window', 'status': 'success', 'message': '指定窗口内无数据'}

        # 预处理：根据S波参数进行滤波
        tr_copy_windowed = tr.copy() # Create a copy of the trace to modify its data
        tr_copy_windowed.data = windowed_data # Assign only the windowed data to this trace
        tr_copy_windowed.stats.starttime = tr.stats.starttime + start_sample / tr.stats.sampling_rate # Adjust start time
        tr_copy_windowed.stats.npts = len(windowed_data) # Adjust number of points

        if 'frequency_band' in self.s_wave_params and self.s_wave_params['frequency_band']:
            freq_band = self.s_wave_params['frequency_band']
            tr_copy_windowed.filter('bandpass', freqmin=freq_band[0], freqmax=freq_band[1])
        
        # 计算STA/LTA (仅对窗口内数据)
        stalta_window = self._calculate_stalta(tr_copy_windowed.data, 
                                              tr_copy_windowed.stats.sampling_rate, 
                                              self.s_wave_params)
        
        # 寻找触发点
        triggers = self._find_triggers_advanced(stalta_window, self.s_wave_params, tr_copy_windowed.stats.sampling_rate)

        # 记录STA/LTA分析报告
        # 注意：这里传入的是tr_copy_windowed，因为stalta和triggers是针对windowed_data计算的
        self._log_stalta_analysis(stalta_window, tr_copy_windowed, triggers, self.s_wave_params)

        s_picks = []
        for trigger in triggers:
            # 拾取时间需要加上窗口的起始偏移
            pick_time = tr.stats.starttime + (start_sample + trigger['index']) / tr.stats.sampling_rate
            s_picks.append({
                'time': pick_time.isoformat(),
                'confidence': float(trigger['confidence']),
                'stalta_ratio': float(trigger['ratio']),
                'duration': float(trigger['duration']),
                'phase_type': 'S'
            })
        
        logger.info(f"窗口内S波检测完成: {len(s_picks)}个")
        return {
            'p_picks': [],
            's_picks': s_picks,
            'algorithm': 'STA/LTA-S-Window',
            'status': 'success',
            'message': f'窗口内检测到{len(s_picks)}个S波'
        }

    def _log_stalta_analysis(self, stalta, tr, triggers, params):
        """
        详细的STA/LTA分析日志
        """
        logger.debug("=== STA/LTA 分析报告 ===")
        logger.debug(f"波形信息: {tr.id}")
        logger.debug(f"采样率: {tr.stats.sampling_rate} Hz")
        logger.debug(f"数据长度: {len(tr.data)} 点 ({len(tr.data)/tr.stats.sampling_rate:.1f} 秒)")
        logger.debug(f"STA/LTA参数: STA={params['sta_window']}s, LTA={params['lta_window']}s, 阈值={params['threshold']}, 去触发={params['detrigger_threshold']}")
        logger.debug(f"STA/LTA统计: 最大值={np.max(stalta):.4f}, 平均值={np.mean(stalta):.4f}")
        
        # 打印关键时间点的比值
        key_times = [5, 10, 15, 20, 25, 30, 35, 40]  # 关键时间点
        for t in key_times:
            idx = int(t * tr.stats.sampling_rate)
            if idx < len(stalta):
                logger.debug(f"时间 {t:2d}s (索引{idx:4d}): STA/LTA = {stalta[idx]:.4f}")
        
        # 打印检测结果
        if triggers:
            logger.debug(f"检测到 {len(triggers)} 个触发点:")
            for i, trigger in enumerate(triggers):
                time_sec = trigger['index'] / tr.stats.sampling_rate
                logger.debug(f"  触发点{i+1}: 时间={time_sec:.2f}s, 比值={trigger['ratio']:.4f}, "
                            f"置信度={trigger['confidence']:.3f}, 持续={trigger['duration']:.2f}s")
        else:
            logger.debug("未检测到任何触发点")
        
        logger.debug("=== 分析报告结束 ===") 

    def _filter_p_wave_picks(self, triggers, sampling_rate, expected_time_range=(18, 22)):
        """
        过滤P波检测结果，保留最可能的P波
        """
        if not triggers:
            return []
        
        # 按STA/LTA比值排序，取最强的
        triggers_sorted = sorted(triggers, key=lambda x: x['ratio'], reverse=True)
        
        # 优先选择在预期时间范围内的触发点
        for trigger in triggers_sorted:
            time_sec = trigger['index'] / sampling_rate
            if expected_time_range[0] <= time_sec <= expected_time_range[1]:
                return [trigger]  # 返回最可能的P波
        
        # 如果没有在预期范围内的，返回最强的
        return [triggers_sorted[0]] 