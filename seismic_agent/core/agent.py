"""
地震智能体核心模块
"""
import os
import logging
import json
from dotenv import load_dotenv
from openai import OpenAI
from ..utils.waveform import WaveformProcessor
from ..tools.pickers.stalta import STALTAPicker
# from ..tools.pickers.phasenet import PhaseNetPicker
import obspy
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class SeismicAgent:
    """地震智能体类"""
    
    def __init__(self):
        """初始化地震智能体"""
        load_dotenv()
        DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
        if not DEEPSEEK_API_KEY:
            logger.error("DEEPSEEK_API_KEY 环境变量未设置。请在 .env 文件中设置它。")
            raise ValueError("DeepSeek API Key is not configured.")
        
        # 初始化 DeepSeek 客户端 (兼容 OpenAI API)
        self.deepseek_client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com" # DeepSeek API 的基准 URL
        )
        self.model_name = "deepseek-chat" # 使用 DeepSeek 的聊天模型

        self.waveform_processor = WaveformProcessor()
        self.stalta_picker = STALTAPicker(
            mode='dual',
            p_wave_params={
                'sta_window': 0.1,
                'lta_window': 12.0,
                'threshold': 1.05,
                'detrigger_threshold': 1.05,
                'min_duration': 1.0,
                'frequency_band': [1, 15]
            },
            s_wave_params={
                'sta_window': 0.8,
                'lta_window': 8.0,
                'threshold': 1.5,
                'detrigger_threshold': 1.3,
                'min_duration': 1.0,
                'frequency_band': [0.5, 10],
                'search_window': 30.0
            }
        )
        self.picker = self.stalta_picker
        # self.phasenet_picker = PhaseNetPicker()
        
    def process_request(self, request: Dict) -> Dict:
        """
        处理用户请求
        
        Args:
            request: 包含用户查询和文件路径的字典
            
        Returns:
            Dict: 处理结果
        """
        try:
            user_query = request.get('user_query', '')
            file_path = request.get('file_path')
            
            if not file_path:
                return {'status': 'error', 'message': '未提供波形文件路径'}
                
            # 读取波形数据
            st = self.waveform_processor.get_waveform(file_path)
            if not st:
                return {'status': 'error', 'message': '无法读取波形数据'}
                
            # 解析用户查询
            logger.debug(f"开始解析用户查询: {user_query}")
            query_result = self._interpret_user_query(user_query)
            if query_result['status'] == 'error':
                logger.error(f"用户查询解析失败: {query_result['message']}")
                return query_result
            logger.debug(f"用户查询解析成功: {query_result}")
                
            intent = query_result['data']['intent']
            phase_type = query_result['data']['phase_type']
            stalta_params = query_result['data'].get('stalta_params', {})
            logger.debug(f"提取的意图: {intent}, 震相类型: {phase_type}, STA/LTA参数: {stalta_params}")
            
            final_report_content = {}
            final_p_picks = []
            final_s_picks = []
            final_snr_value = 0.0 # Initialize with a default
            final_waveform_data_json = {}
            
            # 定义 stalta_analysis_content 变量，无论意图是什么，都确保它被定义
            stalta_analysis_content_for_report = None

            if intent == 'analyze_stalta':
                logger.info("意图为分析STA/LTA参数，将同时执行拾取和参数评估")
                # 首先执行STA/LTA参数评估，获取报告内容
                stalta_analysis_result = self.analyze_stalta_parameters(
                    file_path,
                    stalta_params.get('sta_window', 0.5),
                    stalta_params.get('lta_window', 10.0)
                )
                stalta_analysis_content_for_report = stalta_analysis_result.get('report', {}).get('report_content', '')
                
                # 使用用户指定的STA/LTA参数初始化拾取器
                logger.info(f"使用用户指定的STA/LTA参数进行拾取: {stalta_params}")
                p_wave_params = {
                    'sta_window': stalta_params.get('sta_window') if stalta_params.get('sta_window') is not None else 0.1,
                    'lta_window': stalta_params.get('lta_window') if stalta_params.get('lta_window') is not None else 12.0,
                    'threshold': stalta_params.get('threshold') if stalta_params.get('threshold') is not None else 1.05,
                    'detrigger_threshold': stalta_params.get('detrigger_threshold', 1.05),
                    'min_duration': stalta_params.get('min_duration', 1.0),
                    'frequency_band': stalta_params.get('frequency_band', [1, 15])
                }
                # S波参数从用户查询中获取可能不准确，此处仍使用默认或预设值，除非用户明确指定S波参数
                s_wave_params = {
                    'sta_window': 0.5,
                    'lta_window': 10.0,
                    'threshold': 1.5,
                    'detrigger_threshold': 1.3,
                    'min_duration': 1.0,
                    'frequency_band': [0.5, 10],
                    'search_window': 30.0
                }
                self.picker = STALTAPicker(
                    mode='dual',
                    p_wave_params=p_wave_params,
                    s_wave_params=s_wave_params
                )
                logger.debug(f"为analyze_stalta意图创建的拾取器参数: P波={p_wave_params}")

                # 计算信噪比 (现在在两种意图下都会计算)
                final_snr_value = self.waveform_processor.calculate_snr(st)
                logger.info(f"计算得到的信噪比: {final_snr_value}")

                # 执行震相拾取
                logger.debug("准备执行震相拾取")
                picking_result = self._handle_phase_picking(st, phase_type)
                if picking_result['status'] == 'error':
                    return picking_result # Return error immediately for picking failures

                final_p_picks = picking_result.get('result', {}).get('p_picks', [])
                final_s_picks = picking_result.get('result', {}).get('s_picks', [])
                final_waveform_data_json = self.waveform_processor.stream_to_json(st)

                # 生成包含STA/LTA评估和拾取结果的报告
                final_report_content = self._generate_analysis_report(st, picking_result, final_snr_value, user_query, stalta_analysis_content_for_report)

            else: # pick_phase 或 other 意图，执行常规震相拾取
                # Calculate SNR
                final_snr_value = self.waveform_processor.calculate_snr(st)
                logger.info(f"计算得到的信噪比: {final_snr_value}")

                # 如果用户指定了STA/LTA参数（非 analyze_stalta 意图下的自定义拾取参数），使用这些参数
                # 注意：这里假设 user_query 中的 stalta_params 是针对 pick_phase 意图的自定义拾取参数
                if stalta_params and intent == 'pick_phase': # 仅在 pick_phase 意图下才考虑这些参数作为自定义拾取参数
                    logger.info(f"使用用户指定STA/LTA参数进行常规拾取: {stalta_params}")
                    # 从用户参数中获取值，如果为None则使用默认值
                    p_wave_params = {
                        'sta_window': stalta_params.get('p_sta_window', 0.1),
                        'lta_window': stalta_params.get('p_lta_window', 12.0),
                        'threshold': stalta_params.get('p_threshold', 1.05),
                        'detrigger_threshold': stalta_params.get('p_detrigger_threshold', 1.05),
                        'min_duration': stalta_params.get('p_min_duration', 1.0),
                        'frequency_band': stalta_params.get('p_frequency_band', [1, 15])
                    }
                    
                    s_wave_params = {
                        'sta_window': stalta_params.get('s_sta_window', 0.5),
                        'lta_window': stalta_params.get('s_lta_window', 10.0),
                        'threshold': stalta_params.get('s_threshold', 1.5),
                        'detrigger_threshold': stalta_params.get('s_detrigger_threshold', 1.3),
                        'min_duration': stalta_params.get('s_min_duration', 1.0),
                        'frequency_band': stalta_params.get('s_frequency_band', [0.5, 10]),
                        'search_window': stalta_params.get('s_search_window', 30.0)
                    }

                    self.picker = STALTAPicker(
                        mode='dual',
                        p_wave_params=p_wave_params,
                        s_wave_params=s_wave_params
                    )
                    logger.debug(f"新创建的拾取器参数: P波={p_wave_params}, S波={s_wave_params}")
                else:
                    logger.info("使用默认STA/LTA参数")
                    self.picker = self.stalta_picker
                    logger.debug("使用默认拾取器参数")

                # 执行震相拾取
                logger.debug("准备执行震相拾取")
                picking_result = self._handle_phase_picking(st, phase_type)
                if picking_result['status'] == 'error':
                    return picking_result # Return error immediately for picking failures

                final_p_picks = picking_result.get('result', {}).get('p_picks', [])
                final_s_picks = picking_result.get('result', {}).get('s_picks', [])
                final_waveform_data_json = self.waveform_processor.stream_to_json(st)

                # 生成分析报告 (不带STA/LTA评估内容，除非是 analyze_stalta 意图)
                final_report_content = self._generate_analysis_report(st, picking_result, final_snr_value, user_query)

            # 统一的返回结构
            response_to_frontend = {
                'status': 'success', # 假设成功，除非之前有明确的错误返回
                'report': final_report_content,
                'p_picks': final_p_picks,
                's_picks': final_s_picks,
                'snr': final_snr_value,
                'waveform_data': final_waveform_data_json
            }
            logger.info(f"即将返回给前端的完整响应数据: {response_to_frontend}")
            return response_to_frontend
            
        except Exception as e:
            logger.error(f"处理请求失败: {str(e)}")
            return {'status': 'error', 'message': f'处理请求失败: {str(e)}'}
            
    def _interpret_user_query(self, query: str) -> Dict:
        """
        使用DeepSeek解析用户查询
        
        Args:
            query: 用户查询文本
            
        Returns:
            Dict: 解析结果
        """
        try:
            prompt = f"""请分析以下用户查询，提取关键信息并以JSON格式返回：

用户查询: {query}

请提取以下信息：
1. intent: 用户意图，可能是以下之一：
   - analyze_stalta: 分析STA/LTA参数
   - pick_phase: 拾取震相
   - other: 其他意图
2. phase_type: 震相类型，可能是以下之一：
   - P: P波
   - S: S波
   - both: 两者都要
3. stalta_params: STA/LTA参数（如果有指定）：
   - sta_window: 短时窗口长度（秒）
   - lta_window: 长时窗口长度（秒）
   - threshold: 触发阈值

请以JSON格式返回，例如：
{{
    "intent": "analyze_stalta",
    "phase_type": null,
    "stalta_params": {{
        "sta_window": 0.2,
        "lta_window": 15.0,
        "threshold": null
    }}
}}

如果用户没有指定某些参数，请使用null值。"""

            response = self.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            logger.debug(f"DeepSeek解析用户查询的原始响应: {response.choices[0].message.content}")

            result = json.loads(response.choices[0].message.content)
            logger.info(f"DeepSeek解析用户查询结果: {result}")
            
            return {
                'status': 'success',
                'data': result
            }
            
        except Exception as e:
            logger.error(f"使用DeepSeek解析用户查询失败: {str(e)}")
            return {
                'status': 'error',
                'message': f'解析用户查询失败: {str(e)}'
            }

    def _handle_phase_picking(self, waveform, phase_type):
        """
        处理震相拾取
        
        Args:
            waveform: 波形数据
            phase_type: 震相类型
            
        Returns:
            dict: 处理结果
        """
        try:
            # 计算信噪比
            snr = self.waveform_processor.calculate_snr(waveform)
            logger.info(f"计算的信噪比 (SNR): {snr:.2f}")
            
            # 使用STA/LTA进行拾取
            result = self.picker.pick(waveform)
            logger.info(f"STALTAPicker 返回的原始结果: {result}")
            
            if result['status'] == 'error':
                return result
                
            return {
                "status": "success",
                "result": result,
                "snr": snr
            }
            
        except Exception as e:
            logger.error(f"震相拾取失败: {str(e)}")
            return {
                "status": "error",
                "message": f"震相拾取失败: {str(e)}"
            }
            
    def _handle_batch_processing(self, params):
        """
        处理批量处理请求
        
        Args:
            params: 参数字典
            
        Returns:
            dict: 处理结果
        """
        return {
            "status": "error",
            "message": "批量处理功能尚未实现"
        }
        
    def _handle_explanation(self, params):
        """
        处理解释请求
        
        Args:
            params: 参数字典
            
        Returns:
            dict: 处理结果
        """
        return {
            "status": "error",
            "message": "解释功能尚未实现"
        }
        
    def _generate_response_message(self, result, phase_type):
        """
        生成响应消息
        
        Args:
            result: 处理结果
            phase_type: 震相类型
            
        Returns:
            str: 响应消息
        """
        logger.info(f"_generate_response_message 接收到的结果: {result}")
        
        # 添加更严格的检查
        if not isinstance(result, dict):
            logger.error(f"_generate_response_message 接收到的结果不是字典类型: {type(result)}")
            return "内部错误：拾取结果格式不正确"
        
        # 使用 .get() 方法安全地访问 'status' 键
        status = result.get("status")

        if status == "error":
            return result.get("message", "未知错误") # 安全获取message
            
        if phase_type == "P":
            picks = result.get("p_picks", [])
            if picks:
                return f"成功检测到{len(picks)}个P波震相"
            else:
                return "未检测到P波震相"
        elif phase_type == "S":
            picks = result.get("s_picks", [])
            if picks:
                return f"成功检测到{len(picks)}个S波震相"
            else:
                return "未检测到S波震相"
        else:
            p_picks = result.get("p_picks", [])
            s_picks = result.get("s_picks", [])
            return f"成功检测到{len(p_picks)}个P波震相和{len(s_picks)}个S波震相"

    def _generate_analysis_report(self, st: obspy.Stream, picks_data: Dict, snr: float, user_query: str, stalta_analysis_content: Optional[str] = None) -> Dict:
        """
        根据震相拾取结果、信噪比、用户查询和可选的STA/LTA分析内容生成智能报告。
        
        Args:
            st: 波形数据流
            picks_data: 震相拾取结果
            snr: 信噪比
            user_query: 用户查询
            stalta_analysis_content: 可选的STA/LTA参数评估报告内容
            
        Returns:
            Dict: 包含分析报告内容的字典，键为 "report_content"
        """
        try:
            # 准备报告内容
            p_picks = picks_data.get('result', {}).get('p_picks', [])
            s_picks = picks_data.get('result', {}).get('s_picks', [])

            p_pick_details = ""
            if p_picks:
                p_pick_details = "\nP波到时信息：\n"
                for i, pick in enumerate(p_picks):
                    p_pick_details += f"  - P{i+1}: 时间={pick.get('time', 'N/A')} (UTC), " \
                                     f"置信度={pick.get('confidence', 'N/A'):.2f}, " \
                                     f"STA/LTA比值={pick.get('stalta_ratio', 'N/A'):.2f}\n"
            else:
                p_pick_details = "\n未拾取到P波。\n"

            s_pick_details = ""
            if s_picks:
                s_pick_details = "\nS波到时信息：\n"
                for i, pick in enumerate(s_picks):
                    s_pick_details += f"  - S{i+1}: 时间={pick.get('time', 'N/A')} (UTC), " \
                                     f"置信度={pick.get('confidence', 'N/A'):.2f}, " \
                                     f"STA/LTA比值={pick.get('stalta_ratio', 'N/A'):.2f}\n"
            else:
                s_pick_details = "\n未拾取到S波。\n"

            report_content = {
                "waveform_info": {
                    "station": st[0].stats.station,
                    "channel": st[0].stats.channel,
                    "start_time": st[0].stats.starttime.isoformat(),
                    "end_time": st[0].stats.endtime.isoformat(),
                    "sampling_rate": st[0].stats.sampling_rate,
                    "duration": st[0].stats.endtime - st[0].stats.starttime
                },
                "analysis_results": {
                    "snr": snr,
                    "p_picks": p_picks,
                    "s_picks": s_picks
                },
                "user_query": user_query
            }
            
            # 构建提示
            stalta_report_section = f"\nSTA/LTA参数评估报告：\n{stalta_analysis_content}" if stalta_analysis_content else ""
            prompt = f"""请根据以下信息生成一份详细的地震波形分析报告。报告应以JSON格式返回，并且**所有的报告文本内容都应该作为一个单一的字符串**，放置在 "report_content" 键下。**不要在 "report_content" 内部嵌套其他JSON对象或结构。**

波形信息：
- 台站：{report_content['waveform_info']['station']}
- 通道：{report_content['waveform_info']['channel']}
- 开始时间：{report_content['waveform_info']['start_time']}
- 结束时间：{report_content['waveform_info']['end_time']}
- 采样率：{report_content['waveform_info']['sampling_rate']} Hz
- 持续时间：{report_content['waveform_info']['duration']} 秒

分析结果：
- 信噪比 (SNR)：{report_content['analysis_results']['snr']:.2f}
- P波拾取数量：{len(p_picks)}
{p_pick_details}
- S波拾取数量：{len(s_picks)}
{s_pick_details}
{stalta_report_section}

用户查询：{report_content['user_query']}

报告应包括以下内容，并**全部整合到 "report_content" 这个字符串中**：
1. 波形质量评估
2. 震相拾取结果分析 (请详细说明P波和S波的到时、置信度等信息)
3. 对用户查询的具体回应
4. 建议和注意事项

请用中文回答。
示例JSON格式:
{{
    "report_content": "这里是您的专业分析报告内容。所有分析结果和建议都应该包含在这个字符串中，不要有额外的JSON结构。"
}}"""

            # 调用DeepSeek生成报告
            response = self.deepseek_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            logger.debug(f"DeepSeek生成报告的原始响应: {response.choices[0].message.content}")

            # 解析响应
            report_json = json.loads(response.choices[0].message.content)
            
            # 假设报告内容在 'report_content' 键下
            if isinstance(report_json, dict) and 'report_content' in report_json:
                return {'report_content': report_json['report_content']}
            else:
                # Fallback in case model doesn't follow the exact format, but it should with json_object
                logger.warning(f"DeepSeek did not return expected 'report_content' key. Raw response: {response.choices[0].message.content}")
                return {'report_content': response.choices[0].message.content}
            
        except Exception as e:
            logger.error(f"生成分析报告失败: {str(e)}")
            return {'report_content': f"生成分析报告失败: {str(e)}"}

    def analyze_stalta_parameters(self, file_path: str, sta_window: float = 0.5, lta_window: float = 10.0) -> Dict:
        """
        分析STA/LTA参数
        
        Args:
            file_path: 波形文件路径
            sta_window: 短时窗口长度（秒）
            lta_window: 长时窗口长度（秒）
            
        Returns:
            Dict: 分析结果
        """
        try:
            # 读取波形数据
            st = self.waveform_processor.get_waveform(file_path)
            if not st:
                return {'status': 'error', 'message': '无法读取波形数据'}
                
            # 计算STA/LTA比值
            stalta_result = self.waveform_processor.calculate_stalta_ratio(st, sta_window, lta_window)
            
            if stalta_result['status'] == 'error':
                return stalta_result
                
            # 获取统计信息
            stats = stalta_result['data']['statistics']
            trigger_points = stalta_result['data']['trigger_points']
            
            # 生成分析报告
            report = f"""STA/LTA参数分析报告：

1. 当前参数：
   - 短时窗口 (STA): {sta_window}秒
   - 长时窗口 (LTA): {lta_window}秒

2. 统计信息：
   - 最大比值: {stats['max_ratio']:.2f}
   - 平均比值: {stats['mean_ratio']:.2f}
   - 标准差: {stats['std_ratio']:.2f}

3. 不同阈值下的触发点数量：
"""
            for threshold, points in trigger_points.items():
                report += f"   - 阈值 {threshold}: {len(points)}个触发点\n"
                
            report += """
4. 参数建议：
"""
            # 根据统计信息给出建议
            if stats['max_ratio'] < 2.0:
                report += "   - 当前参数可能过于严格，建议减小阈值或缩短STA窗口\n"
            elif stats['max_ratio'] > 5.0:
                report += "   - 当前参数可能过于宽松，建议增加阈值或延长STA窗口\n"
            else:
                report += "   - 当前参数设置合理，可以尝试微调以获得更好的效果\n"
                
            if len(trigger_points['2.0']) > 5:
                report += "   - 触发点较多，建议增加阈值以减少误报\n"
            elif len(trigger_points['2.0']) == 0:
                report += "   - 未检测到触发点，建议降低阈值或调整窗口长度\n"
                
            return {
                'status': 'success',
                'report': {"report_content": report}, # 将报告字符串封装在字典中
                'data': stalta_result['data']
            }
            
        except Exception as e:
            logger.error(f"分析STA/LTA参数失败: {str(e)}")
            return {'status': 'error', 'message': f'分析STA/LTA参数失败: {str(e)}'} 