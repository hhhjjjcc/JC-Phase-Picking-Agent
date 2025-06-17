"""
Web应用模块
"""
import os
import logging
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename
from ..core.agent import SeismicAgent
import obspy
import numpy as np

# 设置日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# 专门为 seismic_agent 模块设置 DEBUG 级别，以便查看详细的STA/LTA计算过程
logging.getLogger('seismic_agent').setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 创建SocketIO实例
socketio = SocketIO(app)

# 创建地震智能体实例
agent = SeismicAgent()

@app.route('/')
def index():
    """渲染主页"""
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    处理文件上传
    """
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': '没有文件'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': '没有选择文件'}), 400
            
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            logger.debug(f"尝试保存文件到: {file_path}")
            file.save(file_path)
            
            logger.info(f"文件 {filename} 上传成功")
            
            # 文件上传成功，只返回文件路径，不在此处进行分析
            return jsonify({
                'status': 'success',
                'message': f'文件 {filename} 上传成功。',
                'file_path': file_path  # 返回文件路径
            })
            
    except Exception as e:
        logger.error(f"文件上传失败: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_data():
    """
    处理震相拾取和智能体分析请求
    """
    try:
        request_data = request.json
        if not request_data:
            return jsonify({'status': 'error', 'message': '请求数据为空'}), 400

        file_path = request_data.get('file_path')
        phase_type = request_data.get('phase_type', 'both')
        precision = request_data.get('precision', 'medium')
        user_query = request_data.get('user_query', '')

        if not file_path:
            return jsonify({'status': 'error', 'message': '未提供文件路径'}), 400
        if not os.path.exists(file_path):
            return jsonify({'status': 'error', 'message': '指定文件不存在'}), 404

        logger.info(f"接收到分析请求：文件路径={file_path}, 震相类型={phase_type}, 精度={precision}, 用户查询='{user_query}'")

        # 调用地震智能体处理请求
        result = agent.process_request({
            'file_path': file_path,
            'phase_type': phase_type,
            'precision': precision,
            'user_query': user_query
        })
        
        # 确保返回的JSON结构一致，并包含文件路径以便前端请求波形数据
        response_data = {
            'status': result.get('status', 'error'),
            'report': result.get('report', {}), # 直接获取报告字典
            'p_picks': result.get('p_picks', []), # 直接从顶层获取p_picks
            's_picks': result.get('s_picks', []), # 直接从顶层获取s_picks
            'snr': result.get('snr', 0.0), # 直接从顶层获取snr
            'waveform_data': result.get('waveform_data', {}), # 直接从顶层获取waveform_data
            'file_path': file_path  # 方便前端后续操作
        }
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"处理分析请求失败: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/waveform_data')
def get_waveform_data():
    """
    提供波形数据给前端绘图
    """
    file_path = request.args.get('file')
    if not file_path or not os.path.exists(file_path):
        return jsonify({'status': 'error', 'message': '波形文件未找到'}), 404

    try:
        stream = obspy.read(file_path)
        if not stream or len(stream) == 0:
            return jsonify({'status': 'error', 'message': '波形数据为空'}), 400
        
        # 提取第一个分量的数据进行绘制（通常是Z分量）
        trace = stream[0]
        data = trace.data.tolist()
        times = np.arange(0, trace.stats.npts / trace.stats.sampling_rate, 1 / trace.stats.sampling_rate).tolist()

        return jsonify({
            'status': 'success',
            'message': '波形数据获取成功',
            'data': {
                'times': times,
                'amplitudes': data,
                'sampling_rate': trace.stats.sampling_rate,
                'starttime': str(trace.stats.starttime)
            }
        })

    except Exception as e:
        logger.error(f"获取波形数据失败: {str(e)}")
        return jsonify({'status': 'error', 'message': f'获取波形数据失败: {str(e)}'}), 500

@socketio.on('connect')
def handle_connect():
    """处理WebSocket连接"""
    logger.info('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    """处理WebSocket断开连接"""
    logger.info('Client disconnected')

@socketio.on('analyze')
def handle_analysis(data):
    """处理分析请求"""
    try:
        file_path = data.get('file_path')
        requirements = data.get('requirements', {})
        
        # 执行分析
        result = agent.process_request({
            'intent': 'phase_picking',
            'file_path': file_path,
            'requirements': requirements
        })
        
        # 发送结果
        socketio.emit('analysis_result', result)
        
    except Exception as e:
        logger.error(f"分析失败: {str(e)}")
        socketio.emit('error', {'message': str(e)})

@socketio.on('batch_process')
def handle_batch_processing(data):
    """处理批量处理请求"""
    try:
        file_pattern = data.get('file_pattern')
        requirements = data.get('requirements', {})
        
        # 执行批量处理
        result = agent.process_request({
            'intent': 'batch_processing',
            'file_pattern': file_pattern,
            'requirements': requirements
        })
        
        # 发送结果
        socketio.emit('batch_result', result)
        
    except Exception as e:
        logger.error(f"批量处理失败: {str(e)}")
        socketio.emit('error', {'message': str(e)})

@app.route('/api/waveform')
def get_waveform():
    try:
        file_path = request.args.get('file')
        if not file_path:
            return jsonify({
                'success': False,
                'error': '未提供文件路径'
            })
            
        # 读取波形数据
        st = obspy.read(file_path)
        
        # 转换为Plotly格式
        traces = []
        for tr in st:
            trace = {
                'x': tr.times(),
                'y': tr.data,
                'name': f'{tr.stats.channel}',
                'type': 'scatter'
            }
            traces.append(trace)
            
        return jsonify({
            'success': True,
            'traces': traces
        })
        
    except Exception as e:
        logger.error(f"获取波形数据失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

def allowed_file(filename):
    """检查文件类型是否允许"""
    ALLOWED_EXTENSIONS = {'sac', 'mseed', 'seed', 'gse2'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    socketio.run(app, debug=True) 