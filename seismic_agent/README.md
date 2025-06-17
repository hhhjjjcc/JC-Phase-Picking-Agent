# 地震学智能体震相拾取系统: JC-Phase-Picking-Agent

基于大语言模型(LLM)的智能体系统，专门用于地震学领域的震相到时拾取任务。系统通过自然语言交互，自动调用多种震相拾取算法，为用户提供专业、准确、易用的地震波形分析服务。

## 功能特点

- 智能化：LLM自动选择最优算法组合
- 专业化：集成地震学专业知识和经验
- 易用性：自然语言交互，降低技术门槛
- 可靠性：多算法融合，提供置信度评估
- 可视化：波形数据实时显示，震相到时标记清晰直观
- 参数化：支持通过自然语言调整STA/LTA等算法参数
- 报告生成：自动生成专业分析报告

## 系统要求

- Python 3.8+
- ObsPy 1.2+
- Flask 2.0+
- Flask-SocketIO 5.0+
- Plotly 5.0+
- DeepSeek API 密钥
- 其他依赖见 requirements.txt

## 安装方法

1. 克隆代码库：
```bash
git clone https://github.com/hhhjjjcc/JC-Phase-Picking-Agent
cd path/to/folder
```

2. 创建虚拟环境：
```bash
conda create -n earthquake-agent python=3.10
conda activate earthquake-agent
```

3. 安装依赖：
```bash
pip install -r requirements.txt
pip install -e .  # 以开发模式安装包
```

4. 配置环境变量：
```bash
# 创建.env文件
echo "DEEPSEEK_API_KEY=your_api_key_here" > .env
```

## 使用方法

1. 启动Web服务：
```bash
flask --app seismic_agent.web.app run
```

2. 打开浏览器访问：
```
http://localhost:5000
```

3. 上传波形文件并进行分析：
   - 支持的文件格式：SAC、MiniSEED、SEED、GSE2
   - 选择震相类型（P波/S波/两者）
   - 选择精度要求（高精度/中等精度/低精度）
   - 输入自然语言指令（如"LTA窗口16s"）进行参数调整

4. 查看分析结果：
   - 波形图显示，包含震相到时标记（竖虚线）
   - 专业分析报告，包含波形质量评估、震相拾取结果分析等

5. 示例波形见seismic_agent/example_data

## 使用示例

以下是一些通过自然语言与系统交互的示例：

### 基本震相拾取

上传波形文件后，直接点击"开始分析"按钮，系统将使用默认参数进行P波和S波拾取。

### 参数调整示例

1. **调整STA/LTA窗口**
   - 输入: "使用0.2秒的STA窗口和15秒的LTA窗口进行P波拾取"
   - 系统将调整参数并执行拾取，在波形图上标记P波到时

2. **调整触发阈值**
   - 输入: "将STA/LTA触发阈值设为1.8"
   - 系统将使用更高的阈值进行更严格的震相筛选

3. **指定震相类型**
   - 输入: "只拾取S波，并使用1.0秒的STA窗口"
   - 系统将只检测S波震相

### 分析请求示例

1. **波形质量分析**
   - 输入: "分析这个波形的信噪比并评估其质量"
   - 系统将计算信噪比并给出波形质量评估

2. **多参数综合分析**
   - 输入: "使用LTA窗口16秒，分析P波到时并计算信噪比"
   - 系统将调整参数，执行P波拾取，并计算信噪比

3. **专业问题咨询**
   - 输入: "这个波形的P波和S波时差是多少？这说明什么？"
   - 系统将计算P-S时差并给出地震学解释

### 结果解释示例

1. **拾取结果解释**
   - 输入: "解释一下为什么在20.87秒处检测到P波"
   - 系统将分析该位置的STA/LTA比值和波形特征，解释拾取原因

2. **参数效果比较**
   - 输入: "比较不同STA窗口对P波拾取的影响"
   - 系统将分析不同参数设置下的拾取效果差异

## 主要功能

### 震相拾取
- STA/LTA算法：支持单模式和双模式拾取
- 参数自适应：根据波形特征自动调整参数
- 多震相识别：同时支持P波和S波拾取

### 波形分析
- 信噪比计算：评估波形质量
- 频谱分析：分析波形频率特征
- 波形预处理：滤波、去趋势等

### 交互式界面
- 实时波形显示：上传后立即可视化
- 震相标记：清晰标注到时位置
- 参数调整：通过自然语言修改算法参数

## 项目结构
```
seismic_agent/
├── core/           # 核心功能模块
│   └── agent.py    # 智能体主类
├── tools/          # 工具库
│   ├── pickers/    # 震相拾取器
│   ├── processors/ # 波形处理工具
│   └── visualizers/ # 可视化工具
├── models/         # 模型目录
│   └── PhaseNet/   # PhaseNet模型
├── utils/          # 工具函数
│   └── waveform.py # 波形处理函数
├── web/            # Web界面
│   ├── app.py      # Flask应用
│   └── templates/  # HTML模板
├── data/           # 数据目录
│   └── uploads/    # 上传文件存储
└── tests/          # 测试用例
```

## 开发指南

### 添加新算法
1. 在 `tools/pickers/` 目录下创建新的拾取器类
2. 继承 `BasePicker` 类并实现 `pick()` 方法
3. 在 `core/agent.py` 中注册新算法

### 运行测试
```bash
pytest tests/
```

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

MIT License

## 联系方式

- 作者：Jiachen Hu
- 邮箱：jiachen.hu@zju.edu.cn
- 项目主页：https://github.com/hhhjjjcc/JC-Phase-Picking-Agent 