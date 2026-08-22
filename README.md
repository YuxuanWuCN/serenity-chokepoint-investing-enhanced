# StockDashboard v3.0 & Serenity Chokepoint Investing Framework (Enhanced v2.1)

> **Core Research Thesis / 核心投资哲学：**
> **“Find the bottleneck first. Find the listed exposure second. Verify through evidence third. Validate with statistics fourth. Size by alpha significance fifth.”**
>
> 先定性找供应链核心瓶颈，再找上市标的敞口，经三方事实互印证，用统计回归剔除市场/规模/价值/动量噪音，通过战术波浪择时定买点，由虚拟盘闭环自校准。

---

## 1. 架构总览 (Four-Layer Quantitative Architecture)

为解决 v2.0 乘法模型导致的严重假阴性与信号稀释问题（例如：$0.7 \times 0.7 \times 0.7 \times 0.7 = 0.24$ 导致优质机会被错误丢弃），v3.0 全面重构为**四层递进式量化投研架构**：

```
┌─────────────────────────────────────────────────────────────────────────┐
│              StockDashboard v3.0 四层量化协同架构                        │
├─────────────────────────────────────────────────────────────────────────┤
│ Layer 1: 舆情与三角事实验证层 (Fact & Sentiment Verification)            │
│  ├─ [FOI] 事实/观点/推断 标注与降级系统 (Fact-Opinion-Inference)          │
│  ├─ 间接供应链上下游财务交叉互证 (Cross-Validation)                      │
│  └─ 领先指标与海关/预付款异动追踪 (Leading Indicators vs Lagged Reports) │
├─────────────────────────────────────────────────────────────────────────┤
│ Layer 2: 统计学去噪与因子回归层 (Statistical Denoising & Alpha Gate)     │
│  ├─ Fama-MacBeth 4/5 因子时序与截面回归 (MKT, SMB, HML, MOM, UMD)       │
│  ├─ HAC / Newey-West / GMM 统计稳健性检验 (异方差自相关一致估计)         │
│  └─ Alpha 显著性门禁 (p < 0.05) 与 信息比率门禁 (IR ≥ 0.3 / 0.5)         │
├─────────────────────────────────────────────────────────────────────────┤
│ Layer 3: KHunter 战术波浪与择时门禁 (Tactical Wave Timing & Risk Gate)  │
│  ├─ ZigZag 量化波段切分与 Wave 4 调整浪识别                              │
│  ├─ Fibonacci 0.500 / 0.618 黄金分割回踩确认与 ATR 动态止损              │
│  └─ 因子拥挤度监控、因子半衰期衰减模型与翻倍减仓风控规则                 │
├─────────────────────────────────────────────────────────────────────────┤
│ Layer 4: 仿真投资组合与反馈自适应闭环 (Paper Portfolio & RLSP Loop)     │
│  ├─ 100万 CNY 虚拟盘双轨跟踪 (Robust 稳健 vs Aggressive 激进)           │
│  ├─ Brier Score 概率校准引擎 (校准误差压降 ≥ 25%)                        │
│  └─ KNN 历史相似走势复盘与 LLM 提示词闭环自演进                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 模块结构与代码目录 (Directory Layout)

```text
serenity-chokepoint-investing-enhanced/
├── docs/                                  # 架构决策记录 (ADR 0001 - 0006)
│   ├── adr/
│   │   ├── 0001-two-stage-multiplicative-model.md
│   │   ├── 0002-supply-chain-ontology-by-process-node.md
│   │   ├── 0003-evidence-stage-transition-map.md
│   │   ├── 0004-fact-opinion-triangulation.md
│   │   ├── 0005-cross-validation-layer.md
│   │   └── 0006-catalyst-driven-position-sizing.md
├── src/                                   # 核心量化与分析引擎源码
│   ├── phase1/                            # Phase 1: 舆情与三角事实互证
│   │   ├── leading_indicator_tracker.py   # FOI 标签解析与领先指标追踪
│   │   └── supply_chain_cross_validator.py# 上下游财务勾稽交叉验证
│   ├── phase2/                            # Phase 2: 统计学去噪与 Alpha 门禁
│   │   ├── fama_macbeth.py                # Fama-MacBeth 4/5 因子回归模型 (HAC)
│   │   └── alpha_gate.py                  # Alpha 显著性 (p<0.05) 与 IR 门禁过滤
│   ├── phase3/                            # Phase 3: KHunter 战术波浪与择时
│   │   ├── zigzag_wave_analyzer.py        # ZigZag 极值波段切分与 ATR 计算
│   │   └── fibonacci_retracement_gate.py  # 斐波那契 0.500/0.618 回踩与止损门禁
│   ├── phase4/                            # Phase 4: 模拟闭环与自适应校准
│   │   ├── paper_portfolio_feedback.py    # 100万 CNY 双轨模拟组合 (翻倍减半)
│   │   └── calibration_rlsp_loop.py       # Brier Score 概率校准与 KNN 提示词优化
│   └── utils/
│       ├── config.py                      # 全局参数与配置阈值
│       └── data_fetcher.py                # 因子与股票数据发生/提取器
├── tests/                                 # 单元测试与端到端测试套件
│   ├── test_fama_macbeth.py
│   └── test_full_pipeline.py
├── serenity-chokepoint-investing/
│   └── SKILL.md                           # OpenClaw / Hermes 智能体部署规范
├── requirements.txt
└── README.md
```

---

## 3. 四阶段任务分工与开发周期 (12-Week Roadmap)

| 阶段 | 周期 | 核心文件 / 模块 | 核心功能 | 质量门禁 (Quality Gate / KPI) |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1: 舆情与事实验证** | W1 - W3 | `src/phase1/leading_indicator_tracker.py`<br>`src/phase1/supply_chain_cross_validator.py` | 1. `[FOI]` 事实/观点/推断 标准化提取<br>2. 上下游资本开支与合同负债交叉勾稽<br>3. 领先指标与海关/预付款异动预警 | • 文本事实标注覆盖率 100%<br>• RAG 幻觉率 < 5%<br>• 财务勾稽异常检出率 > 90% |
| **Phase 2: 因子回归与 Alpha 门禁** | W4 - W6 | `src/phase2/fama_macbeth.py`<br>`src/phase2/alpha_gate.py` | 1. CSMAR / French 四因子时序回归<br>2. Newey-West HAC 异方差自相关调整<br>3. Alpha 门禁 ($p<0.05$) 与 $IR \ge 0.3$ 硬拦截 | • OLS 遇缺失值自动安全清洗<br>• 支持 200+ 标的批量回归<br>• 单次回归延时 < 50ms |
| **Phase 3: KHunter 战术择时** | W7 - W9 | `src/phase3/zigzag_wave_analyzer.py`<br>`src/phase3/fibonacci_retracement_gate.py` | 1. ZigZag 量化高低点切分与波段跟踪<br>2. Wave 4 浪型与 0.500/0.618 黄金回踩点<br>3. 基于 2.5×ATR 的动态跟踪止损与翻倍减半 | • 波峰波谷自动切分准确率 > 95%<br>• 因子拥挤度过热预警<br>• 翻倍盈利自动锁定 50% 利润 |
| **Phase 4: 模拟闭环与自适应校准** | W10 - W12 | `src/phase4/paper_portfolio_feedback.py`<br>`src/phase4/calibration_rlsp_loop.py` | 1. 100万 CNY 虚拟盘（稳健型 / 激进型）跟踪<br>2. Brier Score 与 Cross-Entropy 概率校准<br>3. KNN 历史形态匹配驱动 LLM 提示词进化 | • 3-5年回测 Brier 误差压降 $\ge 25\%$<br>• 全流程自动化闭环回测<br>• 预留 PTrade / 实盘交易接口 |

---

## 4. 快速上手操作手册 (Quick Start Guide)

### 4.1 安装环境依赖
```bash
pip install -r requirements.txt
```

### 4.2 运行全套单元测试
```bash
python -m unittest discover -s tests -v
```

### 4.3 核心模块运行示例

#### 示例 1: 运行 Phase 2 因子回归与 Alpha 门禁检验
```python
from src.utils.data_fetcher import DataFetcher
from src.phase2.fama_macbeth import FamaMacBethRegressor
from src.phase2.alpha_gate import AlphaGate

# 1. 加载因子与标的数据
factors = DataFetcher.generate_synthetic_factors(n_periods=250)
stock = DataFetcher.generate_synthetic_stock(factors, true_alpha=0.0005)

# 2. 运行 Fama-MacBeth 回归 (HAC Newey-West)
reg = FamaMacBethRegressor(use_hac=True)
result = reg.run_time_series_ols(stock, factors)

# 3. Alpha 门禁决策
gate = AlphaGate()
decision = gate.evaluate(result)
print(f"Alpha: {result.alpha:.4f}, p-value: {result.alpha_pvalue:.4f}, IR: {result.ir:.2f}")
print(f"Decision: {decision['decision']}, Action: {decision['suggested_action']}")
```

#### 示例 2: 运行 Phase 3 战术波浪与斐波那契回踩门禁
```python
from src.phase3.fibonacci_retracement_gate import FibonacciRetracementGate

fibo_gate = FibonacciRetracementGate()
decision = fibo_gate.evaluate_wave4_retrace(
    wave3_peak=25.0,
    wave2_trough=12.0,
    current_price=17.5,
    current_atr=0.8
)
print(f"Passed: {decision.passed}, Signal: {decision.signal_name}")
print(f"Entry Zone: [{decision.target_entry_low:.2f}, {decision.target_entry_high:.2f}]")
```

---

## 5. GitHub PR 提交流程与红线守则 (Guardrails)

### 5.1 PR 标准工作流
1. **Fork 仓库：** Fork `yuyang-rgb094/serenity-chokepoint-investing-enhanced` 到个人账号。
2. **切出功能分支：**
   ```bash
   git checkout -b feature/fama-macbeth-regressor
   ```
3. **本地测试与验证：** 必须确保 `python -m unittest discover -s tests` 单元测试 100% 通过。
4. **提交 PR：** 附带明确的定量实验数据（如显著性指标、实证检验结果对比表）。

### 5.2 研发红线 (Red-Line Guardrails)
1. **样本量红线：** 任何标的的因子回归必须具备不少于 **250 个交易日（1年）** 的有效历史数据，杜绝小样本过拟合。
2. **未来函数红线：** 虚拟盘跟踪与回测系统严格逻辑隔离，严禁引入未结算前视信息。
3. **确定性代码红线：** Alpha、IR、ATR、波浪点位必须由 Python 纯代码计算，严禁依赖 LLM 直接猜测数字。
