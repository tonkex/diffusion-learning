/* ============================================================
   Chapter 1 — 정규분포
   Content + Chart implementations
   ============================================================ */

/* ---------- 1.1.1 ---------- */
CONTENT['1.1.1'] = () => String.raw`
<div class="page-title">확률 변수와 확률 분포</div>
<div class="page-subtitle">1.1 확률의 기초 — 불확실성을 수학으로 다루는 방법</div>

<div class="section">
  <div class="section-title"><span class="icon">🎲</span> 불확실성과 확률</div>
  <p>현실 세계에는 완벽하게 예측하기 어려운 일들이 많습니다. 주사위를 던졌을 때 어떤 눈이 나올지, 내일 기온이 정확히 몇 도일지 — 이런 <strong>불확실성(uncertainty)</strong>에 맞서기 위해 확률을 활용합니다.</p>
  <p>머신러닝과 딥러닝에서도 데이터를 생성한 <em>확률적 과정</em>을 모델링하는 것이 핵심입니다. 확산모델 역시 확률 분포를 학습하는 모델입니다.</p>
</div>

<div class="section">
  <div class="section-title"><span class="icon">📊</span> 확률 변수 (Random Variable)</div>
  <p><strong>확률 변수(random variable)</strong>란 얻을 수 있는 값이 확률적으로 결정되는 변수입니다. 보통 $x$로 표기합니다.</p>
  <div class="highlight-box">
    주사위의 눈을 확률 변수 $x$라 하면, 눈 3이 나올 확률을 $p(x=3)$으로 표기합니다.
  </div>
  <p>확률 변수가 가질 수 있는 모든 값에 대해 그 확률을 나열한 것이 <strong>확률 분포(probability distribution)</strong>입니다.</p>

  <h4 style="margin:16px 0 8px; color:#374151;">주사위의 확률 분포 예시</h4>
  <table class="prob-table">
    <thead><tr><th>$x$ (눈)</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th><th>6</th></tr></thead>
    <tbody><tr><td>$p(x)$</td><td>1/6</td><td>1/6</td><td>1/6</td><td>1/6</td><td>1/6</td><td>1/6</td></tr></tbody>
  </table>
  <p>이처럼 모든 면이 동일한 확률을 가질 때 <strong>균등 분포(uniform distribution)</strong>라고 합니다.</p>
</div>

<div class="section">
  <div class="section-title"><span class="icon">📐</span> 확률 분포의 조건</div>
  <p>어떤 함수 $p(x)$가 확률 분포가 되려면 두 조건을 만족해야 합니다:</p>
  <div class="math-block">
    $$0 \le p(x_k) \le 1 \quad (k = 1, 2, \ldots, N)$$
  </div>
  <div class="math-block">
    $$\sum_{k=1}^{N} p(x_k) = 1$$
  </div>
  <div class="highlight-box green">
    <strong>핵심:</strong> 각 확률은 0~1 사이이고, 모든 확률의 합은 반드시 1이어야 합니다.
  </div>
</div>

<div class="section">
  <div class="section-title"><span class="icon">🎮</span> 주사위 시뮬레이션</div>
  <p>주사위를 여러 번 굴려 경험적 확률 분포를 관찰해 보세요. 시행 횟수가 늘어날수록 균등 분포에 가까워지는지 확인하세요.</p>
  <div class="interactive-panel">
    <div class="panel-header">🎲 주사위 시뮬레이터</div>
    <div class="panel-body" style="flex-direction:column;">
      <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
        <button class="btn btn-primary" onclick="rollDice(1)">1번 굴리기</button>
        <button class="btn btn-secondary" onclick="rollDice(10)">10번 굴리기</button>
        <button class="btn btn-secondary" onclick="rollDice(100)">100번 굴리기</button>
        <button class="btn btn-secondary" onclick="resetDice()">초기화</button>
        <span style="font-size:0.82rem;color:#64748b;margin-left:8px;">총 시행: <strong id="dice-count">0</strong></span>
      </div>
      <div class="dice-grid" id="dice-display">
        <div class="die" id="die-face">?</div>
      </div>
      <canvas id="dice-chart" height="180"></canvas>
    </div>
  </div>
</div>
`;

/* ---------- 1.1.2 ---------- */
CONTENT['1.1.2'] = () => String.raw`
<div class="page-title">확률 분포의 종류</div>
<div class="page-subtitle">1.1 확률의 기초 — 이산 vs 연속</div>

<div class="section">
  <div class="section-title"><span class="icon">🔢</span> 이산 vs 연속 확률 분포</div>
  <p>확률 분포는 확률 변수의 성질에 따라 두 가지로 나뉩니다.</p>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:16px 0;">
    <div class="highlight-box">
      <strong>이산 확률 분포 (Discrete)</strong><br>
      확률 변수가 1, 2, 3처럼 딱 떨어지는 값을 취합니다.<br>
      예: 주사위, 동전 던지기
    </div>
    <div class="highlight-box purple">
      <strong>연속 확률 분포 (Continuous)</strong><br>
      확률 변수가 연속적인 실수값을 취합니다.<br>
      예: 키, 기온, 측정 오차
    </div>
  </div>
  <p>두 분포의 가장 큰 차이는 <strong>$p(x)$의 의미</strong>입니다. 이산 분포에서 $p(x)$는 <em>확률 그 자체</em>이지만, 연속 분포에서 $p(x)$는 <em>확률 밀도(density)</em>로 적분해야 확률이 됩니다.</p>
</div>

<div class="section">
  <div class="section-title"><span class="icon">📊</span> 이산과 연속 — 나란히 비교하기</div>
  <p>슬라이더로 구간 $[a, b]$를 바꿔가며, 연속 분포에서 확률이 <strong>곡선 아래 면적</strong>임을 직접 확인하세요.</p>

  <div class="interactive-panel">
    <div class="panel-header">📊 그림 1-2 · 이산 확률 분포 vs 연속 확률 분포</div>
    <div class="panel-body" style="flex-direction:column;gap:0;">
      <div style="display:flex;gap:24px;flex-wrap:wrap;">

        <!-- ===== LEFT: Discrete ===== -->
        <div style="flex:1;min-width:260px;">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
            <span class="tag">이산(Discrete)</span>
            <span style="font-size:0.82rem;color:#374151;">주사위 눈의 확률</span>
          </div>
          <canvas id="discrete-chart" height="230"></canvas>
          <div class="highlight-box" style="margin-top:10px;font-size:0.82rem;line-height:1.8;">
            ✅ $p(x)$ = <strong>확률</strong> (막대 높이 그 자체)<br>
            $$\sum_{k=1}^{6} p(x_k) = 6 \times \tfrac{1}{6} = 1$$
          </div>
        </div>

        <!-- ===== RIGHT: Continuous ===== -->
        <div style="flex:1;min-width:260px;">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
            <span class="tag purple">연속(Continuous)</span>
            <span style="font-size:0.82rem;color:#374151;">키(cm) · $\mathcal{N}(174,\,6^2)$</span>
          </div>
          <canvas id="continuous-chart" height="230"></canvas>
          <div style="margin-top:10px;">
            <div style="margin-bottom:8px;">
              <div class="ctrl-label">왼쪽 a = <span id="height-a-val">170</span> cm</div>
              <input type="range" id="height-a-slider" min="150" max="200" step="1" value="170"
                style="width:100%;" oninput="updateHeightRange();">
            </div>
            <div>
              <div class="ctrl-label">오른쪽 b = <span id="height-b-val">180</span> cm</div>
              <input type="range" id="height-b-slider" min="150" max="200" step="1" value="180"
                style="width:100%;" oninput="updateHeightRange();">
            </div>
          </div>
          <div style="display:flex;gap:10px;margin-top:10px;align-items:stretch;">
            <div class="stat-card" style="flex:0 0 auto;min-width:100px;display:flex;flex-direction:column;justify-content:center;">
              <div class="label">P(a ≤ x ≤ b)</div>
              <div class="value" id="height-prob" style="color:#ef4444;">—</div>
            </div>
            <div class="highlight-box purple" style="flex:1;margin:0;font-size:0.82rem;line-height:1.8;">
              ✅ $p(x)$ = <strong>밀도</strong> (면적이 확률)<br>
              $$P(a \le x \le b) = \int_a^b p(x)\,dx$$
            </div>
          </div>
        </div>

      </div>
    </div>
  </div>
</div>

<div class="section">
  <div class="section-title"><span class="icon">⚠️</span> 확률 밀도에서 주의할 점</div>
  <div class="highlight-box amber">
    <strong>$p(x) > 1$ 이 될 수 있습니다!</strong> 연속 확률 변수에서 $p(x)$는 확률이 아닌 <em>밀도</em>입니다.
    예컨대 $\sigma = 0.1$인 정규 분포의 최댓값은 $\approx 3.99$입니다. 그러나 전체 면적의 합은 항상 1입니다.
  </div>
  <div class="math-block">
    $$p(x) \ge 0, \qquad \int_{-\infty}^{\infty} p(x)\, dx = 1$$
  </div>
</div>

<div class="section">
  <div class="section-title"><span class="icon">🔗</span> 앞으로 배울 분포들</div>
  <ul class="content-list">
    <li><a class="ch-link" href="#" onclick="loadSection('1.2.1',NAV[0],NAV[0].subs[1],NAV[0].subs[1].sections[0]);return false;">📍 정규 분포 (Chapter 1.2)</a> — 가장 중요한 연속 확률 분포</li>
    <li><a class="ch-link" href="#" onclick="alert('Chapter 2에서 학습합니다.')">📍 최대 가능도 추정 (Chapter 2)</a> — 분포의 매개변수를 데이터로 추정</li>
    <li><a class="ch-link" href="#" onclick="alert('Chapter 3에서 학습합니다.')">📍 다변량 정규 분포 (Chapter 3)</a> — 고차원 데이터로 확장</li>
    <li><a class="ch-link" href="#" onclick="alert('Chapter 4에서 학습합니다.')">📍 가우스 혼합 모델 (Chapter 4)</a> — 여러 정규 분포의 혼합</li>
  </ul>
</div>
`;

/* ---------- 1.1.3 ---------- */
CONTENT['1.1.3'] = () => String.raw`
<div class="page-title">기대값과 분산</div>
<div class="page-subtitle">1.1 확률의 기초 — 분포의 특성을 요약하는 두 수치</div>

<div class="section">
  <div class="section-title"><span class="icon">⚖️</span> 기대값 (Expected Value)</div>
  <p><strong>기대값(expected value)</strong> $\mathbb{E}[x]$은 한 번의 관측으로 얻을 수 있는 값의 평균입니다. 무한히 많이 관측했을 때의 평균값이라고 직관적으로 이해할 수 있습니다.</p>
  <div class="math-block">
    $$\mathbb{E}[x] = \sum_{k=1}^{N} x_k\, p(x_k) \quad \text{(이산)}$$
    $$\mathbb{E}[x] = \int_{-\infty}^{\infty} x\, p(x)\, dx \quad \text{(연속)}$$
  </div>
  <div class="highlight-box">
    주사위의 기대값: $\mu = 1 \cdot \tfrac{1}{6} + 2 \cdot \tfrac{1}{6} + \cdots + 6 \cdot \tfrac{1}{6} = 3.5$
  </div>
</div>

<div class="section">
  <div class="section-title"><span class="icon">📏</span> 분산 (Variance)</div>
  <p><strong>분산(variance)</strong> $\text{Var}[x]$은 값이 기대값 주위에서 얼마나 퍼져 있는지를 나타냅니다. 분산의 제곱근이 <strong>표준편차</strong> $\sigma$입니다.</p>
  <div class="math-block">
    $$\text{Var}[x] = \mathbb{E}[(x - \mu)^2] = \mathbb{E}[x^2] - \mu^2$$
  </div>
  <div class="math-block">
    $$\text{Var}[x] = \sum_{k=1}^{N} (x_k - \mu)^2\, p(x_k) \quad \text{(이산)}$$
  </div>
  <p>주사위의 분산:</p>
  <div class="math-block">
    $$\sigma^2 = \sum_{k=1}^{6}(k - 3.5)^2 \cdot \tfrac{1}{6} = \tfrac{35}{12} \approx 2.917$$
  </div>
  <div class="highlight-box green">
    분산이 <strong>크면</strong> 데이터가 넓게 퍼져 있고, <strong>작으면</strong> 기대값 주위에 몰려 있습니다. 이것이 정규 분포의 $\sigma$(표준편차)가 나타내는 의미입니다.
    <br><a class="ch-link" href="#" onclick="loadSection('1.2.3',NAV[0],NAV[0].subs[1],NAV[0].subs[1].sections[2]);return false;">→ 1.2.3 매개변수의 역할에서 직접 확인</a>
  </div>
</div>

<div class="section">
  <div class="section-title"><span class="icon">📊</span> 기대값의 선형성</div>
  <p>기대값은 다음의 편리한 성질을 가집니다 (이후 챕터에서 자주 사용됩니다):</p>
  <div class="math-block">
    $$\mathbb{E}[ax + b] = a\mathbb{E}[x] + b$$
    $$\mathbb{E}[x + y] = \mathbb{E}[x] + \mathbb{E}[y]$$
  </div>
  <div class="highlight-box amber">
    <strong>독립 변수의 분산:</strong> $x$, $y$가 독립이면 $\text{Var}[x + y] = \text{Var}[x] + \text{Var}[y]$. 이 성질이 <a class="ch-link" href="#" onclick="loadSection('1.3.1',NAV[0],NAV[0].subs[2],NAV[0].subs[2].sections[0]);return false;">중심 극한 정리</a>의 핵심입니다.
  </div>
</div>
`;

/* ---------- 1.2.1 ---------- */
CONTENT['1.2.1'] = () => String.raw`
<div class="page-title">정규 분포의 확률 밀도 함수</div>
<div class="page-subtitle">1.2 정규 분포 — 자연계에서 가장 자주 등장하는 분포</div>

<div class="section">
  <div class="section-title"><span class="icon">🔔</span> 정규 분포 (Gaussian Distribution)</div>
  <p>정규 분포는 <strong>평균 $\mu$</strong>와 <strong>표준편차 $\sigma$</strong> 두 매개변수만으로 결정되는 연속 확률 분포입니다. 종 모양(bell curve)의 대칭 분포입니다.</p>
  <div class="math-block">
    $$\mathcal{N}(x;\,\mu,\,\sigma) = \frac{1}{\sqrt{2\pi}\,\sigma} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$
    <div class="formula-label">정규 분포의 확률 밀도 함수 (PDF)</div>
  </div>
  <div class="highlight-box">
    <strong>표기법:</strong> "$x$가 평균 $\mu$, 표준편차 $\sigma$인 정규 분포를 따른다"를 $x \sim \mathcal{N}(\mu, \sigma^2)$ 또는 $x \sim \mathcal{N}(\mu, \sigma)$로 씁니다.
  </div>
</div>

<div class="section">
  <div class="section-title"><span class="icon">🔍</span> 수식 분해하기</div>
  <ul class="content-list">
    <li><strong>$\frac{1}{\sqrt{2\pi}\,\sigma}$</strong> — 정규화 상수. 전체 면적이 1이 되도록 조정합니다.</li>
    <li><strong>$\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$</strong> — 가우스 함수. $x=\mu$에서 최댓값 1, $x$가 $\mu$에서 멀어질수록 0에 가까워집니다.</li>
    <li><strong>$(x - \mu)^2$</strong> — $\mu$로부터의 거리 제곱. 대칭성을 만들어냅니다.</li>
    <li><strong>$2\sigma^2$</strong> — 분모. $\sigma$가 클수록 분포가 완만(넓게)해집니다.</li>
  </ul>
</div>

<div class="section">
  <div class="section-title"><span class="icon">📐</span> 표준 정규 분포</div>
  <p>$\mu = 0$, $\sigma = 1$인 특별한 경우를 <strong>표준 정규 분포</strong>라 합니다.</p>
  <div class="math-block">
    $$\mathcal{N}(x;\,0,\,1) = \frac{1}{\sqrt{2\pi}} \exp\!\left(-\frac{x^2}{2}\right)$$
  </div>

  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:16px;">
    <div class="stat-card">
      <div class="label">평균 $\mu$로부터 $\pm\sigma$ 이내</div>
      <div class="value" style="font-size:1.1rem;">68.3%</div>
    </div>
    <div class="stat-card">
      <div class="label">$\pm 2\sigma$ 이내</div>
      <div class="value" style="font-size:1.1rem;">95.4%</div>
    </div>
    <div class="stat-card">
      <div class="label">$\pm 3\sigma$ 이내</div>
      <div class="value" style="font-size:1.1rem;">99.7%</div>
    </div>
  </div>
  <p style="margin-top:12px;">이것을 <strong>68-95-99.7 법칙</strong> (또는 경험 법칙)이라 합니다.</p>
</div>

<div class="section">
  <div class="section-title"><span class="icon">🔗</span> 정규 분포의 기대값과 분산</div>
  <p>정규 분포의 기대값과 분산은 매개변수와 직접 연결됩니다:</p>
  <div class="math-block">
    $$\mathbb{E}[x] = \mu, \qquad \text{Var}[x] = \sigma^2$$
  </div>
  <div class="highlight-box green">
    즉, 매개변수 $\mu$가 바로 평균이고, $\sigma$가 표준편차입니다. 직접 확인하려면 <a class="ch-link" href="#" onclick="loadSection('1.2.3',NAV[0],NAV[0].subs[1],NAV[0].subs[1].sections[2]);return false;">→ 1.2.3 매개변수의 역할</a>을 참고하세요.
  </div>
</div>
`;

/* ---------- 1.2.2 ---------- */
CONTENT['1.2.2'] = () => String.raw`
<div class="page-title">정규 분포 시각화</div>
<div class="page-subtitle">1.2 정규 분포 — 표준 정규 분포를 코드로 그려보기</div>

<div class="section">
  <div class="section-title"><span class="icon">💻</span> Python 코드</div>
  <p>다음은 표준 정규 분포를 NumPy와 Matplotlib으로 그리는 코드입니다:</p>
  <pre style="background:#1e293b;color:#e2e8f0;padding:16px;border-radius:8px;font-size:0.82rem;line-height:1.7;overflow-x:auto;"><code>import numpy as np
import matplotlib.pyplot as plt

def normal(x, mu=0, sigma=1):
    y = 1 / (np.sqrt(2 * np.pi) * sigma) * \
        np.exp(-(x - mu)**2 / (2 * sigma**2))
    return y

x = np.linspace(-5, 5, 100)
y = normal(x)          # mu=0, sigma=1

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('p(x)')
plt.show()</code></pre>
</div>

<div class="section">
  <div class="section-title"><span class="icon">📊</span> 인터렉티브 시각화</div>
  <p>표준 정규 분포 $\mathcal{N}(x; 0, 1)$와 특정 구간의 확률(면적)을 확인해 보세요.</p>
  <div class="interactive-panel">
    <div class="panel-header">📈 표준 정규 분포 · 구간 확률 계산</div>
    <div class="panel-body">
      <div class="chart-wrap"><canvas id="normal-basic-chart" height="220"></canvas></div>
      <div class="controls-wrap">
        <div class="ctrl-group">
          <div class="ctrl-label">구간 왼쪽: a = <span id="a-val">-1.00</span></div>
          <input type="range" id="a-slider" min="-4" max="4" step="0.1" value="-1"
            oninput="document.getElementById('a-val').textContent=parseFloat(this.value).toFixed(2); updateNormalBasic();">
        </div>
        <div class="ctrl-group">
          <div class="ctrl-label">구간 오른쪽: b = <span id="b-val">1.00</span></div>
          <input type="range" id="b-slider" min="-4" max="4" step="0.1" value="1"
            oninput="document.getElementById('b-val').textContent=parseFloat(this.value).toFixed(2); updateNormalBasic();">
        </div>
        <div class="stat-grid" style="grid-template-columns:1fr;">
          <div class="stat-card">
            <div class="label">P(a ≤ x ≤ b)</div>
            <div class="value" id="prob-value">68.3%</div>
          </div>
          <div class="stat-card">
            <div class="label">최댓값 p(0)</div>
            <div class="value">0.399</div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
`;

/* ---------- 1.2.3 ---------- */
CONTENT['1.2.3'] = () => String.raw`
<div class="page-title">매개변수의 역할</div>
<div class="page-subtitle">1.2 정규 분포 — μ와 σ가 분포 모양에 어떤 영향을 주는가?</div>

<div class="section">
  <div class="section-title"><span class="icon">🎛️</span> μ: 위치 매개변수</div>
  <p>$\mu$는 분포의 <strong>중심 위치</strong>를 결정합니다. $\mu$가 바뀌면 분포 전체가 좌우로 이동하지만 <em>모양은 변하지 않습니다</em>.</p>
</div>

<div class="section">
  <div class="section-title"><span class="icon">🎛️</span> σ: 척도 매개변수</div>
  <p>$\sigma$는 분포의 <strong>퍼짐(너비)</strong>을 결정합니다. $\sigma$가 클수록 분포가 넓고 납작해지며, 작을수록 좁고 뾰족해집니다.</p>
</div>

<div class="section">
  <div class="section-title"><span class="icon">📊</span> 인터렉티브 탐색기</div>
  <p>슬라이더를 움직여 $\mu$와 $\sigma$가 분포에 미치는 영향을 직접 확인하세요.</p>
  <div class="interactive-panel">
    <div class="panel-header">🎛️ 매개변수 탐색기 (슬라이더)</div>
    <div class="panel-body">
      <div class="chart-wrap"><canvas id="param-chart" height="250"></canvas></div>
      <div class="controls-wrap">
        <div class="ctrl-group">
          <div class="ctrl-label">μ (평균): <span id="mu-val">0.0</span></div>
          <div class="ctrl-row">
            <input type="range" id="mu-slider" min="-4" max="4" step="0.1" value="0"
              oninput="document.getElementById('mu-val').textContent=parseFloat(this.value).toFixed(1); updateParamChart();">
            <span class="val-badge" id="mu-badge">0.0</span>
          </div>
        </div>
        <div class="ctrl-group">
          <div class="ctrl-label">σ (표준편차): <span id="sigma-val">1.0</span></div>
          <div class="ctrl-row">
            <input type="range" id="sigma-slider" min="0.2" max="3" step="0.1" value="1"
              oninput="document.getElementById('sigma-val').textContent=parseFloat(this.value).toFixed(1); updateParamChart();">
            <span class="val-badge" id="sigma-badge">1.0</span>
          </div>
        </div>
        <div class="ctrl-group" style="margin-top:8px;">
          <div class="ctrl-label">비교용 분포 표시</div>
          <label style="font-size:0.8rem;color:#374151;cursor:pointer;">
            <input type="checkbox" id="show-ref" checked onchange="updateParamChart()"> 기준 분포 (μ=0, σ=1)
          </label>
        </div>
        <div class="stat-grid" style="grid-template-columns:1fr;margin-top:12px;">
          <div class="stat-card">
            <div class="label">p(μ) 최댓값</div>
            <div class="value" id="peak-val">0.399</div>
          </div>
          <div class="stat-card">
            <div class="label">분산 σ²</div>
            <div class="value" id="var-val">1.00</div>
          </div>
        </div>
        <div class="highlight-box" style="margin-top:12px;font-size:0.8rem;">
          <strong>관찰:</strong> σ가 2배 → 최댓값은 절반, 너비는 2배
        </div>
      </div>
    </div>
  </div>
</div>

<div class="section">
  <div class="section-title"><span class="icon">📐</span> 수식 요약</div>
  <div class="math-block">
    $$\mathcal{N}(x;\,\mu,\,\sigma) = \frac{1}{\sqrt{2\pi}\,\sigma} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$
  </div>
  <ul class="content-list">
    <li>$\mu$ 변화 → 분포가 수평 이동 (모양 불변)</li>
    <li>$\sigma$ 변화 → 분포의 폭과 높이 변화 (정규화 상수 때문)</li>
    <li>$\sigma \to 0$ → 디랙 델타 함수 $\delta(x-\mu)$에 수렴</li>
  </ul>
  <div class="highlight-box purple">
    이 두 매개변수가 바로 <a class="ch-link" href="#" onclick="alert('Chapter 2에서 학습합니다.')">최대 가능도 추정(MLE, Chapter 2)</a>을 통해 데이터로부터 학습하는 대상입니다.
  </div>
</div>
`;

/* ---------- 1.3.1 ---------- */
CONTENT['1.3.1'] = () => String.raw`
<div class="page-title">중심 극한 정리란?</div>
<div class="page-subtitle">1.3 중심 극한 정리 — 정규 분포가 어디서나 나타나는 이유</div>

<div class="section">
  <div class="section-title"><span class="icon">🌟</span> 통계학에서 가장 아름다운 정리</div>
  <p><strong>중심 극한 정리(Central Limit Theorem, CLT)</strong>는 정규 분포가 가장 중요한 확률 분포인 이유를 설명합니다. 직관적으로 말하면:</p>
  <div class="highlight-box" style="font-size:1rem;text-align:center;">
    <strong>어떤 분포를 따르는 확률 변수든,<br>
    표본 평균은 표본 크기가 커질수록 정규 분포에 가까워진다.</strong>
  </div>
</div>

<div class="section">
  <div class="section-title"><span class="icon">📐</span> 수식으로 보는 CLT</div>
  <p>같은 분포에서 독립적으로 추출한 $N$개의 표본을 $x^{(1)}, x^{(2)}, \ldots, x^{(N)}$이라 할 때, <strong>표본 평균</strong>은:</p>
  <div class="math-block">
    $$\bar{x} = \frac{x^{(1)} + x^{(2)} + \cdots + x^{(N)}}{N}$$
  </div>
  <p>원래 분포의 평균이 $\mu$, 분산이 $\sigma^2$이면:</p>
  <div class="math-block">
    $$\mathbb{E}[\bar{x}] = \mu, \qquad \text{Var}[\bar{x}] = \frac{\sigma^2}{N}$$
  </div>
  <p>그리고 $N \to \infty$이면:</p>
  <div class="math-block">
    $$\bar{x} \xrightarrow{d} \mathcal{N}\!\left(\mu,\, \frac{\sigma}{\sqrt{N}}\right)$$
  </div>
  <div class="highlight-box green">
    <strong>핵심:</strong> 원래 분포가 무엇이든 상관없이, $N$이 커지면 표본 평균은 정규 분포로 수렴합니다.
  </div>
</div>

<div class="section">
  <div class="section-title"><span class="icon">🔬</span> CLT가 중요한 이유</div>
  <div class="steps">
    <div class="step active"><div class="step-num">1</div>어떤 분포에서든 표본 추출</div>
    <div class="step active"><div class="step-num">2</div>N개씩 묶어 평균 계산</div>
    <div class="step active"><div class="step-num">3</div>N이 크면 정규 분포!</div>
    <div class="step active"><div class="step-num">4</div>통계 추론 가능</div>
  </div>
  <ul class="content-list">
    <li>현실의 많은 현상이 정규 분포를 따르는 이유를 설명합니다 (<a class="ch-link" href="#" onclick="loadSection('1.5.1',NAV[0],NAV[0].subs[4],NAV[0].subs[4].sections[0]);return false;">→ 1.5 우리 주변의 정규 분포</a>)</li>
    <li>통계 검정과 신뢰구간의 이론적 근거입니다</li>
    <li>표본 크기가 클수록 추정이 정확해지는 이유입니다</li>
  </ul>
  <div class="highlight-box purple">
    직접 시뮬레이션으로 확인해 보세요! → <a class="ch-link" href="#" onclick="loadSection('1.3.2',NAV[0],NAV[0].subs[2],NAV[0].subs[2].sections[1]);return false;">1.3.2 CLT 시뮬레이션</a>
  </div>
</div>
`;

/* ---------- 1.3.2 ---------- */
CONTENT['1.3.2'] = () => String.raw`
<div class="page-title">CLT 시뮬레이션</div>
<div class="page-subtitle">1.3 중심 극한 정리 — 균등 분포의 표본 평균이 정규 분포로 수렴하는 과정</div>

<div class="section">
  <div class="section-title"><span class="icon">🧪</span> 실험 설계</div>
  <p>균등 분포 $\text{Uniform}(0, 1)$에서 $N$개를 표본 추출하여 평균을 구하는 실험을 10,000번 반복합니다.</p>
  <ul class="content-list">
    <li>$N = 1$: 균등 분포 그 자체 (직사각형)</li>
    <li>$N = 2$: 삼각형 분포</li>
    <li>$N \ge 4$: 정규 분포와 비슷해집니다</li>
    <li>$N = 30$: 일반적으로 충분히 정규 분포에 가깝다고 봅니다</li>
  </ul>
</div>

<div class="section">
  <div class="section-title"><span class="icon">🎮</span> 인터렉티브 시뮬레이션</div>
  <div class="interactive-panel">
    <div class="panel-header">🔬 중심 극한 정리 실험 (균등 분포)</div>
    <div class="panel-body">
      <div class="chart-wrap"><canvas id="clt-chart" height="260"></canvas></div>
      <div class="controls-wrap">
        <div class="ctrl-group">
          <div class="ctrl-label">표본 크기 N = <span id="clt-n-val">4</span></div>
          <input type="range" id="clt-n-slider" min="1" max="50" step="1" value="4"
            oninput="document.getElementById('clt-n-val').textContent=this.value;">
        </div>
        <div class="ctrl-group">
          <div class="ctrl-label">실험 횟수</div>
          <select id="clt-trials" style="width:100%;padding:6px;border:1px solid #e2e8f0;border-radius:6px;font-size:0.83rem;">
            <option value="1000">1,000번</option>
            <option value="5000" selected>5,000번</option>
            <option value="10000">10,000번</option>
          </select>
        </div>
        <button class="btn btn-primary" style="width:100%;margin-bottom:10px;" onclick="runCLT()">▶ 시뮬레이션 실행</button>
        <div class="stat-grid" style="grid-template-columns:1fr 1fr;">
          <div class="stat-card">
            <div class="label">이론 평균</div>
            <div class="value" id="clt-theory-mu">0.50</div>
          </div>
          <div class="stat-card">
            <div class="label">실험 평균</div>
            <div class="value" id="clt-emp-mu">—</div>
          </div>
          <div class="stat-card">
            <div class="label">이론 σ</div>
            <div class="value" id="clt-theory-sigma">0.289</div>
          </div>
          <div class="stat-card">
            <div class="label">실험 σ</div>
            <div class="value" id="clt-emp-sigma">—</div>
          </div>
        </div>
        <div class="highlight-box" style="margin-top:10px;font-size:0.78rem;">
          이론값: 균등 분포의 $\mu=0.5$, $\sigma=1/\sqrt{12N}$
        </div>
      </div>
    </div>
  </div>
</div>
`;

/* ---------- 1.4.1 ---------- */
CONTENT['1.4.1'] = () => String.raw`
<div class="page-title">표본 합의 기대값과 분산</div>
<div class="page-subtitle">1.4 표본 합의 확률 분포</div>

<div class="section">
  <div class="section-title"><span class="icon">➕</span> 표본 합 정의</div>
  <p>표본 평균 대신 <strong>표본 합(sample sum)</strong>을 생각해봅니다:</p>
  <div class="math-block">
    $$s = x^{(1)} + x^{(2)} + \cdots + x^{(N)} = N\bar{x}$$
  </div>
</div>

<div class="section">
  <div class="section-title"><span class="icon">📐</span> 표본 합의 기대값</div>
  <div class="math-block">
    $$\mathbb{E}[s] = \mathbb{E}[N\bar{x}] = N\,\mathbb{E}[\bar{x}] = N\mu$$
  </div>
  <p>도출 과정:</p>
  <div class="math-block">
    $$\mathbb{E}[N\bar{x}] = \int N\bar{x}\, p(\bar{x})\, d\bar{x} = N\int \bar{x}\, p(\bar{x})\, d\bar{x} = N\mathbb{E}[\bar{x}] = N\mu$$
  </div>
</div>

<div class="section">
  <div class="section-title"><span class="icon">📐</span> 표본 합의 분산</div>
  <div class="math-block">
    $$\text{Var}[s] = \text{Var}[N\bar{x}] = N^2\,\text{Var}[\bar{x}] = N^2 \cdot \frac{\sigma^2}{N} = N\sigma^2$$
  </div>
  <p>따라서 표본 합의 표준편차는 $\sqrt{N}\,\sigma$ 입니다. 반면 표본 평균의 표준편차는 $\sigma/\sqrt{N}$이었습니다.</p>
  <div class="highlight-box amber">
    <strong>비교:</strong><br>
    표본 평균의 분산: $\sigma^2/N$ → $N$이 증가하면 <em>감소</em><br>
    표본 합의 분산: $N\sigma^2$ → $N$이 증가하면 <em>증가</em>
  </div>
</div>

<div class="section">
  <div class="section-title"><span class="icon">📊</span> 표본 합의 정규 분포 수렴</div>
  <p>CLT에 의해 표본 합도 $N$이 커질수록 정규 분포에 가까워집니다:</p>
  <div class="math-block">
    $$s \xrightarrow{d} \mathcal{N}\!\left(N\mu,\, \sqrt{N}\,\sigma\right)$$
  </div>
  <div class="highlight-box green">
    직접 시뮬레이션으로 확인: <a class="ch-link" href="#" onclick="loadSection('1.4.2',NAV[0],NAV[0].subs[3],NAV[0].subs[3].sections[1]);return false;">→ 1.4.2 표본 합 시뮬레이션</a>
  </div>
</div>
`;

/* ---------- 1.4.2 ---------- */
CONTENT['1.4.2'] = () => String.raw`
<div class="page-title">표본 합 시뮬레이션</div>
<div class="page-subtitle">1.4 표본 합의 확률 분포 — 합도 정규 분포로 수렴한다</div>

<div class="section">
  <div class="section-title"><span class="icon">🧪</span> 실험: 균등 분포 표본의 합</div>
  <p>균등 분포 $\text{Uniform}(0,1)$에서 $N$개를 뽑아 <em>합</em>을 구하는 실험을 반복합니다.</p>
  <p>균등 분포의 $\mu = 0.5$, $\sigma^2 = 1/12$이므로, 표본 합의 이론 분포는:</p>
  <div class="math-block">
    $$s \sim \mathcal{N}\!\left(N \cdot 0.5,\; \sqrt{N/12}\right)$$
  </div>
  <div class="interactive-panel">
    <div class="panel-header">🔬 표본 합 시뮬레이션</div>
    <div class="panel-body">
      <div class="chart-wrap"><canvas id="sum-chart" height="260"></canvas></div>
      <div class="controls-wrap">
        <div class="ctrl-group">
          <div class="ctrl-label">표본 크기 N = <span id="sum-n-val">5</span></div>
          <input type="range" id="sum-n-slider" min="1" max="30" step="1" value="5"
            oninput="document.getElementById('sum-n-val').textContent=this.value;">
        </div>
        <button class="btn btn-green" style="width:100%;margin-bottom:10px;" onclick="runSumSim()">▶ 실행</button>
        <div class="stat-grid" style="grid-template-columns:1fr 1fr;">
          <div class="stat-card"><div class="label">이론 평균</div><div class="value" id="sum-theory-mu">2.50</div></div>
          <div class="stat-card"><div class="label">실험 평균</div><div class="value" id="sum-emp-mu">—</div></div>
          <div class="stat-card"><div class="label">이론 σ</div><div class="value" id="sum-theory-sigma">0.645</div></div>
          <div class="stat-card"><div class="label">실험 σ</div><div class="value" id="sum-emp-sigma">—</div></div>
        </div>
      </div>
    </div>
  </div>
</div>
`;

/* ---------- 1.4.3 ---------- */
CONTENT['1.4.3'] = () => String.raw`
<div class="page-title">균등 분포의 평균과 분산</div>
<div class="page-subtitle">1.4 표본 합의 확률 분포 — 균등 분포의 통계량 도출</div>

<div class="section">
  <div class="section-title"><span class="icon">📐</span> 균등 분포 Uniform(0, 1)</div>
  <p>균등 분포의 확률 밀도 함수:</p>
  <div class="math-block">
    $$p(x) = \begin{cases} 1 & 0 \le x \le 1 \\ 0 & \text{otherwise} \end{cases}$$
  </div>
</div>

<div class="section">
  <div class="section-title"><span class="icon">⚖️</span> 균등 분포의 평균 도출</div>
  <div class="math-block">
    $$\mu = \int_{-\infty}^{\infty} x\, p(x)\, dx = \int_{0}^{1} x \cdot 1\, dx = \left[\frac{x^2}{2}\right]_0^1 = \frac{1}{2}$$
  </div>
</div>

<div class="section">
  <div class="section-title"><span class="icon">📏</span> 균등 분포의 분산 도출</div>
  <div class="math-block">
    $$\sigma^2 = \int_{0}^{1} \left(x - \frac{1}{2}\right)^2 dx
    = \int_{0}^{1} \left(x^2 - x + \frac{1}{4}\right) dx
    = \left[\frac{x^3}{3} - \frac{x^2}{2} + \frac{x}{4}\right]_0^1
    = \frac{1}{3} - \frac{1}{2} + \frac{1}{4} = \frac{1}{12}$$
  </div>
  <div class="stat-grid">
    <div class="stat-card"><div class="label">평균 μ</div><div class="value">1/2 = 0.5</div></div>
    <div class="stat-card"><div class="label">분산 σ²</div><div class="value">1/12 ≈ 0.083</div></div>
    <div class="stat-card"><div class="label">표준편차 σ</div><div class="value">1/√12 ≈ 0.289</div></div>
  </div>
  <div class="highlight-box green">
    이 값들이 CLT 시뮬레이션의 이론값으로 사용됩니다. <a class="ch-link" href="#" onclick="loadSection('1.3.2',NAV[0],NAV[0].subs[2],NAV[0].subs[2].sections[1]);return false;">→ 1.3.2 CLT 시뮬레이션 확인</a>
  </div>
</div>
`;

/* ---------- 1.5.1 ---------- */
CONTENT['1.5.1'] = () => String.raw`
<div class="page-title">우리 주변의 정규 분포</div>
<div class="page-subtitle">1.5 — 중심 극한 정리가 자연과 사회를 설명하는 방식</div>

<div class="section">
  <div class="section-title"><span class="icon">🌍</span> 왜 정규 분포가 어디서나 나타날까?</div>
  <p>중심 극한 정리에 의해, 여러 독립적인 요인들의 <em>누적 합</em>으로 결정되는 현상은 정규 분포를 따르는 경향이 있습니다.</p>
</div>

<div class="section">
  <div class="section-title"><span class="icon">🔬</span> 1. 측정 오차</div>
  <p>어떤 기구로 같은 대상을 반복 측정할 때 나타나는 오차는 보통 정규 분포를 따릅니다.</p>
  <div class="math-block">
    $$\epsilon = \epsilon^{(1)} + \epsilon^{(2)} + \cdots + \epsilon^{(N)}$$
  </div>
  <p>각 측정마다 환경, 기기 특성, 조작 등 다양한 미세 요인이 독립적으로 오차에 영향을 줍니다. 이 합이 정규 분포에 수렴합니다.</p>
  <div class="highlight-box amber">
    만약 측정값이 정규 분포를 크게 벗어난다면 기기 고장이나 계통 오차를 의심해야 합니다.
  </div>
</div>

<div class="section">
  <div class="section-title"><span class="icon">🏭</span> 2. 제품 크기</div>
  <p>공장에서 생산된 500ml 생수병의 실제 용량을 여러 개 측정하면, 500ml 주변의 정규 분포를 따릅니다.</p>
  <p>제품이 완성되기까지 여러 공정마다 미세한 오차가 누적되기 때문입니다.</p>
</div>

<div class="section">
  <div class="section-title"><span class="icon">🧑</span> 3. 사람의 키</div>
  <p>같은 나이·성별의 사람들 키는 정규 분포에 근사합니다. 유전, 식습관, 운동, 환경 등 다양한 독립적 요인의 복합 작용으로 설명됩니다.</p>
  <div class="stat-grid">
    <div class="stat-card"><div class="label">한국 성인 남성 평균 키</div><div class="value">≈ 174cm</div></div>
    <div class="stat-card"><div class="label">표준편차</div><div class="value">≈ 6cm</div></div>
    <div class="stat-card"><div class="label">160~188cm 확률</div><div class="value">≈ 95.4%</div></div>
  </div>
</div>

<div class="section">
  <div class="section-title"><span class="icon">🔗</span> 다음 단계로</div>
  <p>정규 분포를 실제 데이터에 적합(fitting)시키려면 매개변수 $\mu$, $\sigma$를 추정해야 합니다. 이것이 다음 챕터의 주제입니다.</p>
  <div class="highlight-box purple">
    <a class="ch-link" href="#" onclick="alert('Chapter 2 최대 가능도 추정은 준비중입니다.')">→ Chapter 2 최대 가능도 추정(MLE) 학습하기</a>
  </div>
</div>
`;

/* ============================================================
   CHART INIT REGISTRATIONS
   ============================================================ */
CHART_INITS['1.1.1'] = init_dice;
CHART_INITS['1.1.2'] = init_discrete_continuous;
CHART_INITS['1.2.2'] = init_normal_basic;
CHART_INITS['1.2.3'] = init_param_explorer;
CHART_INITS['1.3.2'] = init_clt_sim;
CHART_INITS['1.4.2'] = init_sum_sim;

/* ============================================================
   CHART IMPLEMENTATIONS
   ============================================================ */

/* --- 1.1.1 Dice --- */
let diceFreq = [0,0,0,0,0,0];
let diceTotal = 0;
let diceChartInst = null;

function init_dice() {
  const ctx = document.getElementById('dice-chart');
  if (!ctx) return;
  diceFreq = [0,0,0,0,0,0]; diceTotal = 0;
  diceChartInst = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['1','2','3','4','5','6'],
      datasets: [
        { label: '관측 빈도', data: [0,0,0,0,0,0], backgroundColor: 'rgba(59,130,246,0.6)', borderColor: '#3b82f6', borderWidth: 1 },
        { label: '이론값 (1/6)', data: [1/6,1/6,1/6,1/6,1/6,1/6], type:'line', borderColor:'#ef4444', borderDash:[5,5], pointRadius:0, borderWidth:2, fill:false }
      ]
    },
    options: {
      responsive:true, plugins:{ legend:{position:'top'} },
      scales:{ y:{ title:{display:true,text:'상대 빈도'}, min:0, max:0.5 } }
    }
  });
  activeChartInstances['dice'] = diceChartInst;
}

window.rollDice = function(n) {
  for (let i = 0; i < n; i++) {
    const r = Math.floor(Math.random() * 6);
    diceFreq[r]++;
    diceTotal++;
  }
  document.getElementById('dice-count').textContent = diceTotal;
  const lastRoll = Math.floor(Math.random() * 6) + 1;
  const face = document.getElementById('die-face');
  if (face) { face.classList.remove('rolling'); void face.offsetWidth; face.classList.add('rolling'); face.textContent = lastRoll; }
  if (diceChartInst) {
    const rel = diceFreq.map(f => diceTotal > 0 ? f/diceTotal : 0);
    diceChartInst.data.datasets[0].data = rel;
    diceChartInst.update();
  }
};

window.resetDice = function() {
  diceFreq = [0,0,0,0,0,0]; diceTotal = 0;
  document.getElementById('dice-count').textContent = 0;
  if (diceChartInst) { diceChartInst.data.datasets[0].data = [0,0,0,0,0,0]; diceChartInst.update(); }
};

/* --- 1.1.2 Discrete vs Continuous --- */
let discreteChart = null;
let continuousChart = null;

function init_discrete_continuous() {
  // ---- Discrete: die (이산 확률 분포) ----
  const ctx1 = document.getElementById('discrete-chart');
  if (!ctx1) return;
  discreteChart = new Chart(ctx1, {
    type: 'bar',
    data: {
      labels: ['1', '2', '3', '4', '5', '6'],
      datasets: [{
        label: 'p(x)',
        data: [1/6, 1/6, 1/6, 1/6, 1/6, 1/6],
        backgroundColor: [
          'rgba(59,130,246,0.75)', 'rgba(79,112,234,0.75)', 'rgba(99,102,241,0.75)',
          'rgba(99,102,241,0.75)', 'rgba(79,112,234,0.75)', 'rgba(59,130,246,0.75)'
        ],
        borderColor: '#3b82f6',
        borderWidth: 1.5,
        borderRadius: 4,
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => `p(${ctx.label}) = 1/6 ≈ ${(1/6).toFixed(4)}` } },
        annotation: {}
      },
      scales: {
        y: {
          title: { display: true, text: '확률 p(x)' },
          min: 0, max: 0.28,
          ticks: { callback: v => v.toFixed(2) }
        },
        x: { title: { display: true, text: '주사위 눈 x' } }
      }
    }
  });
  activeChartInstances['discrete'] = discreteChart;

  // ---- Continuous: height N(174, 6²) (연속 확률 분포) ----
  const ctx2 = document.getElementById('continuous-chart');
  if (!ctx2) return;
  const MU = 174, SIGMA = 6;
  const xs = linspace(150, 200, 300);
  const ys = xs.map(x => normalPDF(x, MU, SIGMA));

  continuousChart = new Chart(ctx2, {
    type: 'line',
    data: {
      labels: xs.map(x => x.toFixed(1)),
      datasets: [
        {
          label: '확률 밀도 p(x)',
          data: ys,
          borderColor: '#7c3aed',
          borderWidth: 2.5,
          fill: false,
          pointRadius: 0,
          tension: 0.4
        },
        {
          label: '선택 구간',
          data: new Array(300).fill(null),
          backgroundColor: 'rgba(124,58,237,0.22)',
          borderWidth: 0,
          fill: true,
          pointRadius: 0,
          tension: 0.4,
          spanGaps: false
        }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => `p(x) = ${ctx.raw !== null ? Number(ctx.raw).toFixed(5) : '—'}` } }
      },
      scales: {
        y: {
          title: { display: true, text: '확률 밀도 p(x)' },
          min: 0,
          ticks: { callback: v => v.toFixed(3) }
        },
        x: {
          title: { display: true, text: '키 x (cm)' },
          ticks: { maxTicksLimit: 8, callback: function(v) { return Math.round(parseFloat(this.getLabelForValue(v))); } }
        }
      }
    }
  });
  activeChartInstances['continuous'] = continuousChart;
  updateHeightRange();
}

window.updateHeightRange = function() {
  if (!continuousChart) return;
  const a = parseFloat(document.getElementById('height-a-slider').value);
  const b = parseFloat(document.getElementById('height-b-slider').value);
  const MU = 174, SIGMA = 6;
  const xs = linspace(150, 200, 300);
  const ys = xs.map(x => normalPDF(x, MU, SIGMA));
  const lo = Math.min(a, b), hi = Math.max(a, b);
  continuousChart.data.datasets[1].data = xs.map((x, i) => (x >= lo && x <= hi) ? ys[i] : null);
  continuousChart.update('none');
  document.getElementById('height-a-val').textContent = a;
  document.getElementById('height-b-val').textContent = b;
  const prob = numericalIntegral(lo, hi, x => normalPDF(x, MU, SIGMA), 500);
  document.getElementById('height-prob').textContent = (prob * 100).toFixed(1) + '%';
};

/* --- 1.2.2 Normal basic with area --- */
let normalBasicChart = null;

function init_normal_basic() {
  const ctx = document.getElementById('normal-basic-chart');
  if (!ctx) return;
  const xs = linspace(-4, 4, 200);
  const ys = xs.map(x => normalPDF(x, 0, 1));

  normalBasicChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: xs.map(x => x.toFixed(2)),
      datasets: [
        { label: 'N(0,1)', data: ys, borderColor: '#3b82f6', borderWidth: 2.5, fill: false, pointRadius: 0, tension: 0.4 },
        { label: '선택 구간', data: new Array(200).fill(null), backgroundColor: 'rgba(59,130,246,0.25)', borderWidth: 0, fill: true, pointRadius: 0 }
      ]
    },
    options: { responsive:true, plugins:{ legend:{position:'top'} }, scales:{ y:{ min:0, max:0.45, title:{display:true,text:'p(x)'} }, x:{ title:{display:true,text:'x'}, ticks:{ maxTicksLimit:9, callback:(v,i) => parseFloat(xs[i]).toFixed(1) } } } }
  });
  activeChartInstances['normalBasic'] = normalBasicChart;
  updateNormalBasic();
}

window.updateNormalBasic = function() {
  if (!normalBasicChart) return;
  const a = parseFloat(document.getElementById('a-slider').value);
  const b = parseFloat(document.getElementById('b-slider').value);
  const xs = linspace(-4, 4, 200);
  const areaData = xs.map(x => (x >= Math.min(a,b) && x <= Math.max(a,b)) ? normalPDF(x,0,1) : null);
  normalBasicChart.data.datasets[1].data = areaData;
  normalBasicChart.update();
  const prob = numericalIntegral(Math.min(a,b), Math.max(a,b), x => normalPDF(x,0,1), 500);
  document.getElementById('prob-value').textContent = (prob * 100).toFixed(1) + '%';
};

/* --- 1.2.3 Param explorer --- */
let paramChart = null;

function init_param_explorer() {
  const ctx = document.getElementById('param-chart');
  if (!ctx) return;
  const xs = linspace(-8, 8, 300);

  paramChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: xs.map(x => x.toFixed(2)),
      datasets: [
        { label: '현재 분포', data: xs.map(x => normalPDF(x,0,1)), borderColor: '#3b82f6', borderWidth: 2.5, fill: false, pointRadius: 0, tension: 0.4 },
        { label: 'N(0,1) 기준', data: xs.map(x => normalPDF(x,0,1)), borderColor: '#94a3b8', borderWidth: 1.5, borderDash: [5,4], fill: false, pointRadius: 0, tension: 0.4 }
      ]
    },
    options: {
      responsive: true,
      plugins: { legend: { position: 'top' } },
      scales: {
        y: { min: 0, max: 2.1, title: { display: true, text: 'p(x)' } },
        x: { title: { display: true, text: 'x' }, ticks: { maxTicksLimit: 9, callback: (v,i) => { const xs2=linspace(-8,8,300); return xs2[i]!==undefined?parseFloat(xs2[i]).toFixed(1):''; } } }
      }
    }
  });
  activeChartInstances['param'] = paramChart;
}

window.updateParamChart = function() {
  if (!paramChart) return;
  const mu = parseFloat(document.getElementById('mu-slider').value);
  const sigma = parseFloat(document.getElementById('sigma-slider').value);
  const showRef = document.getElementById('show-ref').checked;
  document.getElementById('mu-badge').textContent = mu.toFixed(1);
  document.getElementById('sigma-badge').textContent = sigma.toFixed(1);
  document.getElementById('mu-val').textContent = mu.toFixed(1);
  document.getElementById('sigma-val').textContent = sigma.toFixed(1);
  const xs = linspace(-8, 8, 300);
  paramChart.data.datasets[0].data = xs.map(x => normalPDF(x, mu, sigma));
  paramChart.data.datasets[1].hidden = !showRef;
  const peak = normalPDF(mu, mu, sigma);
  document.getElementById('peak-val').textContent = peak.toFixed(3);
  document.getElementById('var-val').textContent = (sigma*sigma).toFixed(2);
  paramChart.options.scales.y.max = Math.max(2.1, peak * 1.2);
  paramChart.update();
};

/* --- 1.3.2 CLT Simulation --- */
let cltChart = null;

function init_clt_sim() {
  const ctx = document.getElementById('clt-chart');
  if (!ctx) return;
  cltChart = new Chart(ctx, {
    type: 'bar',
    data: { labels: [], datasets: [
      { label: '표본 평균 히스토그램', data: [], backgroundColor: 'rgba(59,130,246,0.5)', borderColor:'#3b82f6', borderWidth:1 },
      { label: '이론 정규 분포', data: [], type: 'line', borderColor: '#ef4444', borderWidth: 2, pointRadius: 0, fill: false }
    ]},
    options: { responsive: true, plugins: { legend: { position: 'top' } },
      scales: { y: { title: { display: true, text: '빈도 (밀도)' } }, x: { title: { display: true, text: '표본 평균' }, ticks: { maxTicksLimit: 10 } } }
    }
  });
  activeChartInstances['clt'] = cltChart;
}

window.runCLT = function() {
  if (!cltChart) return;
  const N = parseInt(document.getElementById('clt-n-slider').value);
  const trials = parseInt(document.getElementById('clt-trials').value);
  const mu_theory = 0.5, sigma_theory = Math.sqrt(1/12/N);

  const means = [];
  for (let t = 0; t < trials; t++) {
    let s = 0;
    for (let i = 0; i < N; i++) s += Math.random();
    means.push(s / N);
  }

  const minV = Math.min(...means), maxV = Math.max(...means);
  const bins = 40;
  const binSize = (maxV - minV) / bins || 0.01;
  const freq = new Array(bins).fill(0);
  means.forEach(v => {
    const idx = Math.min(Math.floor((v - minV) / binSize), bins - 1);
    freq[idx]++;
  });
  const density = freq.map(f => f / (trials * binSize));
  const labels = Array.from({length: bins}, (_, i) => (minV + (i + 0.5) * binSize).toFixed(3));
  const normY = labels.map(x => normalPDF(parseFloat(x), mu_theory, sigma_theory));

  cltChart.data.labels = labels;
  cltChart.data.datasets[0].data = density;
  cltChart.data.datasets[1].data = normY;
  cltChart.update();

  const empMu = means.reduce((a,b)=>a+b,0)/means.length;
  const empVar = means.reduce((a,b)=>a+(b-empMu)**2,0)/means.length;
  document.getElementById('clt-theory-mu').textContent = mu_theory.toFixed(3);
  document.getElementById('clt-emp-mu').textContent = empMu.toFixed(3);
  document.getElementById('clt-theory-sigma').textContent = sigma_theory.toFixed(3);
  document.getElementById('clt-emp-sigma').textContent = Math.sqrt(empVar).toFixed(3);
};

/* --- 1.4.2 Sum Simulation --- */
let sumChart = null;

function init_sum_sim() {
  const ctx = document.getElementById('sum-chart');
  if (!ctx) return;
  sumChart = new Chart(ctx, {
    type: 'bar',
    data: { labels: [], datasets: [
      { label: '표본 합 히스토그램', data: [], backgroundColor: 'rgba(34,197,94,0.5)', borderColor:'#22c55e', borderWidth:1 },
      { label: '이론 정규 분포', data: [], type: 'line', borderColor: '#ef4444', borderWidth: 2, pointRadius: 0, fill: false }
    ]},
    options: { responsive: true, plugins: { legend: { position: 'top' } },
      scales: { y: { title: { display: true, text: '빈도 (밀도)' } }, x: { title: { display: true, text: '표본 합' }, ticks: { maxTicksLimit: 10 } } }
    }
  });
  activeChartInstances['sum'] = sumChart;
}

window.runSumSim = function() {
  if (!sumChart) return;
  const N = parseInt(document.getElementById('sum-n-slider').value);
  const trials = 5000;
  const mu_s = N * 0.5, sigma_s = Math.sqrt(N / 12);

  const sums = [];
  for (let t = 0; t < trials; t++) {
    let s = 0;
    for (let i = 0; i < N; i++) s += Math.random();
    sums.push(s);
  }

  const minV = Math.min(...sums), maxV = Math.max(...sums);
  const bins = 40;
  const binSize = (maxV - minV) / bins || 0.01;
  const freq = new Array(bins).fill(0);
  sums.forEach(v => {
    const idx = Math.min(Math.floor((v - minV) / binSize), bins - 1);
    freq[idx]++;
  });
  const density = freq.map(f => f / (trials * binSize));
  const labels = Array.from({length: bins}, (_, i) => (minV + (i + 0.5) * binSize).toFixed(2));
  const normY = labels.map(x => normalPDF(parseFloat(x), mu_s, sigma_s));

  sumChart.data.labels = labels;
  sumChart.data.datasets[0].data = density;
  sumChart.data.datasets[1].data = normY;
  sumChart.update();

  const empMu = sums.reduce((a,b)=>a+b,0)/sums.length;
  const empVar = sums.reduce((a,b)=>a+(b-empMu)**2,0)/sums.length;
  document.getElementById('sum-theory-mu').textContent = mu_s.toFixed(3);
  document.getElementById('sum-emp-mu').textContent = empMu.toFixed(3);
  document.getElementById('sum-theory-sigma').textContent = sigma_s.toFixed(3);
  document.getElementById('sum-emp-sigma').textContent = Math.sqrt(empVar).toFixed(3);
  document.getElementById('sum-n-val').textContent = N;
};
