/* ============================================================
   Chapter 2 — 최대 가능도 추정 (MLE)
   Content + Chart implementations
   ============================================================ */

/* ─────── 챕터 2 수학 유틸리티 ─────── */
(function () {
  function bm(mu, si) {
    let u1; do { u1 = Math.random(); } while (u1 === 0);
    return mu + si * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * Math.random());
  }
  function erf(x) {
    const s = x >= 0 ? 1 : -1, a = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * a);
    const p = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    return s * (1 - p * Math.exp(-a * a));
  }
  window.ch2 = {
    pdf: (x, mu, si) => Math.exp(-0.5 * ((x - mu) / si) ** 2) / (si * Math.sqrt(2 * Math.PI)),
    cdf: (x, mu, si) => 0.5 * (1 + erf((x - mu) / (si * Math.sqrt(2)))),
    rnd: bm,
    gen: (n, mu, si) => Array.from({ length: n }, () => bm(mu, si)),
    mu: a => a.reduce((s, x) => s + x, 0) / a.length,
    si: a => { const m = a.reduce((s, x) => s + x, 0) / a.length; return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / a.length); },
    logL: (d, mu, si) => d.reduce((s, x) => s + Math.log(Math.max(Math.exp(-0.5 * ((x - mu) / si) ** 2) / (si * Math.sqrt(2 * Math.PI)), 1e-300)), 0),
    ls: (a, b, n) => Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1)),
    hd: null
  };
  window.ch2.hd = window.ch2.gen(300, 170.7, 5.5);
})();


/* ─────────────────────── 2.1.1 ─────────────────────── */
CONTENT['2.1.1'] = () => String.raw`
<div class="page-title">생성 모델의 목표</div>
<div class="page-subtitle">2.1 생성 모델 개요</div>

<div class="section">
  <div class="section-title"><span class="icon">🎯</span> 생성 모델이란?</div>
  <p>생성 모델(generative model)의 핵심 목표는 주어진 데이터 $\{x^{(1)}, \ldots, x^{(N)}\}$으로부터 <strong>확률 분포 $p(x)$를 모델링</strong>하고, 그 분포에서 원본과 유사한 새 데이터를 생성할 수 있게 하는 것입니다.</p>
  <div class="highlight-box">
    <strong>예시:</strong> 25,000명의 키 데이터를 관측 → 키 분포 $p(x)$를 정규 분포로 모델링 → 모델에서 새로운 가상의 키 데이터를 얼마든지 생성
  </div>
</div>

<div class="section">
  <div class="section-title"><span class="icon">🔄</span> 생성 모델 구축 2단계</div>
  <p>생성 모델을 만들기 위해서는 두 가지 핵심 단계를 거칩니다:</p>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:16px 0;">
    <div class="highlight-box">
      <strong>① 모델링 (Modeling)</strong><br><br>
      모집단 분포를 <em>매개변수로 조정 가능한 확률 분포</em>로 가정합니다.<br><br>
      예: 키 데이터는 정규 분포를 따른다고 가정<br>
      → $p(x;\mu,\sigma) = \mathcal{N}(x;\mu,\sigma)$
    </div>
    <div class="highlight-box green">
      <strong>② 매개변수 추정 (Estimation)</strong><br><br>
      샘플 데이터에 가장 잘 부합하도록 매개변수를 추정합니다.<br><br>
      <strong>이 챕터의 핵심:</strong><br>
      최대 가능도 추정(MLE, Maximum Likelihood Estimation)으로 $\hat{\mu}$, $\hat{\sigma}$ 추정
    </div>
  </div>
  <div class="steps">
    <div class="step active"><div class="step-num">1</div>데이터 수집</div>
    <div class="step active"><div class="step-num">2</div>모델링 선택<br><small>(정규분포 가정)</small></div>
    <div class="step active"><div class="step-num">3</div>매개변수 추정<br><small>(MLE, Maximum<br>Likelihood Estimation)</small></div>
    <div class="step active"><div class="step-num">4</div>새 데이터 생성</div>
  </div>
</div>

<div class="section">
  <div class="section-title"><span class="icon">📐</span> 매개변수로 결정되는 정규 분포</div>
  <p>정규 분포의 모양은 두 매개변수로 완전히 결정됩니다:</p>
  <div class="math-block">
    $$\mathcal{N}(x;\,\mu,\,\sigma) = \frac{1}{\sqrt{2\pi}\,\sigma}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$
  </div>
  <ul class="content-list">
    <li>$\mu$ (평균, mean): 분포의 <strong>중심 위치</strong>를 결정</li>
    <li>$\sigma$ (표준편차, std dev): 분포의 <strong>퍼짐 정도</strong>를 결정</li>
  </ul>
  <div class="highlight-box amber">
    <strong>핵심 과제:</strong> 샘플 데이터 $\mathcal{D} = \{x^{(1)}, \ldots, x^{(N)}\}$가 주어졌을 때,
    이를 가장 잘 설명하는 $\hat{\mu}$와 $\hat{\sigma}$를 어떻게 찾을까?<br>
    → <strong>최대 가능도 추정(MLE, Maximum Likelihood Estimation)</strong>을 사용합니다.
  </div>
</div>

<div class="section">
  <div class="section-title"><span class="icon">🌍</span> 정규 분포로 모델링하는 이유</div>
  <ul class="content-list">
    <li><strong>중심극한정리:</strong> 많은 독립적인 원인이 합쳐진 현상은 정규 분포에 근사
      <a href="#" class="ch-link" onclick="(function(){var s=FLAT_SECTIONS.find(function(x){return x.id==='1.3.1';});if(s)loadSection(s.id,s.ch,s.sub,s.sec);})();return false;">→ Ch.1.3 중심극한정리</a>
    </li>
    <li><strong>계산의 편리함:</strong> 수학적으로 다루기 쉬운 성질을 많이 가짐</li>
    <li><strong>자연 현상과의 부합:</strong> 키, 몸무게, 혈압 등 많은 실제 데이터가 정규 분포에 근사</li>
  </ul>
</div>
`;


/* ─────────────────────── 2.1.2 ─────────────────────── */
CONTENT['2.1.2'] = () => String.raw`
<div class="page-title">모집단과 샘플</div>
<div class="page-subtitle">2.1 생성 모델 개요</div>

<div class="section">
  <div class="section-title"><span class="icon">🌐</span> 모집단과 샘플의 관계</div>
  <p>통계학과 머신러닝에서 핵심적인 구분이 있습니다:</p>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:16px 0;">
    <div class="highlight-box">
      <strong>모집단 (Population)</strong><br><br>
      관심 대상의 <em>전체 집합</em>. 규모가 방대하여 현실적으로 전수 관측 불가.<br><br>
      예: 전 세계 18세 청소년의 키<br>
      → 실제 분포 $p^*(x)$는 <strong>알 수 없음</strong>
    </div>
    <div class="highlight-box green">
      <strong>샘플 (Sample)</strong><br><br>
      모집단에서 무작위로 선택한 <em>제한된 수의 관측값</em>.<br><br>
      예: 25,000명의 키 측정값<br>
      → 이 샘플로 $p^*(x)$를 <strong>추정</strong>
    </div>
  </div>
</div>

<div class="section">
  <div class="section-title"><span class="icon">🔍</span> 모집단 분포 추정 절차</div>
  <ul class="content-list">
    <li><strong>모집단 분포 $p^*(x)$:</strong> 샘플을 뒷받침하는 실제 확률 분포. 현실적으로 알 수 없음.</li>
    <li><strong>모델링:</strong> $p^*(x)$를 매개변수로 조정 가능한 확률 분포 $p(x;\theta)$로 근사.</li>
    <li><strong>매개변수 추정:</strong> 샘플 데이터에 가장 잘 부합하는 $\hat{\theta}$를 찾음 (MLE, Maximum Likelihood Estimation).</li>
  </ul>
  <div class="math-block">$$p^*(x) \approx p(x;\,\hat{\theta})$$</div>
  <p class="formula-label">모집단 분포 ≈ 추정된 분포 (매개변수 $\hat{\theta}$로 결정)</p>
</div>

<div class="section">
  <div class="section-title"><span class="icon">🎮</span> 샘플링 시뮬레이션</div>
  <p>아래 데모에서 모집단 분포 $\mathcal{N}(170.7,\;5.5^2)$로부터 샘플을 추출해 보세요.<br>
  샘플이 많아질수록 추정된 평균이 모집단 평균 <strong>170.7 cm</strong>에 수렴합니다.</p>
  <div class="interactive-panel">
    <div class="panel-header">🎲 모집단 샘플링 시뮬레이터</div>
    <div class="panel-body" style="flex-direction:column;gap:12px;">
      <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;">
        <button class="btn btn-primary" onclick="ch2_212_add(1)">+1개</button>
        <button class="btn btn-secondary" onclick="ch2_212_add(10)">+10개</button>
        <button class="btn btn-secondary" onclick="ch2_212_add(100)">+100개</button>
        <button class="btn btn-secondary" onclick="ch2_212_reset()">초기화</button>
      </div>
      <div class="stat-grid" style="grid-template-columns:repeat(3,1fr);">
        <div class="stat-card"><div class="label">샘플 수</div><div class="value" id="c212-n">0</div></div>
        <div class="stat-card"><div class="label">샘플 평균 <span style="font-size:0.7rem">(추정값)</span></div><div class="value" id="c212-mean" style="color:#3b82f6;">—</div></div>
        <div class="stat-card"><div class="label">모집단 평균 <span style="font-size:0.7rem">(진짜값)</span></div><div class="value" style="color:#22c55e;">170.7</div></div>
      </div>
      <canvas id="c212-chart" height="220"></canvas>
    </div>
  </div>
</div>
`;

CHART_INITS['2.1.2'] = function () {
  const MU = 170.7, SI = 5.5;
  let samples = [];
  const xs = ch2.ls(150, 195, 120);
  const ctx = document.getElementById('c212-chart').getContext('2d');
  const chart = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: '모집단 PDF N(170.7, 5.5²)',
          type: 'line',
          data: xs.map(x => ({ x, y: ch2.pdf(x, MU, SI) })),
          borderColor: '#22c55e', borderWidth: 2,
          backgroundColor: 'rgba(34,197,94,0.08)', fill: true,
          pointRadius: 0, tension: 0.4, order: 2
        },
        {
          label: '샘플',
          data: [],
          backgroundColor: 'rgba(59,130,246,0.65)',
          pointRadius: 5, order: 1
        }
      ]
    },
    options: {
      responsive: true, animation: { duration: 150 },
      plugins: { legend: { labels: { font: { size: 11 } } }, tooltip: { enabled: false } },
      scales: {
        x: { type: 'linear', min: 150, max: 195, title: { display: true, text: '키 (cm)', font: { size: 11 } } },
        y: { beginAtZero: true, title: { display: true, text: '확률 밀도', font: { size: 11 } } }
      }
    }
  });
  activeChartInstances['c212'] = chart;

  window.ch2_212_add = (n) => {
    for (let i = 0; i < n; i++) samples.push(ch2.rnd(MU, SI));
    chart.data.datasets[1].data = samples.map(x => ({ x, y: 0.0008 + Math.random() * 0.001 }));
    chart.update();
    document.getElementById('c212-n').textContent = samples.length;
    document.getElementById('c212-mean').textContent = ch2.mu(samples).toFixed(2) + ' cm';
  };
  window.ch2_212_reset = () => {
    samples = [];
    chart.data.datasets[1].data = [];
    chart.update();
    document.getElementById('c212-n').textContent = 0;
    document.getElementById('c212-mean').textContent = '—';
  };
};


/* ─────────────────────── 2.2.1 ─────────────────────── */
CONTENT['2.2.1'] = () => String.raw`
<div class="page-title">키 데이터셋 불러오기</div>
<div class="page-subtitle">2.2 실제 데이터로 생성 모델 구현</div>

<div class="section">
  <div class="section-title"><span class="icon">📂</span> 키 데이터셋 소개</div>
  <p>이 챕터에서는 실제 데이터를 기반으로 생성 모델을 구현합니다. 사용하는 데이터는:</p>
  <ul class="content-list">
    <li><strong>1993년 홍콩 18세 청소년</strong>의 키 데이터</li>
    <li>총 <strong>25,000개</strong>의 관측값 (cm 단위)</li>
    <li>히스토그램으로 그리면 <strong>정규 분포(좌우 대칭 종 모양)</strong>와 유사</li>
  </ul>
  <div class="highlight-box">
    이 챕터에서는 웹 환경을 위해 동일 분포 $\mathcal{N}(170.7,\;5.5^2)$에서 300개를 시뮬레이션하여 사용합니다.
  </div>
</div>

<div class="section">
  <div class="section-title"><span class="icon">💻</span> 데이터 로드 및 시각화 코드</div>
  <pre class="code-pre"><code class="language-python">import numpy as np
import matplotlib.pyplot as plt

path = 'height.txt'
xs = np.loadtxt(path)
print(xs.shape)  # (25000,)

plt.hist(xs, bins='auto', density=True)
plt.xlabel('Height(cm)')
plt.ylabel('Probability Density')
plt.show()</code></pre>
</div>

<div class="section">
  <div class="section-title"><span class="icon">📊</span> 키 데이터 히스토그램 (시뮬레이션)</div>
  <div class="interactive-panel">
    <div class="panel-header">📊 키 데이터 히스토그램 — N(170.7, 5.5²) 시뮬레이션</div>
    <div class="panel-body" style="flex-direction:column;gap:12px;">
      <div class="stat-grid" style="grid-template-columns:repeat(3,1fr);">
        <div class="stat-card"><div class="label">샘플 수</div><div class="value" id="c221-n">—</div></div>
        <div class="stat-card"><div class="label">표본 평균 μ̂</div><div class="value" id="c221-mu" style="color:#3b82f6;">—</div></div>
        <div class="stat-card"><div class="label">표본 표준편차 σ̂</div><div class="value" id="c221-si" style="color:#7c3aed;">—</div></div>
      </div>
      <canvas id="c221-chart" height="250"></canvas>
      <button class="btn btn-secondary" onclick="ch2_221_resample()">🔄 새로운 샘플로 다시 그리기</button>
    </div>
  </div>
</div>
`;

CHART_INITS['2.2.1'] = function () {
  const NUM_BINS = 20, MIN = 150, MAX = 195;
  const bw = (MAX - MIN) / NUM_BINS;

  function makeBins(d) {
    const counts = new Array(NUM_BINS).fill(0);
    d.forEach(x => { const b = Math.min(Math.floor((x - MIN) / bw), NUM_BINS - 1); if (b >= 0) counts[b]++; });
    return { labels: Array.from({ length: NUM_BINS }, (_, i) => (MIN + (i + 0.5) * bw).toFixed(1)), densities: counts.map(c => c / (d.length * bw)) };
  }

  function updateStats(d) {
    document.getElementById('c221-n').textContent = d.length;
    document.getElementById('c221-mu').textContent = ch2.mu(d).toFixed(2) + ' cm';
    document.getElementById('c221-si').textContent = ch2.si(d).toFixed(2) + ' cm';
  }

  let data = [...ch2.hd];
  const { labels, densities } = makeBins(data);
  const ctx = document.getElementById('c221-chart').getContext('2d');
  const chart = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets: [{ label: '확률 밀도', data: densities, backgroundColor: 'rgba(59,130,246,0.5)', borderColor: '#2563eb', borderWidth: 1, barPercentage: 1.0, categoryPercentage: 1.0 }] },
    options: {
      responsive: true, animation: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: '키 (cm)', font: { size: 11 } }, ticks: { maxTicksLimit: 10, font: { size: 10 } } },
        y: { beginAtZero: true, title: { display: true, text: '확률 밀도', font: { size: 11 } } }
      }
    }
  });
  activeChartInstances['c221'] = chart;
  updateStats(data);

  window.ch2_221_resample = () => {
    data = ch2.gen(300, 170.7, 5.5);
    ch2.hd = data;
    const { labels: lb, densities: dn } = makeBins(data);
    chart.data.labels = lb;
    chart.data.datasets[0].data = dn;
    chart.update();
    updateStats(data);
  };
};


/* ─────────────────────── 2.2.2 ─────────────────────── */
CONTENT['2.2.2'] = () => String.raw`
<div class="page-title">정규 분포를 따르는 생성 모델</div>
<div class="page-subtitle">2.2 실제 데이터로 생성 모델 구현</div>

<div class="section">
  <div class="section-title"><span class="icon">🧠</span> 정규 분포로 모델링</div>
  <p>키 데이터를 정규 분포로 모델링하는 방법은 간단합니다:</p>
  <ul class="content-list">
    <li><strong>모델링:</strong> 키 데이터가 정규 분포를 따른다고 가정 → $p(x;\mu,\sigma) = \mathcal{N}(x;\mu,\sigma)$</li>
    <li><strong>매개변수 추정:</strong> 샘플의 <em>평균</em>과 <em>표준편차</em>를 계산하여 $\hat{\mu}$, $\hat{\sigma}$ 추정</li>
  </ul>
  <div class="math-block">
    $$\hat{\mu} = \frac{1}{N}\sum_{n=1}^N x^{(n)}, \qquad \hat{\sigma} = \sqrt{\frac{1}{N}\sum_{n=1}^N \bigl(x^{(n)} - \hat{\mu}\bigr)^2}$$
  </div>
  <p>이렇게 추정한 $\hat{\mu}$, $\hat{\sigma}$로 정규 분포 곡선을 그리면 실제 데이터 분포와 매우 잘 맞습니다.</p>
</div>

<div class="section">
  <div class="section-title"><span class="icon">💻</span> 피팅 코드</div>
  <pre class="code-pre"><code class="language-python">import numpy as np
import matplotlib.pyplot as plt

xs = np.loadtxt('height.txt')
mu_hat = np.mean(xs)      # 표본 평균 → μ 추정
sigma_hat = np.std(xs)    # 표본 표준편차 → σ 추정

def normal_pdf(x, mu, sigma):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(x - mu)**2 / (2 * sigma**2))

x_range = np.linspace(150, 195, 500)
plt.hist(xs, bins='auto', density=True, label='데이터')
plt.plot(x_range, normal_pdf(x_range, mu_hat, sigma_hat), 'r-', lw=2, label='정규분포 피팅')
plt.legend()
plt.show()</code></pre>
</div>

<div class="section">
  <div class="section-title"><span class="icon">🎛️</span> 인터랙티브 피팅 데모</div>
  <p>슬라이더로 $\mu$와 $\sigma$를 조정하면서 히스토그램에 정규 분포를 맞춰보세요.<br>
  <strong>MLE(Maximum Likelihood Estimation) 최적값 적용</strong> 버튼을 누르면 최대 가능도 추정값으로 자동 설정됩니다.</p>
  <div class="interactive-panel">
    <div class="panel-header">🎯 정규 분포 피팅 — 히스토그램 + 추정 PDF</div>
    <div class="panel-body" style="flex-direction:column;gap:14px;">
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
        <div class="ctrl-group" style="margin:0;">
          <div class="ctrl-label">평균 μ = <span id="c222-mu-lbl">170.7</span> cm</div>
          <input type="range" id="c222-mu-sl" min="160" max="182" step="0.1" value="170.7" style="width:100%;" oninput="ch2_222_update()">
        </div>
        <div class="ctrl-group" style="margin:0;">
          <div class="ctrl-label">표준편차 σ = <span id="c222-si-lbl">5.5</span> cm</div>
          <input type="range" id="c222-si-sl" min="1" max="12" step="0.1" value="5.5" style="width:100%;" oninput="ch2_222_update()">
        </div>
      </div>
      <div class="stat-grid" style="grid-template-columns:repeat(3,1fr);">
        <div class="stat-card"><div class="label">현재 μ</div><div class="value" id="c222-mu-v" style="color:#3b82f6;">—</div></div>
        <div class="stat-card"><div class="label">현재 σ</div><div class="value" id="c222-si-v" style="color:#7c3aed;">—</div></div>
        <div class="stat-card"><div class="label">log L(μ, σ)</div><div class="value" id="c222-logl" style="font-size:0.85rem;color:#64748b;">—</div></div>
      </div>
      <canvas id="c222-chart" height="260"></canvas>
      <button class="btn btn-green" onclick="ch2_222_setMLE()">✨ MLE 최적값 적용 (μ̂, σ̂)</button>
    </div>
  </div>
</div>
`;

CHART_INITS['2.2.2'] = function () {
  const data = ch2.hd;
  const muMLE = ch2.mu(data), siMLE = ch2.si(data);
  const NUM_BINS = 20, MIN = 150, MAX = 195;
  const bw = (MAX - MIN) / NUM_BINS;
  const binCounts = new Array(NUM_BINS).fill(0);
  data.forEach(x => { const b = Math.min(Math.floor((x - MIN) / bw), NUM_BINS - 1); if (b >= 0) binCounts[b]++; });
  const densities = binCounts.map(c => c / (data.length * bw));
  const binLabels = Array.from({ length: NUM_BINS }, (_, i) => (MIN + (i + 0.5) * bw).toFixed(1));
  const binCenters = Array.from({ length: NUM_BINS }, (_, i) => MIN + (i + 0.5) * bw);
  let mu = muMLE, si = siMLE;

  const ctx = document.getElementById('c222-chart').getContext('2d');
  const chart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: binLabels,
      datasets: [
        { label: '데이터 히스토그램', data: densities, backgroundColor: 'rgba(99,102,241,0.4)', borderColor: '#4f46e5', borderWidth: 1, barPercentage: 1.0, categoryPercentage: 1.0, order: 2 },
        { label: '정규분포 PDF', type: 'line', data: binCenters.map(x => ch2.pdf(x, mu, si)), borderColor: '#ef4444', borderWidth: 2.5, backgroundColor: 'rgba(0,0,0,0)', pointRadius: 0, tension: 0.4, order: 1 }
      ]
    },
    options: {
      responsive: true, animation: false,
      plugins: { legend: { labels: { font: { size: 11 } } } },
      scales: {
        x: { title: { display: true, text: '키 (cm)', font: { size: 11 } }, ticks: { maxTicksLimit: 10, font: { size: 10 } } },
        y: { beginAtZero: true, title: { display: true, text: '확률 밀도', font: { size: 11 } } }
      }
    }
  });
  activeChartInstances['c222'] = chart;

  function refresh() {
    document.getElementById('c222-mu-lbl').textContent = mu.toFixed(1);
    document.getElementById('c222-si-lbl').textContent = si.toFixed(1);
    document.getElementById('c222-mu-v').textContent = mu.toFixed(2);
    document.getElementById('c222-si-v').textContent = si.toFixed(2);
    document.getElementById('c222-logl').textContent = ch2.logL(data, mu, si).toFixed(2);
    chart.data.datasets[1].data = binCenters.map(x => ch2.pdf(x, mu, si));
    chart.update();
  }

  window.ch2_222_update = () => {
    mu = parseFloat(document.getElementById('c222-mu-sl').value);
    si = parseFloat(document.getElementById('c222-si-sl').value);
    refresh();
  };
  window.ch2_222_setMLE = () => {
    mu = muMLE; si = siMLE;
    document.getElementById('c222-mu-sl').value = mu.toFixed(1);
    document.getElementById('c222-si-sl').value = si.toFixed(1);
    refresh();
  };
  refresh();
};


/* ─────────────────────── 2.3.1 ─────────────────────── */
CONTENT['2.3.1'] = () => String.raw`
<div class="page-title">가능도 최대화</div>
<div class="page-subtitle">2.3 최대 가능도 추정 이론</div>

<div class="section">
  <div class="section-title"><span class="icon">🔗</span> 결합 확률과 가능도 함수</div>
  <p>샘플 $\mathcal{D} = \{x^{(1)}, x^{(2)}, \ldots, x^{(N)}\}$을 얻었다고 가정합니다. 각 샘플은 확률 분포 $p(x;\theta)$에 따라 <strong>독립적으로</strong> 생성되므로, 이 샘플 전체를 관측할 확률 밀도는:</p>
  <div class="math-block">
    $$p(\mathcal{D};\theta) = \prod_{n=1}^N p(x^{(n)};\theta) = p(x^{(1)};\theta)\cdot p(x^{(2)};\theta)\cdots p(x^{(N)};\theta)$$
  </div>
  <p>이를 매개변수 $\theta$의 함수로 볼 때 <strong>가능도 함수(Likelihood function)</strong>라고 합니다:</p>
  <div class="math-block">
    $$L(\theta) = p(\mathcal{D};\theta) = \prod_{n=1}^N p(x^{(n)};\theta)$$
  </div>
  <div class="highlight-box">
    <strong>직관:</strong> $L(\theta)$가 크다 = "매개변수 $\theta$일 때 이 샘플들이 관측될 확률이 높다" = 데이터를 잘 설명하는 $\theta$
  </div>
</div>

<div class="section">
  <div class="section-title"><span class="icon">📝</span> 로그 가능도를 사용하는 이유</div>
  <p>실제로는 <strong>로그 가능도(log-likelihood)</strong>를 최대화합니다:</p>
  <div class="math-block">
    $$\log L(\theta) = \log \prod_{n=1}^N p(x^{(n)};\theta) = \sum_{n=1}^N \log p(x^{(n)};\theta)$$
  </div>
  <ul class="content-list">
    <li><strong>수치 안정성:</strong> 작은 확률값을 여러 번 곱하면 수치적으로 0에 가까워지지만, 로그를 취하면 합산 가능</li>
    <li><strong>최적값 불변:</strong> $\log$는 단조증가 함수이므로, $L(\theta)$의 최댓값과 $\log L(\theta)$의 최댓값은 같은 $\theta$에서 달성</li>
  </ul>
</div>

<div class="section">
  <div class="section-title"><span class="icon">🔬</span> 인터랙티브: 가능도 탐색기 — MLE(Maximum Likelihood Estimation)의 핵심 원리</div>
  <p>아래 5개 샘플 데이터에 대해 슬라이더로 $\mu$를 조절하면서 로그 가능도가 어떻게 변하는지 관찰하세요.</p>
  <div class="highlight-box amber">
    <strong>탐구 과제:</strong> 로그 가능도 $\log L(\mu)$가 최대가 되는 $\mu$를 찾아보세요. <strong>샘플 평균</strong>과 어떤 관계가 있나요?
  </div>
  <div class="interactive-panel">
    <div class="panel-header">🔬 가능도 함수 탐색기 (σ = 5.5 고정, μ를 탐색)</div>
    <div class="panel-body" style="flex-direction:column;gap:14px;">
      <div class="highlight-box" style="margin:0;font-family:monospace;font-size:0.88rem;">
        <strong>샘플 데이터 (5개):</strong> 167.3, 172.1, 168.8, 175.2, 170.5 cm
        &nbsp;|&nbsp; <strong>샘플 평균 (MLE 최적 μ):</strong>
        <span id="c231-mle" style="color:#22c55e;font-weight:700;">—</span>
      </div>
      <div class="ctrl-group" style="margin:0;">
        <div class="ctrl-label">시도 매개변수 μ = <span id="c231-mu-v" style="font-weight:700;color:#7c3aed;">—</span> cm</div>
        <div style="display:flex;align-items:center;gap:8px;">
          <span style="font-size:0.75rem;color:#94a3b8;">160</span>
          <input type="range" id="c231-sl" min="160" max="182" step="0.1" value="168" style="flex:1;" oninput="ch2_231_update()">
          <span style="font-size:0.75rem;color:#94a3b8;">182</span>
        </div>
      </div>
      <div class="stat-grid" style="grid-template-columns:repeat(4,1fr);">
        <div class="stat-card">
          <div class="label">샘플 평균 (최적 μ)</div>
          <div class="value" id="c231-mle2" style="color:#22c55e;font-size:0.95rem;">—</div>
        </div>
        <div class="stat-card">
          <div class="label">현재 log L(μ)</div>
          <div class="value" id="c231-logl" style="color:#3b82f6;font-size:0.85rem;">—</div>
        </div>
        <div class="stat-card">
          <div class="label">최적 log L (MLE)</div>
          <div class="value" id="c231-logl-max" style="color:#22c55e;font-size:0.85rem;">—</div>
        </div>
        <div class="stat-card">
          <div class="label">손실 Δ (클수록 나쁨)</div>
          <div class="value" id="c231-delta" style="color:#ef4444;font-size:0.85rem;">—</div>
        </div>
      </div>
      <div>
        <div style="font-size:0.8rem;color:#64748b;margin-bottom:4px;">
          <strong>① 확률 밀도 함수</strong> $\mathcal{N}(x;\mu,\;5.5^2)$
          &nbsp;·&nbsp; 🔴 빨간점 = 각 데이터의 밀도값 $p(x^{(i)};\mu)$
          &nbsp;·&nbsp; 🟣 점선 = 현재 μ 위치
        </div>
        <canvas id="c231-pdf" height="200"></canvas>
      </div>
      <div>
        <div style="font-size:0.8rem;color:#64748b;margin-bottom:4px;">
          <strong>② 로그 가능도 프로파일</strong> $\log L(\mu)$
          &nbsp;·&nbsp; 🔴 = 현재 μ
          &nbsp;·&nbsp; 🟢 ★ = MLE 최적 μ (샘플 평균)
        </div>
        <canvas id="c231-logl-c" height="160"></canvas>
      </div>
    </div>
  </div>
</div>
`;

CHART_INITS['2.3.1'] = function () {
  const DATA = [167.3, 172.1, 168.8, 175.2, 170.5];
  const SIGMA = 5.5;
  const mleMu = ch2.mu(DATA);  // = 170.78
  const maxLogL = ch2.logL(DATA, mleMu, SIGMA);

  document.getElementById('c231-mle').textContent = mleMu.toFixed(2) + ' cm';
  document.getElementById('c231-mle2').textContent = mleMu.toFixed(2) + ' cm';
  document.getElementById('c231-logl-max').textContent = maxLogL.toFixed(3);

  const xs = ch2.ls(156, 187, 140);
  const muRange = ch2.ls(160, 182, 110);
  const logLCurve = muRange.map(mu => ch2.logL(DATA, mu, SIGMA));

  let muTry = 168;  // intentionally suboptimal start

  // === Chart 1: PDF + data points ===
  const ctx1 = document.getElementById('c231-pdf').getContext('2d');
  const pdfChart = new Chart(ctx1, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'PDF N(μ, 5.5²)',
          type: 'line',
          data: xs.map(x => ({ x, y: ch2.pdf(x, muTry, SIGMA) })),
          borderColor: '#3b82f6', borderWidth: 2,
          backgroundColor: 'rgba(59,130,246,0.07)', fill: true,
          pointRadius: 0, tension: 0.4, order: 3
        },
        {
          label: '데이터 p(xᵢ;μ)',
          data: DATA.map(x => ({ x, y: ch2.pdf(x, muTry, SIGMA) })),
          backgroundColor: '#ef4444', borderColor: '#b91c1c',
          pointRadius: 7, order: 1
        },
        {
          label: 'μ 위치',
          type: 'line',
          data: [{ x: muTry, y: 0 }, { x: muTry, y: ch2.pdf(muTry, muTry, SIGMA) * 1.2 }],
          borderColor: '#7c3aed', borderWidth: 2,
          borderDash: [5, 4], pointRadius: 0, order: 2
        }
      ]
    },
    options: {
      responsive: true, animation: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (c) => c.datasetIndex === 1 ? `p(${c.parsed.x.toFixed(1)}; μ) = ${c.parsed.y.toFixed(5)}` : '' } }
      },
      scales: {
        x: { type: 'linear', min: 156, max: 187, title: { display: true, text: '키 (cm)', font: { size: 11 } } },
        y: { beginAtZero: true, title: { display: true, text: '확률 밀도 p(x; μ)', font: { size: 11 } }, ticks: { font: { size: 10 } } }
      }
    }
  });

  // === Chart 2: Log-likelihood profile ===
  const ctx2 = document.getElementById('c231-logl-c').getContext('2d');
  const logLChart = new Chart(ctx2, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'log L(μ)',
          type: 'line',
          data: muRange.map((mu, i) => ({ x: mu, y: logLCurve[i] })),
          borderColor: '#6366f1', borderWidth: 2,
          backgroundColor: 'rgba(99,102,241,0.07)', fill: true,
          pointRadius: 0, tension: 0.4, order: 3
        },
        {
          label: '현재 μ',
          data: [{ x: muTry, y: ch2.logL(DATA, muTry, SIGMA) }],
          backgroundColor: '#ef4444', borderColor: '#b91c1c',
          pointRadius: 9, pointStyle: 'circle', order: 1
        },
        {
          label: 'MLE 최적 μ ★',
          data: [{ x: mleMu, y: maxLogL }],
          backgroundColor: '#22c55e', borderColor: '#15803d',
          pointRadius: 10, pointStyle: 'star', order: 2
        }
      ]
    },
    options: {
      responsive: true, animation: false,
      plugins: { legend: { labels: { font: { size: 10 } } } },
      scales: {
        x: { type: 'linear', min: 160, max: 182, title: { display: true, text: 'μ (cm)', font: { size: 11 } } },
        y: { title: { display: true, text: 'log L(μ)', font: { size: 11 } }, ticks: { font: { size: 10 } } }
      }
    }
  });

  activeChartInstances['c231pdf'] = pdfChart;
  activeChartInstances['c231logl'] = logLChart;

  window.ch2_231_update = () => {
    muTry = parseFloat(document.getElementById('c231-sl').value);
    const logLVal = ch2.logL(DATA, muTry, SIGMA);
    const delta = logLVal - maxLogL;

    document.getElementById('c231-mu-v').textContent = muTry.toFixed(1);
    document.getElementById('c231-logl').textContent = logLVal.toFixed(3);
    document.getElementById('c231-delta').textContent = delta.toFixed(3);

    pdfChart.data.datasets[0].data = xs.map(x => ({ x, y: ch2.pdf(x, muTry, SIGMA) }));
    pdfChart.data.datasets[1].data = DATA.map(x => ({ x, y: ch2.pdf(x, muTry, SIGMA) }));
    pdfChart.data.datasets[2].data = [{ x: muTry, y: 0 }, { x: muTry, y: ch2.pdf(muTry, muTry, SIGMA) * 1.2 }];
    pdfChart.update();

    logLChart.data.datasets[1].data = [{ x: muTry, y: logLVal }];
    logLChart.update();
  };

  window.ch2_231_update();
};


/* ─────────────────────── 2.3.2 ─────────────────────── */
CONTENT['2.3.2'] = () => String.raw`
<div class="page-title">미분으로 최댓값 찾기</div>
<div class="page-subtitle">2.3 최대 가능도 추정 이론</div>

<div class="section">
  <div class="section-title"><span class="icon">📈</span> 미분과 최댓값의 관계</div>
  <p>함수의 <strong>최댓값</strong>은 도함수(미분)가 0이 되는 지점에서 발생합니다 (2차항 계수가 음수인 경우).</p>
  <div class="highlight-box">
    함수 $y = f(x)$의 최댓값 → 기울기(접선의 기울기)가 0인 지점 → $\dfrac{dy}{dx} = 0$
  </div>
  <p>예시 함수를 생각해봅시다:</p>
  <div class="math-block">$$y = -2x^2 + 3x + 4$$</div>
  <p>$x$에 대해 미분하면:</p>
  <div class="math-block">$$\frac{dy}{dx} = -4x + 3$$</div>
  <p>최댓값은 $\dfrac{dy}{dx} = 0$인 점:</p>
  <div class="math-block">$$-4x + 3 = 0 \quad \Leftrightarrow \quad x = \frac{3}{4} = 0.75$$</div>
  <div class="highlight-box green">
    <strong>MLE(Maximum Likelihood Estimation)에의 적용:</strong> 로그 가능도 $\log L(\theta)$를 $\theta$에 대해 미분하여 0으로 놓으면<br>
    → 최대 가능도 추정값 $\hat{\theta}$ 도출
  </div>
</div>

<div class="section">
  <div class="section-title"><span class="icon">🎮</span> 인터랙티브: 미분으로 최댓값 탐색</div>
  <p>슬라이더로 $x$ 값을 이동하면서 접선의 기울기($dy/dx$)가 0이 되는 지점을 찾아보세요.</p>
  <div class="interactive-panel">
    <div class="panel-header">📐 y = −2x² + 3x + 4 — 미분으로 최댓값 찾기</div>
    <div class="panel-body" style="flex-direction:column;gap:14px;">
      <div class="ctrl-group" style="margin:0;">
        <div class="ctrl-label">x = <span id="c232-x">−0.5</span></div>
        <div style="display:flex;align-items:center;gap:8px;">
          <span style="font-size:0.75rem;color:#94a3b8;">-1</span>
          <input type="range" id="c232-sl" min="-1" max="2" step="0.01" value="-0.5" style="flex:1;" oninput="ch2_232_update()">
          <span style="font-size:0.75rem;color:#94a3b8;">2</span>
        </div>
      </div>
      <div class="stat-grid" style="grid-template-columns:repeat(3,1fr);">
        <div class="stat-card"><div class="label">현재 x</div><div class="value" id="c232-xv">−0.5</div></div>
        <div class="stat-card"><div class="label">y 값</div><div class="value" id="c232-yv" style="color:#3b82f6;">—</div></div>
        <div class="stat-card"><div class="label">기울기 dy/dx</div><div class="value" id="c232-dydx" style="color:#ef4444;">—</div></div>
      </div>
      <div id="c232-hint" style="text-align:center;font-size:0.85rem;font-weight:600;color:#22c55e;padding:6px;background:#f0fdf4;border-radius:8px;display:none;">
        ✅ 기울기 ≈ 0 — 최댓값 부근입니다!
      </div>
      <canvas id="c232-chart" height="260"></canvas>
    </div>
  </div>
</div>
`;

CHART_INITS['2.3.2'] = function () {
  const f = x => -2 * x * x + 3 * x + 4;
  const df = x => -4 * x + 3;
  const xs = ch2.ls(-1, 2, 120);
  let xCur = -0.5;

  function tangent(x0, span = 0.45) {
    const y0 = f(x0), k = df(x0);
    return [{ x: x0 - span, y: y0 - k * span }, { x: x0 + span, y: y0 + k * span }];
  }

  const ctx = document.getElementById('c232-chart').getContext('2d');
  const chart = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'y = -2x² + 3x + 4',
          type: 'line',
          data: xs.map(x => ({ x, y: f(x) })),
          borderColor: '#6366f1', borderWidth: 2.5,
          backgroundColor: 'rgba(99,102,241,0.07)', fill: true,
          pointRadius: 0, tension: 0.4, order: 3
        },
        {
          label: '접선 (기울기 = dy/dx)',
          type: 'line',
          data: tangent(xCur),
          borderColor: '#ef4444', borderWidth: 2,
          borderDash: [5, 3], pointRadius: 0, order: 2
        },
        {
          label: '현재 점',
          data: [{ x: xCur, y: f(xCur) }],
          backgroundColor: '#3b82f6', borderColor: '#1d4ed8',
          pointRadius: 8, order: 1
        },
        {
          label: '최댓값 (x = 0.75) ★',
          data: [{ x: 0.75, y: f(0.75) }],
          backgroundColor: '#22c55e', borderColor: '#15803d',
          pointRadius: 9, pointStyle: 'star', order: 0
        }
      ]
    },
    options: {
      responsive: true, animation: false,
      plugins: { legend: { labels: { font: { size: 11 } } } },
      scales: {
        x: { type: 'linear', min: -1, max: 2, title: { display: true, text: 'x', font: { size: 11 } } },
        y: { title: { display: true, text: 'y', font: { size: 11 } } }
      }
    }
  });
  activeChartInstances['c232'] = chart;

  window.ch2_232_update = () => {
    xCur = parseFloat(document.getElementById('c232-sl').value);
    const yVal = f(xCur), dydxVal = df(xCur);
    document.getElementById('c232-x').textContent = xCur.toFixed(2);
    document.getElementById('c232-xv').textContent = xCur.toFixed(2);
    document.getElementById('c232-yv').textContent = yVal.toFixed(3);
    document.getElementById('c232-dydx').textContent = dydxVal.toFixed(3);
    document.getElementById('c232-hint').style.display = Math.abs(dydxVal) < 0.2 ? 'block' : 'none';
    chart.data.datasets[1].data = tangent(xCur);
    chart.data.datasets[2].data = [{ x: xCur, y: yVal }];
    chart.update();
  };
  window.ch2_232_update();
};


/* ─────────────────────── 2.3.3 ─────────────────────── */
CONTENT['2.3.3'] = () => String.raw`
<div class="page-title">정규 분포의 MLE(Maximum Likelihood Estimation) 유도</div>
<div class="page-subtitle">2.3 최대 가능도 추정 이론</div>

<div class="section">
  <div class="section-title"><span class="icon">🧮</span> 로그 가능도 전개</div>
  <p>정규 분포 $\mathcal{N}(x;\mu,\sigma)$에 대해 로그 가능도를 전개합니다.
  (정규 분포의 기본 성질은
  <a href="#" class="ch-link" onclick="(function(){var s=FLAT_SECTIONS.find(function(x){return x.id==='1.2.1';});if(s)loadSection(s.id,s.ch,s.sub,s.sec);})();return false;">→ Ch.1.2 정규분포</a>
  참고)</p>
  <div class="math-block">
    $$\log p(\mathcal{D};\mu,\sigma)
    = \log \prod_{n=1}^N \frac{1}{\sqrt{2\pi}\,\sigma}\exp\!\left(-\frac{(x^{(n)}-\mu)^2}{2\sigma^2}\right)$$
  </div>
  <div class="math-block">
    $$= -\frac{N}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{n=1}^N(x^{(n)}-\mu)^2$$
  </div>
</div>

<div class="section">
  <div class="section-title"><span class="icon">①</span> μ̂ 유도 — 평균에 대한 편미분</div>
  <p>$\mu$에 대해 편미분하여 0으로 놓습니다:</p>
  <div class="math-block">
    $$\frac{\partial \log L}{\partial \mu} = \frac{1}{\sigma^2}\sum_{n=1}^N\!\bigl(x^{(n)}-\mu\bigr) = 0$$
  </div>
  <div class="math-block">
    $$\Leftrightarrow \sum_{n=1}^N x^{(n)} = N\mu
    \qquad \therefore\; \boxed{\hat{\mu} = \frac{1}{N}\sum_{n=1}^N x^{(n)}}$$
  </div>
  <div class="highlight-box green">
    결론: <strong>MLE(Maximum Likelihood Estimation) 평균 추정값 $\hat{\mu}$는 샘플의 산술 평균</strong>과 일치합니다.
  </div>
</div>

<div class="section">
  <div class="section-title"><span class="icon">②</span> σ̂ 유도 — 표준편차에 대한 편미분</div>
  <p>$\hat{\mu}$를 대입 후, $\sigma$에 대해 편미분하여 0으로 놓습니다:</p>
  <div class="math-block">
    $$\frac{\partial \log L}{\partial \sigma}\bigg|_{\mu=\hat{\mu}}
    = -\frac{N}{\sigma} + \frac{1}{\sigma^3}\sum_{n=1}^N\!\bigl(x^{(n)}-\hat{\mu}\bigr)^2 = 0$$
  </div>
  <div class="math-block">
    $$\Leftrightarrow \sigma^2 = \frac{1}{N}\sum_{n=1}^N\!\bigl(x^{(n)}-\hat{\mu}\bigr)^2
    \qquad \therefore\; \boxed{\hat{\sigma} = \sqrt{\frac{1}{N}\sum_{n=1}^N\!\bigl(x^{(n)}-\hat{\mu}\bigr)^2}}$$
  </div>
  <div class="highlight-box green">
    결론: <strong>MLE(Maximum Likelihood Estimation) 표준편차 추정값 $\hat{\sigma}$는 샘플의 표준편차</strong>와 일치합니다.
  </div>
</div>

<div class="section">
  <div class="section-title"><span class="icon">🔢</span> 인터랙티브: MLE(Maximum Likelihood Estimation) 계산기</div>
  <p>데이터를 직접 입력하면 MLE(Maximum Likelihood Estimation) 방식으로 $\hat{\mu}$와 $\hat{\sigma}$를 계산합니다.</p>
  <div class="interactive-panel">
    <div class="panel-header">🧮 MLE 계산기</div>
    <div class="panel-body" style="flex-direction:column;gap:12px;">
      <div>
        <div class="ctrl-label">데이터 입력 (쉼표로 구분):</div>
        <input type="text" id="c233-input" value="167.3, 172.1, 168.8, 175.2, 170.5"
          style="width:100%;padding:8px 12px;border:1.5px solid #e2e8f0;border-radius:8px;font-size:0.88rem;font-family:monospace;margin-top:6px;">
      </div>
      <button class="btn btn-primary" onclick="ch2_233_calc()">📐 MLE 계산하기</button>
      <div id="c233-result" style="display:none;">
        <div class="stat-grid" style="grid-template-columns:repeat(3,1fr);">
          <div class="stat-card"><div class="label">N (데이터 수)</div><div class="value" id="c233-n">—</div></div>
          <div class="stat-card"><div class="label">μ̂ (MLE 평균)</div><div class="value" id="c233-mu" style="color:#3b82f6;">—</div></div>
          <div class="stat-card"><div class="label">σ̂ (MLE 표준편차)</div><div class="value" id="c233-si" style="color:#7c3aed;">—</div></div>
        </div>
        <div class="stat-grid" style="grid-template-columns:1fr 1fr;margin-top:8px;">
          <div class="stat-card"><div class="label">log L(μ̂, σ̂)</div><div class="value" id="c233-logl" style="color:#22c55e;font-size:0.9rem;">—</div></div>
          <div class="stat-card"><div class="label">계산 과정 (μ̂)</div><div class="value" id="c233-proc" style="font-size:0.7rem;color:#64748b;font-family:monospace;word-break:break-all;line-height:1.4;">—</div></div>
        </div>
      </div>
    </div>
  </div>
</div>
`;

CHART_INITS['2.3.3'] = function () {
  window.ch2_233_calc = () => {
    const raw = document.getElementById('c233-input').value;
    const data = raw.split(',').map(s => parseFloat(s.trim())).filter(x => !isNaN(x));
    if (data.length < 2) { alert('최소 2개 이상의 숫자를 입력해 주세요.'); return; }
    const n = data.length;
    const muHat = ch2.mu(data), siHat = ch2.si(data);
    document.getElementById('c233-n').textContent = n;
    document.getElementById('c233-mu').textContent = muHat.toFixed(4);
    document.getElementById('c233-si').textContent = siHat.toFixed(4);
    document.getElementById('c233-logl').textContent = ch2.logL(data, muHat, siHat).toFixed(4);
    const sumPart = data.map(x => x.toFixed(1)).join(' + ');
    document.getElementById('c233-proc').textContent = `(${sumPart}) / ${n} = ${muHat.toFixed(4)}`;
    document.getElementById('c233-result').style.display = 'block';
  };
  window.ch2_233_calc();
};


/* ─────────────────────── 2.4.1 ─────────────────────── */
CONTENT['2.4.1'] = () => String.raw`
<div class="page-title">새로운 데이터 생성</div>
<div class="page-subtitle">2.4 생성 모델의 용도</div>

<div class="section">
  <div class="section-title"><span class="icon">✨</span> 생성 모델의 첫 번째 용도: 새 데이터 생성</div>
  <p>매개변수를 추정(학습)하고 나면, 추정된 정규 분포 $\mathcal{N}(\hat{\mu},\,\hat{\sigma}^2)$에서 <strong>새로운 데이터를 무한히 생성</strong>할 수 있습니다.</p>
  <ul class="content-list">
    <li><strong>매개변수 추정 (학습):</strong> 샘플에서 $\hat{\mu}$, $\hat{\sigma}$ 계산</li>
    <li><strong>새 데이터 생성:</strong> $\mathcal{N}(\hat{\mu},\,\hat{\sigma}^2)$에서 샘플링</li>
    <li><strong>검증:</strong> 생성된 데이터와 원본 데이터의 분포가 유사한지 확인</li>
  </ul>
  <div class="highlight-box">
    생성된 데이터는 원본 데이터와 <em>동일하지 않지만</em>, 같은 확률 분포에서 추출된 것이므로 유사한 통계적 특성을 가집니다.
  </div>
</div>

<div class="section">
  <div class="section-title"><span class="icon">💻</span> 생성 코드</div>
  <pre class="code-pre"><code class="language-python">import numpy as np
import matplotlib.pyplot as plt

xs = np.loadtxt('height.txt')
mu_hat = np.mean(xs)
sigma_hat = np.std(xs)

# 새로운 데이터 10,000개 생성
generated = np.random.normal(mu_hat, sigma_hat, 10000)

plt.hist(xs, bins='auto', density=True, alpha=0.7, label='원본 데이터')
plt.hist(generated, bins='auto', density=True, alpha=0.7, label='생성 데이터')
plt.legend()
plt.show()</code></pre>
</div>

<div class="section">
  <div class="section-title"><span class="icon">🎲</span> 인터랙티브: 새 데이터 생성 데모</div>
  <p>버튼을 눌러 추정된 분포에서 새로운 데이터를 생성하고 원본과 비교해 보세요.</p>
  <div class="interactive-panel">
    <div class="panel-header">🎲 데이터 생성 & 원본 비교</div>
    <div class="panel-body" style="flex-direction:column;gap:12px;">
      <div class="stat-grid" style="grid-template-columns:repeat(3,1fr);">
        <div class="stat-card"><div class="label">원본 μ̂</div><div class="value" id="c241-mu" style="color:#3b82f6;">—</div></div>
        <div class="stat-card"><div class="label">원본 σ̂</div><div class="value" id="c241-si" style="color:#7c3aed;">—</div></div>
        <div class="stat-card"><div class="label">생성 샘플 수</div><div class="value" id="c241-gn" style="color:#22c55e;">대기중...</div></div>
      </div>
      <button class="btn btn-green" onclick="ch2_241_gen()">✨ 새 데이터 1000개 생성</button>
      <canvas id="c241-chart" height="260"></canvas>
      <div class="highlight-box green" id="c241-msg" style="display:none;">
        생성된 데이터(주황색)와 원본 데이터(파란색)가 유사한 분포를 보입니다! 🎉<br>
        더 복잡한 생성 모델은 이미지도 생성할 수 있습니다.
        <a href="#" class="ch-link" onclick="toggleChapter(7,true);return false;">→ Ch.7 변이형 오토인코더(VAE)</a>
        <a href="#" class="ch-link" onclick="toggleChapter(8,true);return false;">→ Ch.8 확산모델</a>
      </div>
    </div>
  </div>
</div>
`;

CHART_INITS['2.4.1'] = function () {
  const data = ch2.hd;
  const muHat = ch2.mu(data), siHat = ch2.si(data);
  document.getElementById('c241-mu').textContent = muHat.toFixed(2) + ' cm';
  document.getElementById('c241-si').textContent = siHat.toFixed(2) + ' cm';

  const NUM_BINS = 20, MIN = 150, MAX = 195;
  const bw = (MAX - MIN) / NUM_BINS;
  const binLabels = Array.from({ length: NUM_BINS }, (_, i) => (MIN + (i + 0.5) * bw).toFixed(1));

  function density(d) {
    const c = new Array(NUM_BINS).fill(0);
    d.forEach(x => { const b = Math.min(Math.floor((x - MIN) / bw), NUM_BINS - 1); if (b >= 0) c[b]++; });
    return c.map(v => v / (d.length * bw));
  }

  const ctx = document.getElementById('c241-chart').getContext('2d');
  const chart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: binLabels,
      datasets: [
        { label: '원본 데이터', data: density(data), backgroundColor: 'rgba(59,130,246,0.5)', borderColor: '#2563eb', borderWidth: 1, barPercentage: 1.0, categoryPercentage: 1.0 },
        { label: '생성 데이터', data: new Array(NUM_BINS).fill(0), backgroundColor: 'rgba(249,115,22,0.5)', borderColor: '#ea580c', borderWidth: 1, barPercentage: 1.0, categoryPercentage: 1.0 }
      ]
    },
    options: {
      responsive: true, animation: { duration: 500 },
      plugins: { legend: { labels: { font: { size: 11 } } } },
      scales: {
        x: { title: { display: true, text: '키 (cm)', font: { size: 11 } }, ticks: { maxTicksLimit: 10, font: { size: 10 } } },
        y: { beginAtZero: true, title: { display: true, text: '확률 밀도', font: { size: 11 } } }
      }
    }
  });
  activeChartInstances['c241'] = chart;

  window.ch2_241_gen = () => {
    const generated = ch2.gen(1000, muHat, siHat);
    chart.data.datasets[1].data = density(generated);
    chart.update();
    document.getElementById('c241-gn').textContent = '1000개 생성됨';
    document.getElementById('c241-msg').style.display = 'block';
  };
};


/* ─────────────────────── 2.4.2 ─────────────────────── */
CONTENT['2.4.2'] = () => String.raw`
<div class="page-title">확률 계산</div>
<div class="page-subtitle">2.4 생성 모델의 용도</div>

<div class="section">
  <div class="section-title"><span class="icon">📊</span> 생성 모델의 두 번째 용도: 확률 계산</div>
  <p>확률 분포를 알면 <strong>특정 값이 얼마나 발생하기 쉬운지</strong> 정량적으로 계산할 수 있습니다. 연속 분포에서 특정 구간의 확률은 <strong>누적 분포 함수(CDF)</strong>를 이용합니다.
  (연속 분포와 확률 밀도의 개념은
  <a href="#" class="ch-link" onclick="(function(){var s=FLAT_SECTIONS.find(function(x){return x.id==='1.1.2';});if(s)loadSection(s.id,s.ch,s.sub,s.sec);})();return false;">→ Ch.1.1.2 확률 분포의 종류</a>
  참고)</p>
  <div class="math-block">
    $$P(X \le x) = \int_{-\infty}^{x} p(t;\,\hat{\mu},\,\hat{\sigma})\,dt = \Phi\!\left(\frac{x - \hat{\mu}}{\hat{\sigma}}\right)$$
  </div>
  <ul class="content-list">
    <li>$P(X \le 160)$: 키가 160 cm 이하일 확률</li>
    <li>$P(X > 180) = 1 - P(X \le 180)$: 키가 180 cm 초과일 확률</li>
  </ul>
</div>

<div class="section">
  <div class="section-title"><span class="icon">💻</span> 확률 계산 코드</div>
  <pre class="code-pre"><code class="language-python">import numpy as np
from scipy.stats import norm

xs = np.loadtxt('height.txt')
mu_hat, sigma_hat = np.mean(xs), np.std(xs)

p1 = norm.cdf(160, mu_hat, sigma_hat)
print(f'P(x <= 160): {p1:.4f}')    # ≈ 0.1001

p2 = norm.cdf(180, mu_hat, sigma_hat)
print(f'P(x > 180): {1 - p2:.4f}') # ≈ 0.0721</code></pre>
</div>

<div class="section">
  <div class="section-title"><span class="icon">🎛️</span> 인터랙티브: 누적 확률 계산기</div>
  <p>슬라이더로 키 기준값 $h$를 조절하면서 $P(X \le h)$와 $P(X > h)$를 시각적으로 확인하세요.</p>
  <div class="interactive-panel">
    <div class="panel-header">📊 누적 확률 계산기 — 피팅된 정규분포 N(μ̂, σ̂²)</div>
    <div class="panel-body" style="flex-direction:column;gap:14px;">
      <div class="ctrl-group" style="margin:0;">
        <div class="ctrl-label">키 기준값 h = <span id="c242-h" style="font-weight:700;">170.0</span> cm</div>
        <div style="display:flex;align-items:center;gap:8px;">
          <span style="font-size:0.75rem;color:#94a3b8;">150</span>
          <input type="range" id="c242-sl" min="150" max="195" step="0.5" value="170" style="flex:1;" oninput="ch2_242_update()">
          <span style="font-size:0.75rem;color:#94a3b8;">195</span>
        </div>
      </div>
      <div class="stat-grid" style="grid-template-columns:1fr 1fr;">
        <div class="stat-card">
          <div class="label">P(X ≤ h) — 파란 영역</div>
          <div class="value" id="c242-left" style="color:#3b82f6;font-size:1.15rem;">—</div>
        </div>
        <div class="stat-card">
          <div class="label">P(X > h) — 빨간 영역</div>
          <div class="value" id="c242-right" style="color:#ef4444;font-size:1.15rem;">—</div>
        </div>
      </div>
      <canvas id="c242-chart" height="260"></canvas>
    </div>
  </div>
</div>
`;

CHART_INITS['2.4.2'] = function () {
  const data = ch2.hd;
  const muHat = ch2.mu(data), siHat = ch2.si(data);
  const xs = ch2.ls(147, 197, 180);
  let h = 170;

  function buildDS(cutoff) {
    const leftXs = xs.filter(x => x <= cutoff + 0.5);
    const rightXs = xs.filter(x => x >= cutoff - 0.5);
    return [
      xs.map(x => ({ x, y: ch2.pdf(x, muHat, siHat) })),
      leftXs.map(x => ({ x, y: x <= cutoff ? ch2.pdf(x, muHat, siHat) : 0 })),
      rightXs.map(x => ({ x, y: x >= cutoff ? ch2.pdf(x, muHat, siHat) : 0 }))
    ];
  }

  const [fl, ld, rd] = buildDS(h);
  const ctx = document.getElementById('c242-chart').getContext('2d');
  const chart = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        { label: 'P(X ≤ h)', type: 'line', data: ld, borderColor: 'transparent', backgroundColor: 'rgba(59,130,246,0.35)', fill: 'origin', pointRadius: 0, tension: 0.4, order: 2 },
        { label: 'P(X > h)', type: 'line', data: rd, borderColor: 'transparent', backgroundColor: 'rgba(239,68,68,0.35)', fill: 'origin', pointRadius: 0, tension: 0.4, order: 3 },
        { label: '정규분포 PDF', type: 'line', data: fl, borderColor: '#374151', borderWidth: 2, backgroundColor: 'transparent', pointRadius: 0, tension: 0.4, order: 1 }
      ]
    },
    options: {
      responsive: true, animation: false,
      plugins: { legend: { labels: { font: { size: 11 } } }, tooltip: { enabled: false } },
      scales: {
        x: { type: 'linear', min: 147, max: 197, title: { display: true, text: '키 (cm)', font: { size: 11 } } },
        y: { beginAtZero: true, title: { display: true, text: '확률 밀도', font: { size: 11 } } }
      }
    }
  });
  activeChartInstances['c242'] = chart;

  window.ch2_242_update = () => {
    h = parseFloat(document.getElementById('c242-sl').value);
    document.getElementById('c242-h').textContent = h.toFixed(1);
    const pLeft = ch2.cdf(h, muHat, siHat);
    document.getElementById('c242-left').textContent = (pLeft * 100).toFixed(2) + '%';
    document.getElementById('c242-right').textContent = ((1 - pLeft) * 100).toFixed(2) + '%';
    const [fl2, ld2, rd2] = buildDS(h);
    chart.data.datasets[0].data = ld2;
    chart.data.datasets[1].data = rd2;
    chart.data.datasets[2].data = fl2;
    chart.update();
  };
  window.ch2_242_update();
};
