/* ============================================================
   Chapter 3 — 다변량 정규 분포 (Multivariate Normal Distribution)
   Content + Chart implementations
   ============================================================ */

/* ─────── 챕터 3 수학 유틸리티 ─────── */
(function () {
  function bmNorm() {
    let u1; do { u1 = Math.random(); } while (u1 === 0);
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * Math.random());
  }

  // 2D 다변량 정규 분포 PDF
  function pdf2d(x1, x2, mu1, mu2, s11, s12, s22) {
    const det = s11 * s22 - s12 * s12;
    if (det <= 0) return 0;
    const d1 = x1 - mu1, d2 = x2 - mu2;
    const mah2 = (s22 * d1 * d1 - 2 * s12 * d1 * d2 + s11 * d2 * d2) / det;
    return Math.exp(-0.5 * mah2) / (2 * Math.PI * Math.sqrt(det));
  }

  // 2x2 촐레스키 분해 Σ = L L^T
  function chol(s11, s12, s22) {
    const L11 = Math.sqrt(Math.max(s11, 1e-10));
    const L21 = s12 / L11;
    const L22 = Math.sqrt(Math.max(s22 - L21 * L21, 1e-10));
    return [L11, L21, L22];
  }

  // N개의 2D 정규분포 샘플 생성 (촐레스키 변환)
  function gen2d(n, mu1, mu2, s11, s12, s22) {
    const [L11, L21, L22] = chol(s11, s12, s22);
    return Array.from({ length: n }, () => {
      const z1 = bmNorm(), z2 = bmNorm();
      return [mu1 + L11 * z1, mu2 + L21 * z1 + L22 * z2];
    });
  }

  // k-σ 등고 타원 경계점 생성 ({x, y} 배열)
  function ellipse(mu1, mu2, s11, s12, s22, k, n) {
    n = n || 120;
    const [L11, L21, L22] = chol(s11, s12, s22);
    return Array.from({ length: n + 1 }, (_, i) => {
      const t = 2 * Math.PI * i / n;
      return {
        x: mu1 + k * (L11 * Math.cos(t)),
        y: mu2 + k * (L21 * Math.cos(t) + L22 * Math.sin(t))
      };
    });
  }

  // 2D 데이터의 MLE (최대 가능도 추정)
  function mle2d(data) {
    const n = data.length;
    const mu1 = data.reduce((a, p) => a + p[0], 0) / n;
    const mu2 = data.reduce((a, p) => a + p[1], 0) / n;
    let s11 = 0, s12 = 0, s22 = 0;
    data.forEach(p => {
      const d1 = p[0] - mu1, d2 = p[1] - mu2;
      s11 += d1 * d1; s12 += d1 * d2; s22 += d2 * d2;
    });
    return { mu1, mu2, s11: s11 / n, s12: s12 / n, s22: s22 / n };
  }

  // 밀도값 t(0~1)을 Jet 스타일 색상으로 변환
  function densityColor(t) {
    t = Math.max(0, Math.min(1, t));
    let r, g, b;
    if (t < 0.25) { r = 0; g = Math.round(t * 4 * 255); b = 255; }
    else if (t < 0.5) { r = 0; g = 255; b = Math.round(255 - (t - 0.25) * 4 * 255); }
    else if (t < 0.75) { r = Math.round((t - 0.5) * 4 * 255); g = 255; b = 0; }
    else { r = 255; g = Math.round(255 - (t - 0.75) * 4 * 200); b = 0; }
    return `rgba(${r},${g},${b},0.9)`;
  }

  window.ch3 = { pdf2d, gen2d, ellipse, mle2d, chol, densityColor };
})();


/* ─────────────────────── 3.1.1 ─────────────────────── */
CONTENT['3.1.1'] = () => String.raw`
<div class="page-title">다차원 배열</div>
<div class="page-subtitle">3.1 넘파이와 다차원 배열</div>

<div class="section">
  <div class="section-title"><span class="icon">📦</span> 다차원 배열이란?</div>
  <p>다차원 배열은 여러 값을 한 번에 처리하기 위한 데이터 구조입니다. 원소들이 배열되는 <strong>방향을 축(Axis)</strong>이라 하고, 축의 개수를 <strong>차원(Dimension)</strong>이라 합니다.</p>
  <div style="overflow-x:auto;margin:16px 0;">
    <table style="border-collapse:collapse;width:100%;font-size:0.88rem;">
      <thead>
        <tr style="background:#1e3a5f;color:#bfdbfe;">
          <th style="padding:10px 16px;text-align:center;">차원</th>
          <th style="padding:10px 16px;text-align:center;">이름</th>
          <th style="padding:10px 16px;text-align:center;">shape 예시</th>
          <th style="padding:10px 16px;text-align:left;">예시</th>
        </tr>
      </thead>
      <tbody>
        <tr style="background:#f8fafc;">
          <td style="padding:9px 16px;text-align:center;border:1px solid #e2e8f0;font-weight:700;color:#1d4ed8;">0차원</td>
          <td style="padding:9px 16px;text-align:center;border:1px solid #e2e8f0;">스칼라</td>
          <td style="padding:9px 16px;text-align:center;border:1px solid #e2e8f0;font-family:monospace;">()</td>
          <td style="padding:9px 16px;border:1px solid #e2e8f0;">3.14</td>
        </tr>
        <tr>
          <td style="padding:9px 16px;text-align:center;border:1px solid #e2e8f0;font-weight:700;color:#1d4ed8;">1차원</td>
          <td style="padding:9px 16px;text-align:center;border:1px solid #e2e8f0;">벡터</td>
          <td style="padding:9px 16px;text-align:center;border:1px solid #e2e8f0;font-family:monospace;">(3,)</td>
          <td style="padding:9px 16px;border:1px solid #e2e8f0;">[1, 2, 3]</td>
        </tr>
        <tr style="background:#f8fafc;">
          <td style="padding:9px 16px;text-align:center;border:1px solid #e2e8f0;font-weight:700;color:#1d4ed8;">2차원</td>
          <td style="padding:9px 16px;text-align:center;border:1px solid #e2e8f0;">행렬</td>
          <td style="padding:9px 16px;text-align:center;border:1px solid #e2e8f0;font-family:monospace;">(2, 3)</td>
          <td style="padding:9px 16px;border:1px solid #e2e8f0;">[[1,2,3],[4,5,6]]</td>
        </tr>
        <tr>
          <td style="padding:9px 16px;text-align:center;border:1px solid #e2e8f0;font-weight:700;color:#1d4ed8;">3차원</td>
          <td style="padding:9px 16px;text-align:center;border:1px solid #e2e8f0;">텐서</td>
          <td style="padding:9px 16px;text-align:center;border:1px solid #e2e8f0;font-family:monospace;">(2, 3, 4)</td>
          <td style="padding:9px 16px;border:1px solid #e2e8f0;">이미지 배치 등</td>
        </tr>
      </tbody>
    </table>
  </div>
</div>

<div class="section">
  <div class="section-title"><span class="icon">📐</span> 벡터 표현 방법</div>
  <p>벡터는 열벡터(column vector)와 행벡터(row vector)로 표현할 수 있습니다.</p>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:16px 0;">
    <div class="highlight-box">
      <strong>열벡터 (Column vector)</strong><br>세로 방향으로 배열
      <div class="math-block" style="margin:10px 0;">$$x = \begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_D \end{pmatrix}$$</div>
      shape: $(D, 1)$ 또는 $(D,)$
    </div>
    <div class="highlight-box green">
      <strong>행벡터 (Row vector)</strong><br>가로 방향으로 배열
      <div class="math-block" style="margin:10px 0;">$$x^\top = (x_1,\; x_2,\; \cdots,\; x_D)$$</div>
      shape: $(1, D)$<br>
      <small>* 열벡터의 전치(Transpose)</small>
    </div>
  </div>
  <div class="highlight-box amber">
    <strong>이 책의 규약:</strong> 특별한 설명이 없으면 벡터 $x$는 <strong>열벡터</strong>입니다. shape = $(D,)$ 또는 $(D, 1)$.
  </div>
</div>

<div class="section">
  <div class="section-title"><span class="icon">🎯</span> 왜 다차원 배열이 필요한가?</div>
  <p>1장에서는 키(스칼라, 1차원)를 다뤘습니다. 이번 장에서는 <strong>키와 몸무게를 동시에</strong> 다룹니다 — 이것이 벡터(2차원 데이터)입니다.</p>
  <div class="math-block">$$x = \begin{pmatrix} \text{키} \\ \text{몸무게} \end{pmatrix} \in \mathbb{R}^2$$</div>
  <p>더 일반적으로, $D$개의 특성을 가진 데이터 포인트는 $D$차원 벡터 $x \in \mathbb{R}^D$로 표현됩니다.</p>
</div>
`;


/* ─────────────────────── 3.1.2 ─────────────────────── */
CONTENT['3.1.2'] = () => String.raw`
<div class="page-title">넘파이의 다차원 배열</div>
<div class="page-subtitle">3.1 넘파이와 다차원 배열</div>

<div class="section">
  <div class="section-title"><span class="icon">🐍</span> 넘파이로 벡터와 행렬 만들기</div>
  <p>넘파이(NumPy)의 <code>np.array()</code>로 다차원 배열을 생성합니다. <code>.shape</code>는 각 축의 크기, <code>.ndim</code>은 차원 수를 반환합니다.</p>
  <pre class="code-pre"><code class="language-python">import numpy as np

# 1차원 배열 (벡터)
x = np.array([1, 2, 3])
print(x.__class__)  # <class 'numpy.ndarray'>
print(x.shape)      # (3,)
print(x.ndim)       # 1

# 2차원 배열 (행렬)
W = np.array([[1, 2, 3],
              [4, 5, 6]])
print(W.ndim)   # 2
print(W.shape)  # (2, 3)  -> 2행 3열</code></pre>
</div>

<div class="section">
  <div class="section-title"><span class="icon">🔢</span> shape 읽는 법</div>
  <div class="highlight-box">
    <strong>shape = (행, 열)</strong><br><br>
    <code>shape (2, 3)</code> → 2행 3열 행렬<br>
    <code>shape (3,)</code> → 원소 3개의 1차원 배열 (벡터)<br>
    <code>shape ()</code> → 스칼라 (0차원)
  </div>
  <pre class="code-pre"><code class="language-python"># 형상(shape) 예시
a = np.array(3.14)           # shape: ()   0차원 스칼라
b = np.array([1, 2, 3])      # shape: (3,) 1차원 벡터
c = np.zeros((4, 5))         # shape: (4, 5) 2차원 행렬
d = np.ones((2, 3, 4))       # shape: (2, 3, 4) 3차원 텐서

print(a.ndim, a.shape)  # 0 ()
print(b.ndim, b.shape)  # 1 (3,)
print(c.ndim, c.shape)  # 2 (4, 5)
print(d.ndim, d.shape)  # 3 (2, 3, 4)</code></pre>
</div>

<div class="section">
  <div class="section-title"><span class="icon">🔄</span> 형상 변환: reshape와 전치</div>
  <pre class="code-pre"><code class="language-python">A = np.array([[1, 2, 3], [4, 5, 6]])  # shape (2, 3)

# 전치 (행 열 교환)
print(A.T)         # shape (3, 2)

# reshape (원소 수 동일하게 유지)
B = A.reshape(3, 2)   # shape (3, 2)
C = A.reshape(6)      # shape (6,) -- 1D로 flatten

# 열벡터와 행벡터 변환
x = np.array([1, 2, 3])   # shape (3,)
x_col = x.reshape(-1, 1)  # shape (3, 1) 열벡터
x_row = x.reshape(1, -1)  # shape (1, 3) 행벡터</code></pre>
</div>
`;


/* ─────────────────────── 3.1.3 ─────────────────────── */
CONTENT['3.1.3'] = () => String.raw`
<div class="page-title">원소별 연산</div>
<div class="page-subtitle">3.1 넘파이와 다차원 배열</div>

<div class="section">
  <div class="section-title"><span class="icon">✖</span> 원소별(Element-wise) 연산</div>
  <p>형상이 같은 두 배열에 대해 <strong>같은 위치의 원소끼리</strong> 연산을 수행합니다. 덧셈(<code>+</code>), 뺄셈(<code>-</code>), 곱셈(<code>*</code>)이 모두 원소별로 작동합니다.</p>
  <pre class="code-pre"><code class="language-python">import numpy as np

W = np.array([[1, 2, 3], [4, 5, 6]])
X = np.array([[0, 1, 2], [3, 4, 5]])

print(W + X)
# [[ 1,  3,  5],
#  [ 7,  9, 11]]

print(W * X)   # 야다마르 곱 (Hadamard product)
# [[ 0,  2,  6],
#  [12, 20, 30]]</code></pre>
  <div class="highlight-box">
    <strong>야다마르 곱 (Hadamard Product)</strong>: 원소별 곱셈. 기호 $\odot$ 또는 $\circ$로 표기.<br>
    수식: $(W \odot X)_{ij} = W_{ij} \cdot X_{ij}$<br>
    <em>단, 행렬 곱($WX$)과는 다른 연산임에 주의!</em>
  </div>
</div>

<div class="section">
  <div class="section-title"><span class="icon">📡</span> 브로드캐스팅 (Broadcasting)</div>
  <p>형상이 다른 배열끼리 연산할 때 넘파이는 자동으로 형상을 맞춰줍니다.</p>
  <pre class="code-pre"><code class="language-python">A = np.array([[1, 2, 3],
              [4, 5, 6]])  # shape (2, 3)

# 스칼라 브로드캐스팅
print(A * 2)
# [[ 2,  4,  6],
#  [ 8, 10, 12]]

# 벡터 브로드캐스팅 (각 행에 동일하게 적용)
b = np.array([10, 20, 30])  # shape (3,)
print(A + b)
# [[11, 22, 33],
#  [14, 25, 36]]

# 평균 정규화 예시 (데이터 전처리)
data = np.array([[170, 62], [165, 55], [175, 68]])  # shape (3, 2)
mean = data.mean(axis=0)   # shape (2,) -- 각 열의 평균
normalized = data - mean   # 브로드캐스팅으로 각 행에서 평균 빼기</code></pre>
</div>
`;


/* ─────────────────────── 3.1.4 ─────────────────────── */
CONTENT['3.1.4'] = () => String.raw`
<div class="page-title">벡터의 내적과 행렬 곱</div>
<div class="page-subtitle">3.1 넘파이와 다차원 배열</div>

<div class="section">
  <div class="section-title"><span class="icon">∙</span> 벡터의 내적 (Inner Product / Dot Product)</div>
  <p>원소 수가 $D$인 두 벡터 $x, y$의 내적은 <strong>같은 위치 원소들의 곱을 모두 더한 값</strong>입니다:</p>
  <div class="math-block">$$x \cdot y = x^\top y = x_1 y_1 + x_2 y_2 + \cdots + x_D y_D = \sum_{d=1}^D x_d y_d$$</div>
  <p>내적 결과는 <strong>스칼라</strong>입니다. 기하학적으로는 두 벡터의 유사성(코사인 유사도)과 관련됩니다.</p>
  <pre class="code-pre"><code class="language-python">import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
y = np.dot(a, b)   # 또는 a @ b
print(y)  # 32  (= 1*4 + 2*5 + 3*6)</code></pre>
</div>

<div class="section">
  <div class="section-title"><span class="icon">🔢</span> 행렬 곱 (Matrix Multiplication)</div>
  <p>$m \times n$ 행렬 $A$와 $n \times l$ 행렬 $B$의 곱 $C = AB$는 $m \times l$ 행렬입니다:</p>
  <div class="math-block">$$C_{ij} = \sum_{k=1}^n A_{ik} B_{kj}$$</div>
  <div class="highlight-box amber">
    <strong>차원 규칙:</strong> 앞 행렬의 <em>열 수</em> = 뒤 행렬의 <em>행 수</em>이어야 합니다.<br>
    $(m \times \mathbf{n}) \times (\mathbf{n} \times l) = m \times l$
  </div>
  <pre class="code-pre"><code class="language-python">A = np.array([[1, 2], [3, 4]])  # shape (2, 2)
B = np.array([[5, 6], [7, 8]])  # shape (2, 2)
Y = np.dot(A, B)   # 또는 A @ B
print(Y)
# [[19, 22],
#  [43, 50]]
# Y[0,0] = 1*5 + 2*7 = 19

# 직사각형 행렬
A = np.array([[1, 2, 3], [4, 5, 6]])   # shape (2, 3)
B = np.array([[1, 0], [0, 1], [1, 1]]) # shape (3, 2)
C = A @ B  # shape (2, 2)
print(C.shape)  # (2, 2)</code></pre>
</div>

<div class="section">
  <div class="section-title"><span class="icon">🎮</span> 내적 계산기</div>
  <p>슬라이더로 두 벡터의 원소를 조정하면 내적이 실시간으로 계산됩니다.</p>
  <div class="interactive-panel">
    <div class="panel-header">∙ 벡터 내적 인터랙티브 계산기</div>
    <div class="panel-body" style="flex-direction:column;gap:14px;">
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
        <div>
          <div class="ctrl-label" style="margin-bottom:8px;">벡터 <strong>a</strong> = [a₁, a₂, a₃]</div>
          <div class="ctrl-group"><div class="ctrl-label">a₁</div><div class="ctrl-row"><input type="range" id="c314-a1" min="-3" max="3" step="0.5" value="1" oninput="ch3_314_update()"><span class="val-badge" id="c314-a1-val">1.0</span></div></div>
          <div class="ctrl-group"><div class="ctrl-label">a₂</div><div class="ctrl-row"><input type="range" id="c314-a2" min="-3" max="3" step="0.5" value="2" oninput="ch3_314_update()"><span class="val-badge" id="c314-a2-val">2.0</span></div></div>
          <div class="ctrl-group"><div class="ctrl-label">a₃</div><div class="ctrl-row"><input type="range" id="c314-a3" min="-3" max="3" step="0.5" value="3" oninput="ch3_314_update()"><span class="val-badge" id="c314-a3-val">3.0</span></div></div>
        </div>
        <div>
          <div class="ctrl-label" style="margin-bottom:8px;">벡터 <strong>b</strong> = [b₁, b₂, b₃]</div>
          <div class="ctrl-group"><div class="ctrl-label">b₁</div><div class="ctrl-row"><input type="range" id="c314-b1" min="-3" max="3" step="0.5" value="4" oninput="ch3_314_update()"><span class="val-badge" id="c314-b1-val">4.0</span></div></div>
          <div class="ctrl-group"><div class="ctrl-label">b₂</div><div class="ctrl-row"><input type="range" id="c314-b2" min="-3" max="3" step="0.5" value="5" oninput="ch3_314_update()"><span class="val-badge" id="c314-b2-val">5.0</span></div></div>
          <div class="ctrl-group"><div class="ctrl-label">b₃</div><div class="ctrl-row"><input type="range" id="c314-b3" min="-3" max="3" step="0.5" value="6" oninput="ch3_314_update()"><span class="val-badge" id="c314-b3-val">6.0</span></div></div>
        </div>
      </div>
      <canvas id="c314-chart" height="180"></canvas>
      <div class="stat-grid" style="grid-template-columns:repeat(4,1fr);">
        <div class="stat-card"><div class="label">a₁×b₁</div><div class="value" id="c314-p1" style="font-size:1rem;">4</div></div>
        <div class="stat-card"><div class="label">a₂×b₂</div><div class="value" id="c314-p2" style="font-size:1rem;">10</div></div>
        <div class="stat-card"><div class="label">a₃×b₃</div><div class="value" id="c314-p3" style="font-size:1rem;">18</div></div>
        <div class="stat-card" style="border:2px solid #3b82f6;"><div class="label">내적 a·b</div><div class="value" id="c314-dot" style="color:#1d4ed8;">32</div></div>
      </div>
    </div>
  </div>
</div>
`;

CHART_INITS['3.1.4'] = function () {
  const ctx = document.getElementById('c314-chart').getContext('2d');
  const chart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['a₁×b₁', 'a₂×b₂', 'a₃×b₃'],
      datasets: [{ label: '원소별 곱', data: [4, 10, 18], backgroundColor: ['rgba(59,130,246,0.7)', 'rgba(99,102,241,0.7)', 'rgba(139,92,246,0.7)'] }]
    },
    options: {
      responsive: true, animation: { duration: 100 },
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      scales: { y: { title: { display: true, text: '원소별 곱 값', font: { size: 11 } } } }
    }
  });
  activeChartInstances['c314'] = chart;

  window.ch3_314_update = function () {
    const a1 = parseFloat(document.getElementById('c314-a1').value);
    const a2 = parseFloat(document.getElementById('c314-a2').value);
    const a3 = parseFloat(document.getElementById('c314-a3').value);
    const b1 = parseFloat(document.getElementById('c314-b1').value);
    const b2 = parseFloat(document.getElementById('c314-b2').value);
    const b3 = parseFloat(document.getElementById('c314-b3').value);
    ['a1','a2','a3','b1','b2','b3'].forEach(k => {
      document.getElementById(`c314-${k}-val`).textContent = parseFloat(document.getElementById(`c314-${k}`).value).toFixed(1);
    });
    const p1 = a1*b1, p2 = a2*b2, p3 = a3*b3;
    const dot = p1 + p2 + p3;
    document.getElementById('c314-p1').textContent = p1.toFixed(1);
    document.getElementById('c314-p2').textContent = p2.toFixed(1);
    document.getElementById('c314-p3').textContent = p3.toFixed(1);
    document.getElementById('c314-dot').textContent = dot.toFixed(1);
    chart.data.datasets[0].data = [p1, p2, p3];
    chart.update();
  };
};


/* ─────────────────────── 3.2.1 ─────────────────────── */
CONTENT['3.2.1'] = () => String.raw`
<div class="page-title">다변량 정규 분포 공식</div>
<div class="page-subtitle">3.2 다변량 정규 분포</div>

<div class="section">
  <div class="section-title"><span class="icon">📐</span> 다변량 정규 분포 정의 [식 3.1]</div>
  <p>스칼라 $x$에 대한 1변량 정규 분포를 벡터 $x \in \mathbb{R}^D$로 일반화한 것이 <strong>다변량 정규 분포</strong>입니다.
    <a href="#" class="ch-link" onclick="(function(){var s=FLAT_SECTIONS.find(function(x){return x.id==='1.2.1';});if(s)loadSection(s.id,s.ch,s.sub,s.sec);})();return false;">→ Ch.1.2.1 정규 분포 PDF</a>
  </p>
  <div class="math-block">$$\mathcal{N}(x;\,\mu,\,\Sigma) = \frac{1}{\sqrt{(2\pi)^D \lvert\Sigma\rvert}} \exp\!\left(-\frac{1}{2}(x-\mu)^\top \Sigma^{-1} (x-\mu)\right) \qquad \text{[식 3.1]}$$</div>
  <div style="overflow-x:auto;margin:16px 0;">
    <table style="border-collapse:collapse;width:100%;font-size:0.87rem;">
      <thead><tr style="background:#1e3a5f;color:#bfdbfe;"><th style="padding:9px 14px;">기호</th><th style="padding:9px 14px;">이름</th><th style="padding:9px 14px;">설명</th></tr></thead>
      <tbody>
        <tr style="background:#f8fafc;"><td style="padding:8px 14px;border:1px solid #e2e8f0;font-weight:700;">$x$</td><td style="padding:8px 14px;border:1px solid #e2e8f0;">확률 변수 벡터</td><td style="padding:8px 14px;border:1px solid #e2e8f0;">$D$차원 벡터</td></tr>
        <tr><td style="padding:8px 14px;border:1px solid #e2e8f0;font-weight:700;">$\mu$</td><td style="padding:8px 14px;border:1px solid #e2e8f0;">평균 벡터</td><td style="padding:8px 14px;border:1px solid #e2e8f0;">$D$차원 벡터, 분포의 중심</td></tr>
        <tr style="background:#f8fafc;"><td style="padding:8px 14px;border:1px solid #e2e8f0;font-weight:700;">$\Sigma$</td><td style="padding:8px 14px;border:1px solid #e2e8f0;">공분산 행렬</td><td style="padding:8px 14px;border:1px solid #e2e8f0;">$D \times D$ 양의 정부호 대칭 행렬</td></tr>
        <tr><td style="padding:8px 14px;border:1px solid #e2e8f0;font-weight:700;">$\lvert\Sigma\rvert$</td><td style="padding:8px 14px;border:1px solid #e2e8f0;">행렬식</td><td style="padding:8px 14px;border:1px solid #e2e8f0;">determinant, 분포의 "부피" 척도</td></tr>
        <tr style="background:#f8fafc;"><td style="padding:8px 14px;border:1px solid #e2e8f0;font-weight:700;">$\Sigma^{-1}$</td><td style="padding:8px 14px;border:1px solid #e2e8f0;">역행렬 (정밀도 행렬)</td><td style="padding:8px 14px;border:1px solid #e2e8f0;">precision matrix라고도 함</td></tr>
        <tr><td style="padding:8px 14px;border:1px solid #e2e8f0;font-weight:700;">$(x-\mu)^\top \Sigma^{-1}(x-\mu)$</td><td style="padding:8px 14px;border:1px solid #e2e8f0;">마할라노비스 거리²</td><td style="padding:8px 14px;border:1px solid #e2e8f0;">공분산 보정 후 중심까지의 거리</td></tr>
      </tbody>
    </table>
  </div>
</div>

<div class="section">
  <div class="section-title"><span class="icon">📊</span> 공분산 행렬과 상관계수</div>
  <p>$D=2$의 경우, 상관계수 $\rho$를 이용한 표현:</p>
  <div class="math-block">$$\Sigma = \begin{pmatrix}\sigma_1^2 & \rho\sigma_1\sigma_2 \\ \rho\sigma_1\sigma_2 & \sigma_2^2\end{pmatrix}, \qquad -1 \le \rho \le 1$$</div>
  <div class="highlight-box">
    $\rho > 0$: <strong>양의 상관</strong> — 한 변수 증가 시 다른 변수도 증가<br>
    $\rho = 0$: <strong>무상관</strong> — 두 변수가 독립 (타원이 축에 평행)<br>
    $\rho < 0$: <strong>음의 상관</strong> — 한 변수 증가 시 다른 변수 감소
  </div>
</div>

<div class="section">
  <div class="section-title"><span class="icon">🔵</span> 공분산 행렬에 따른 분포 형태</div>
  <p>아래 버튼을 눌러 다양한 공분산 행렬이 분포 형태에 어떤 영향을 주는지 확인하세요.</p>
  <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px;">
    <button class="btn btn-primary" onclick="ch3_321_set(1,0,1)">① 단위행렬<br><small>원형</small></button>
    <button class="btn btn-secondary" onclick="ch3_321_set(4,0,1)">② σ₁ 큰 경우<br><small>수평 타원</small></button>
    <button class="btn btn-secondary" onclick="ch3_321_set(1,0,4)">③ σ₂ 큰 경우<br><small>수직 타원</small></button>
    <button class="btn btn-secondary" onclick="ch3_321_set(2,1.6,2)">④ 양의 상관<br><small>우상향 기울기</small></button>
    <button class="btn btn-secondary" onclick="ch3_321_set(2,-1.6,2)">⑤ 음의 상관<br><small>우하향 기울기</small></button>
  </div>
  <canvas id="c321-chart" height="280"></canvas>
  <div class="stat-grid" style="grid-template-columns:repeat(4,1fr);margin-top:12px;">
    <div class="stat-card"><div class="label">σ₁₁ (분산 x₁)</div><div class="value" id="c321-s11" style="font-size:1rem;">1.00</div></div>
    <div class="stat-card"><div class="label">σ₁₂ (공분산)</div><div class="value" id="c321-s12" style="font-size:1rem;">0.00</div></div>
    <div class="stat-card"><div class="label">σ₂₂ (분산 x₂)</div><div class="value" id="c321-s22" style="font-size:1rem;">1.00</div></div>
    <div class="stat-card"><div class="label">|Σ| (행렬식)</div><div class="value" id="c321-det" style="font-size:1rem;">1.00</div></div>
  </div>
</div>
`;

CHART_INITS['3.2.1'] = function () {
  let curS11 = 1, curS12 = 0, curS22 = 1;

  function makeDS(s11, s12, s22) {
    return [
      { label: '1σ', type: 'line', data: ch3.ellipse(0,0,s11,s12,s22,1), borderColor: '#ef4444', borderWidth: 2.5, pointRadius: 0, fill: false, tension: 0 },
      { label: '2σ', type: 'line', data: ch3.ellipse(0,0,s11,s12,s22,2), borderColor: '#f97316', borderWidth: 2,   pointRadius: 0, fill: false, tension: 0 },
      { label: '3σ', type: 'line', data: ch3.ellipse(0,0,s11,s12,s22,3), borderColor: '#eab308', borderWidth: 1.5, pointRadius: 0, fill: false, tension: 0 }
    ];
  }

  const ctx = document.getElementById('c321-chart').getContext('2d');
  const chart = new Chart(ctx, {
    type: 'scatter',
    data: { datasets: makeDS(1, 0, 1) },
    options: {
      responsive: true, animation: { duration: 200 },
      plugins: { legend: { labels: { font: { size: 11 } } }, tooltip: { enabled: false } },
      scales: {
        x: { type: 'linear', min: -7, max: 7, title: { display: true, text: 'x₁', font: { size: 11 } } },
        y: { min: -7, max: 7, title: { display: true, text: 'x₂', font: { size: 11 } } }
      }
    }
  });
  activeChartInstances['c321'] = chart;

  window.ch3_321_set = function (s11, s12, s22) {
    curS11 = s11; curS12 = s12; curS22 = s22;
    chart.data.datasets = makeDS(s11, s12, s22);
    chart.update();
    const det = s11 * s22 - s12 * s12;
    document.getElementById('c321-s11').textContent = s11.toFixed(2);
    document.getElementById('c321-s12').textContent = s12.toFixed(2);
    document.getElementById('c321-s22').textContent = s22.toFixed(2);
    document.getElementById('c321-det').textContent = det.toFixed(3);
  };
};


/* ─────────────────────── 3.2.2 ─────────────────────── */
CONTENT['3.2.2'] = () => String.raw`
<div class="page-title">다변량 정규 분포 구현</div>
<div class="page-subtitle">3.2 다변량 정규 분포</div>

<div class="section">
  <div class="section-title"><span class="icon">🐍</span> 파이썬으로 구현하기</div>
  <p>[식 3.1]의 다변량 정규 분포를 넘파이로 직접 구현합니다.</p>
  <pre class="code-pre"><code class="language-python">import numpy as np

def multivariate_normal(x, mu, cov):
    """
    다변량 정규 분포 PDF 계산
    x:   (D,)   입력 벡터
    mu:  (D,)   평균 벡터
    cov: (D, D) 공분산 행렬
    """
    det = np.linalg.det(cov)      # 행렬식 |Sigma|
    inv = np.linalg.inv(cov)      # 역행렬 Sigma^{-1}
    D = len(x)

    # 정규화 상수
    z = 1 / np.sqrt((2 * np.pi) ** D * det)

    # 마할라노비스 거리^2: (x-mu)^T Sigma^{-1} (x-mu)
    diff = x - mu
    mah2 = diff.T @ inv @ diff

    return z * np.exp(-mah2 / 2.0)</code></pre>
</div>

<div class="section">
  <div class="section-title"><span class="icon">🧪</span> 사용 예시</div>
  <pre class="code-pre"><code class="language-python"># 2차원 다변량 정규분포 (키, 몸무게)
mu = np.array([170.7, 62.0])       # 평균 벡터 (D=2)
cov = np.array([[30.25, 16.5],     # 공분산 행렬
                [16.5,  64.0]])    # rho 약 0.375

# 특정 점에서의 확률 밀도 계산
x = np.array([170.0, 60.0])
p = multivariate_normal(x, mu, cov)
print(f"p = {p:.6f}")

# 역행렬 검증: Sigma @ Sigma^{-1} = I
inv_cov = np.linalg.inv(cov)
print(np.allclose(cov @ inv_cov, np.eye(2)))  # True

# scipy로 더 간단하게
from scipy.stats import multivariate_normal as mvn
p2 = mvn.pdf(x, mean=mu, cov=cov)
print(np.isclose(p, p2))  # True</code></pre>
  <div class="highlight-box green">
    <strong>핵심 연산:</strong> 넘파이의 <code>np.linalg.det()</code>(행렬식)과 <code>np.linalg.inv()</code>(역행렬)를 사용합니다. 행렬 곱은 <code>@</code> 연산자로 표현합니다.
  </div>
</div>
`;


/* ─────────────────────── 3.3.1 ─────────────────────── */
CONTENT['3.3.1'] = () => String.raw`
<div class="page-title">3D 그래프 그리기</div>
<div class="page-subtitle">3.3 2차원 정규 분포 시각화</div>

<div class="section">
  <div class="section-title"><span class="icon">🌄</span> 3D 곡면 시각화</div>
  <p>2차원 정규 분포 $\mathcal{N}(x;\mu,\Sigma)$는 <strong>3D 곡면</strong>으로 시각화할 수 있습니다. $(x_1, x_2)$ 평면의 각 점에 대한 확률 밀도값 $p$를 높이(z축)로 표현합니다.</p>
  <pre class="code-pre"><code class="language-python">import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def multivariate_normal(x, mu, cov):
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    D = len(x)
    z = 1 / np.sqrt((2 * np.pi) ** D * det)
    return z * np.exp((x - mu).T @ inv @ (x - mu) / -2.0)

mu  = np.array([0.5, -0.2])
cov = np.array([[2.0, 0.3], [0.3, 0.5]])

xs = ys = np.arange(-5, 5, 0.1)
X, Y = np.meshgrid(xs, ys)
Z = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x = np.array([X[i, j], Y[i, j]])
        Z[i, j] = multivariate_normal(x, mu, cov)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
plt.show()</code></pre>
</div>

<div class="section">
  <div class="section-title"><span class="icon">🎨</span> 밀도 히트맵 (브라우저 시각화)</div>
  <p>각 격자점의 색상이 확률 밀도값을 나타냅니다. 파란색(낮음) → 초록색 → 노란색 → 빨간색(높음).</p>
  <div class="interactive-panel">
    <div class="panel-header">🌈 2D 정규분포 밀도 시각화 — μ=[0,0], Σ=I</div>
    <div class="panel-body" style="flex-direction:column;">
      <canvas id="c331-chart" height="300"></canvas>
      <p style="font-size:0.78rem;color:#64748b;margin-top:8px;">각 점의 색상 = 해당 위치의 확률 밀도값. 3D 곡면을 2D로 투영한 형태입니다.</p>
    </div>
  </div>
</div>
`;

CHART_INITS['3.3.1'] = function () {
  const N = 30, range = 3.5;
  const pts = [], colors = [], denses = [];
  let maxD = 0;

  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const x1 = -range + 2 * range * i / (N - 1);
      const x2 = -range + 2 * range * j / (N - 1);
      const d = ch3.pdf2d(x1, x2, 0, 0, 1, 0, 1);
      pts.push({ x: x1, y: x2 });
      denses.push(d);
      if (d > maxD) maxD = d;
    }
  }
  denses.forEach(d => colors.push(ch3.densityColor(d / maxD)));

  const ctx = document.getElementById('c331-chart').getContext('2d');
  const chart = new Chart(ctx, {
    type: 'scatter',
    data: { datasets: [{ data: pts, backgroundColor: colors, pointRadius: 8, pointHoverRadius: 8 }] },
    options: {
      responsive: true, animation: { duration: 0 },
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      scales: {
        x: { type: 'linear', min: -range, max: range, title: { display: true, text: 'x₁', font: { size: 11 } } },
        y: { min: -range, max: range, title: { display: true, text: 'x₂', font: { size: 11 } } }
      }
    }
  });
  activeChartInstances['c331'] = chart;
};


/* ─────────────────────── 3.3.2 ─────────────────────── */
CONTENT['3.3.2'] = () => String.raw`
<div class="page-title">등고선 그리기</div>
<div class="page-subtitle">3.3 2차원 정규 분포 시각화</div>

<div class="section">
  <div class="section-title"><span class="icon">📍</span> 등고선(Contour)이란?</div>
  <p>3D 곡면을 위에서 내려다 본 것이 <strong>등고선 그림(contour plot)</strong>입니다. 같은 확률 밀도를 갖는 점들을 연결한 곡선이 등고선입니다.</p>
  <div class="highlight-box">
    <strong>다변량 정규분포의 등고선 = 타원(Ellipse)</strong><br>
    $(x-\mu)^\top \Sigma^{-1} (x-\mu) = c^2$를 만족하는 점들의 집합이 타원.<br>
    $c = 1, 2, 3$을 각각 <strong>1σ, 2σ, 3σ 타원</strong>이라 부릅니다.
  </div>
  <pre class="code-pre"><code class="language-python">import numpy as np
import matplotlib.pyplot as plt

mu  = np.array([0.5, -0.2])
cov = np.array([[2.0, 0.3], [0.3, 0.5]])

xs = ys = np.arange(-5, 5, 0.1)
X, Y = np.meshgrid(xs, ys)
Z = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x = np.array([X[i, j], Y[i, j]])
        Z[i, j] = multivariate_normal(x, mu, cov)

# 등고선 (contour) 그리기
plt.contour(X, Y, Z, levels=10)
plt.colorbar()
plt.xlabel('x₁'), plt.ylabel('x₂')
plt.show()</code></pre>
</div>

<div class="section">
  <div class="section-title"><span class="icon">🔵</span> 1σ / 2σ / 3σ 타원의 확률</div>
  <div class="highlight-box amber">
    1차원 정규분포와 달리, 2차원에서 각 타원이 포함하는 확률은 다릅니다:<br>
    <ul style="margin-top:8px;">
      <li><strong>1σ 타원 내부:</strong> 전체 데이터의 약 <strong>39.3%</strong></li>
      <li><strong>2σ 타원 내부:</strong> 전체 데이터의 약 <strong>86.5%</strong></li>
      <li><strong>3σ 타원 내부:</strong> 전체 데이터의 약 <strong>98.9%</strong></li>
    </ul>
    (1D에서 1σ 구간은 68.3%. 차원이 높아질수록 비율이 달라집니다.)
  </div>
</div>
`;


/* ─────────────────────── 3.3.3 ─────────────────────── */
CONTENT['3.3.3'] = () => String.raw`
<div class="page-title">2차원 정규 분포 그래프</div>
<div class="page-subtitle">3.3 2차원 정규 분포 시각화</div>

<div class="section">
  <div class="section-title"><span class="icon">🎮</span> 2차원 정규 분포 인터랙티브 탐색</div>
  <p>슬라이더로 공분산 행렬의 매개변수를 조정하여 분포의 형태 변화를 실시간으로 확인하세요.<br>
  빨간(1σ) · 주황(2σ) · 노란(3σ) 선이 등고 타원이며, 파란점은 해당 분포에서 생성된 샘플입니다.</p>
  <div class="interactive-panel">
    <div class="panel-header">🔵 2D 정규분포 시각화 — μ = [0, 0]</div>
    <div class="panel-body">
      <div class="chart-wrap"><canvas id="c333-chart" height="310"></canvas></div>
      <div class="controls-wrap">
        <div class="ctrl-group">
          <div class="ctrl-label">σ₁ (x₁ 표준편차)</div>
          <div class="ctrl-row">
            <input type="range" id="c333-s1" min="0.4" max="3.0" step="0.1" value="1.5" oninput="ch3_333_update()">
            <span class="val-badge" id="c333-s1-val">1.5</span>
          </div>
        </div>
        <div class="ctrl-group">
          <div class="ctrl-label">σ₂ (x₂ 표준편차)</div>
          <div class="ctrl-row">
            <input type="range" id="c333-s2" min="0.4" max="3.0" step="0.1" value="1.0" oninput="ch3_333_update()">
            <span class="val-badge" id="c333-s2-val">1.0</span>
          </div>
        </div>
        <div class="ctrl-group">
          <div class="ctrl-label">ρ (상관계수)</div>
          <div class="ctrl-row">
            <input type="range" id="c333-rho" min="-0.95" max="0.95" step="0.05" value="0.5" oninput="ch3_333_update()">
            <span class="val-badge" id="c333-rho-val">0.50</span>
          </div>
        </div>
        <button class="btn btn-primary" style="width:100%;margin-bottom:10px;" onclick="ch3_333_regen()">🎲 새 샘플 생성</button>
        <div style="font-size:0.75rem;color:#64748b;margin-bottom:6px;font-weight:700;">공분산 행렬 Σ</div>
        <div class="stat-grid" style="grid-template-columns:1fr 1fr;">
          <div class="stat-card"><div class="label">σ₁₁</div><div class="value" id="c333-v11" style="font-size:0.9rem;">2.25</div></div>
          <div class="stat-card"><div class="label">σ₁₂</div><div class="value" id="c333-v12" style="font-size:0.9rem;">0.75</div></div>
          <div class="stat-card"><div class="label">σ₂₁</div><div class="value" id="c333-v21" style="font-size:0.9rem;">0.75</div></div>
          <div class="stat-card"><div class="label">σ₂₂</div><div class="value" id="c333-v22" style="font-size:0.9rem;">1.00</div></div>
        </div>
        <div class="stat-card" style="margin-top:8px;border:2px solid #3b82f6;"><div class="label">|Σ| (행렬식)</div><div class="value" id="c333-det" style="color:#1d4ed8;">1.688</div></div>
      </div>
    </div>
  </div>
</div>

<div class="section">
  <div class="section-title"><span class="icon">💡</span> 관찰 포인트</div>
  <ul class="content-list">
    <li><strong>ρ → 0:</strong> 타원이 축에 평행 (x₁, x₂ 독립)</li>
    <li><strong>ρ → +1:</strong> 타원이 우상향 기울기 (양의 상관)</li>
    <li><strong>ρ → −1:</strong> 타원이 우하향 기울기 (음의 상관)</li>
    <li><strong>σ₁ 증가:</strong> 타원이 x₁ 방향으로 늘어남</li>
    <li><strong>σ₂ 증가:</strong> 타원이 x₂ 방향으로 늘어남</li>
    <li><strong>|Σ| 감소 → 분포가 좁아짐:</strong> 최대 밀도값이 높아짐</li>
  </ul>
</div>
`;

CHART_INITS['3.3.3'] = function () {
  let sigma1 = 1.5, sigma2 = 1.0, rho = 0.5;

  function getCov() {
    return { s11: sigma1 * sigma1, s12: rho * sigma1 * sigma2, s22: sigma2 * sigma2 };
  }

  let samples = ch3.gen2d(200, 0, 0, sigma1*sigma1, rho*sigma1*sigma2, sigma2*sigma2);

  const ctx = document.getElementById('c333-chart').getContext('2d');
  const chart = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        { label: '샘플 (200개)', data: samples.map(p => ({ x: p[0], y: p[1] })), backgroundColor: 'rgba(59,130,246,0.22)', pointRadius: 3, order: 5 },
        { label: '1σ 타원', type: 'line', data: [], borderColor: '#ef4444', borderWidth: 2.5, pointRadius: 0, fill: false, tension: 0, order: 1 },
        { label: '2σ 타원', type: 'line', data: [], borderColor: '#f97316', borderWidth: 2,   pointRadius: 0, fill: false, tension: 0, order: 2 },
        { label: '3σ 타원', type: 'line', data: [], borderColor: '#eab308', borderWidth: 1.5, pointRadius: 0, fill: false, tension: 0, order: 3 }
      ]
    },
    options: {
      responsive: true, animation: { duration: 80 },
      plugins: { legend: { labels: { font: { size: 11 } } }, tooltip: { enabled: false } },
      scales: {
        x: { type: 'linear', min: -7, max: 7, title: { display: true, text: 'x₁', font: { size: 11 } } },
        y: { min: -7, max: 7, title: { display: true, text: 'x₂', font: { size: 11 } } }
      }
    }
  });
  activeChartInstances['c333'] = chart;

  function updateChart() {
    const { s11, s12, s22 } = getCov();
    const det = s11 * s22 - s12 * s12;
    chart.data.datasets[0].data = samples.map(p => ({ x: p[0], y: p[1] }));
    chart.data.datasets[1].data = ch3.ellipse(0, 0, s11, s12, s22, 1);
    chart.data.datasets[2].data = ch3.ellipse(0, 0, s11, s12, s22, 2);
    chart.data.datasets[3].data = ch3.ellipse(0, 0, s11, s12, s22, 3);
    chart.update();
    document.getElementById('c333-v11').textContent = s11.toFixed(3);
    document.getElementById('c333-v12').textContent = s12.toFixed(3);
    document.getElementById('c333-v21').textContent = s12.toFixed(3);
    document.getElementById('c333-v22').textContent = s22.toFixed(3);
    document.getElementById('c333-det').textContent = det.toFixed(4);
  }

  updateChart();

  window.ch3_333_update = function () {
    sigma1 = parseFloat(document.getElementById('c333-s1').value);
    sigma2 = parseFloat(document.getElementById('c333-s2').value);
    rho    = parseFloat(document.getElementById('c333-rho').value);
    document.getElementById('c333-s1-val').textContent  = sigma1.toFixed(1);
    document.getElementById('c333-s2-val').textContent  = sigma2.toFixed(1);
    document.getElementById('c333-rho-val').textContent = rho.toFixed(2);
    updateChart();
  };

  window.ch3_333_regen = function () {
    const { s11, s12, s22 } = getCov();
    samples = ch3.gen2d(200, 0, 0, s11, s12, s22);
    updateChart();
  };
};


/* ─────────────────────── 3.4.1 ─────────────────────── */
CONTENT['3.4.1'] = () => String.raw`
<div class="page-title">최대 가능도 추정하기</div>
<div class="page-subtitle">3.4 다변량 정규 분포의 최대 가능도 추정(MLE)</div>

<div class="section">
  <div class="section-title"><span class="icon">🎯</span> 다변량 MLE 개요</div>
  <p>2장에서는 1차원 정규분포의 MLE(Maximum Likelihood Estimation)를 공부했습니다. 이제 $D$차원 다변량 정규분포로 확장합니다.
    <a href="#" class="ch-link" onclick="(function(){var s=FLAT_SECTIONS.find(function(x){return x.id==='2.3.3';});if(s)loadSection(s.id,s.ch,s.sub,s.sec);})();return false;">→ Ch.2.3.3 단변량 MLE 유도</a>
  </p>
  <div class="highlight-box">
    <strong>목표:</strong> 데이터셋 $\mathcal{D} = \{x^{(1)}, \ldots, x^{(N)}\}$ ($x^{(n)} \in \mathbb{R}^D$)이 주어졌을 때,<br>
    다변량 정규분포의 매개변수 $\mu$와 $\Sigma$를 데이터에 가장 잘 맞도록 추정한다.
  </div>
</div>

<div class="section">
  <div class="section-title"><span class="icon">📐</span> 가능도 함수</div>
  <p>$N$개 데이터가 i.i.d.로 생성되었다고 가정하면 가능도 함수는:</p>
  <div class="math-block">$$p(\mathcal{D};\mu,\Sigma) = \prod_{n=1}^N \mathcal{N}(x^{(n)};\mu,\Sigma)$$</div>
  <p>로그 가능도를 최대화하는 $\hat{\mu}$와 $\hat{\Sigma}$를 편미분으로 구합니다:</p>
  <div class="math-block">$$\frac{\partial L}{\partial \mu} = 0, \qquad \frac{\partial L}{\partial \Sigma} = 0$$</div>
</div>

<div class="section">
  <div class="section-title"><span class="icon">✅</span> MLE 해 (닫힌 형태)</div>
  <div class="math-block">$$\hat{\mu} = \frac{1}{N} \sum_{n=1}^N x^{(n)} \qquad \text{[식 3.3]}$$</div>
  <div class="math-block">$$\hat{\Sigma} = \frac{1}{N} \sum_{n=1}^N (x^{(n)}-\hat{\mu})(x^{(n)}-\hat{\mu})^\top \qquad \text{[식 3.4]}$$</div>
  <div class="highlight-box green">
    <strong>해석:</strong><br>
    $\hat{\mu}$: 모든 데이터 벡터의 <strong>원소별 평균</strong> (산술 평균)<br>
    $\hat{\Sigma}$: 평균 편차 벡터들의 <strong>외적(outer product)의 평균</strong>
  </div>
  <div class="highlight-box amber">
    <strong>주의:</strong> [식 3.4]는 $\frac{1}{N}$으로 나누는 최대 가능도 추정량(편향 추정량)입니다.<br>
    비편향 추정량은 $\frac{1}{N-1}$로 나눕니다 (Bessel 보정 / numpy의 기본값).
  </div>
</div>

<div class="section">
  <div class="section-title"><span class="icon">📏</span> 형상 분석: outer product $(D \times D)$</div>
  <div style="overflow-x:auto;">
    <table style="border-collapse:collapse;width:100%;font-size:0.87rem;">
      <thead><tr style="background:#1e3a5f;color:#bfdbfe;"><th style="padding:9px 14px;">항</th><th style="padding:9px 14px;">형상</th><th style="padding:9px 14px;">설명</th></tr></thead>
      <tbody>
        <tr style="background:#f8fafc;"><td style="padding:8px 14px;border:1px solid #e2e8f0;">$x^{(n)}$</td><td style="padding:8px 14px;border:1px solid #e2e8f0;">$(D, 1)$</td><td style="padding:8px 14px;border:1px solid #e2e8f0;">각 데이터 벡터</td></tr>
        <tr><td style="padding:8px 14px;border:1px solid #e2e8f0;">$x^{(n)} - \hat{\mu}$</td><td style="padding:8px 14px;border:1px solid #e2e8f0;">$(D, 1)$</td><td style="padding:8px 14px;border:1px solid #e2e8f0;">평균 편차 벡터</td></tr>
        <tr style="background:#f8fafc;"><td style="padding:8px 14px;border:1px solid #e2e8f0;">$(x^{(n)}-\hat{\mu})(x^{(n)}-\hat{\mu})^\top$</td><td style="padding:8px 14px;border:1px solid #e2e8f0;">$(D, D)$</td><td style="padding:8px 14px;border:1px solid #e2e8f0;">외적 (outer product)</td></tr>
        <tr><td style="padding:8px 14px;border:1px solid #e2e8f0;">$\hat{\Sigma}$</td><td style="padding:8px 14px;border:1px solid #e2e8f0;">$(D, D)$</td><td style="padding:8px 14px;border:1px solid #e2e8f0;">추정 공분산 행렬</td></tr>
      </tbody>
    </table>
  </div>
</div>
`;


/* ─────────────────────── 3.4.2 ─────────────────────── */
CONTENT['3.4.2'] = () => String.raw`
<div class="page-title">최대 가능도 추정 구현</div>
<div class="page-subtitle">3.4 다변량 정규 분포의 최대 가능도 추정(MLE)</div>

<div class="section">
  <div class="section-title"><span class="icon">🐍</span> MLE 구현</div>
  <p>[식 3.3]과 [식 3.4]를 넘파이로 구현합니다.</p>
  <pre class="code-pre"><code class="language-python">import numpy as np

# 데이터: N x D 행렬 (N개 D차원 데이터)
xs = np.loadtxt('height_weight.txt')
# xs.shape = (N, 2)  키(cm), 몸무게(kg)

# [식 3.3] 평균 벡터 MLE
# axis=0: 각 열(특성)별로 평균
mu_hat = np.mean(xs, axis=0)   # shape: (2,)
print("평균 벡터 mu_hat:", mu_hat)

# [식 3.4] 공분산 행렬 MLE (1/N 버전)
N = len(xs)
diff = xs - mu_hat             # shape: (N, 2) 브로드캐스팅
cov_hat = (diff.T @ diff) / N  # shape: (2, 2)
print("공분산 행렬 Sigma_hat:\n", cov_hat)

# np.cov로 동일하게 계산 (bias=True -> 1/N)
cov_hat2 = np.cov(xs, rowvar=False, bias=True)
print("검증:", np.allclose(cov_hat, cov_hat2))  # True</code></pre>
</div>

<div class="section">
  <div class="section-title"><span class="icon">🔍</span> 추정 결과 해석</div>
  <pre class="code-pre"><code class="language-python">mu_hat  = np.mean(xs, axis=0)
cov_hat = np.cov(xs, rowvar=False, bias=True)

print(f"평균 키:       {mu_hat[0]:.2f} cm")
print(f"평균 몸무게:   {mu_hat[1]:.2f} kg")

print(f"\n공분산 행렬:")
print(f"  Var[키]:          {cov_hat[0,0]:.3f}  (sigma_키 = {np.sqrt(cov_hat[0,0]):.2f} cm)")
print(f"  Cov[키,몸무게]:   {cov_hat[0,1]:.3f}")
print(f"  Var[몸무게]:      {cov_hat[1,1]:.3f}  (sigma_몸무게 = {np.sqrt(cov_hat[1,1]):.2f} kg)")

# 상관계수 계산
rho = cov_hat[0,1] / (np.sqrt(cov_hat[0,0]) * np.sqrt(cov_hat[1,1]))
print(f"\n추정 상관계수 rho_hat = {rho:.3f}")</code></pre>
  <div class="highlight-box green">
    <strong>핵심:</strong> 어떤 다차원 데이터도 정규 분포만 가정하면 MLE로 완전히 모델링할 수 있습니다.
    추정된 $\hat{\mu}$와 $\hat{\Sigma}$를 [식 3.1]에 대입하면 완성된 다변량 생성 모델이 됩니다.
  </div>
</div>
`;


/* ─────────────────────── 3.4.3 ─────────────────────── */
CONTENT['3.4.3'] = () => String.raw`
<div class="page-title">실제 데이터 사용</div>
<div class="page-subtitle">3.4 다변량 정규 분포의 최대 가능도 추정(MLE)</div>

<div class="section">
  <div class="section-title"><span class="icon">📊</span> 키-몸무게 데이터 MLE 피팅 데모</div>
  <p>키($x_1$, cm)와 몸무게($x_2$, kg) 데이터에 다변량 정규 분포를 피팅합니다. 샘플 수 슬라이더를 조정하며 추정량이 수렴하는 과정을 확인하세요.</p>
  <div class="interactive-panel">
    <div class="panel-header">🧑‍🔬 키-몸무게 MLE 피팅 — N(μ̂, Σ̂) 추정</div>
    <div class="panel-body">
      <div class="chart-wrap"><canvas id="c343-chart" height="310"></canvas></div>
      <div class="controls-wrap">
        <div class="ctrl-group">
          <div class="ctrl-label">샘플 수 N</div>
          <div class="ctrl-row">
            <input type="range" id="c343-n" min="20" max="500" step="10" value="200" oninput="ch3_343_update()">
            <span class="val-badge" id="c343-n-val">200</span>
          </div>
        </div>
        <button class="btn btn-primary" style="width:100%;margin-bottom:12px;" onclick="ch3_343_regen()">🎲 새 샘플 생성</button>
        <div style="font-size:0.75rem;color:#64748b;margin-bottom:6px;font-weight:700;">MLE 추정 결과 (→ 진짜값)</div>
        <div class="stat-grid" style="grid-template-columns:1fr;">
          <div class="stat-card"><div class="label">μ̂₁ 평균 키 → 170.7 cm</div><div class="value" id="c343-mu1" style="font-size:1rem;">—</div></div>
          <div class="stat-card"><div class="label">μ̂₂ 평균 몸무게 → 62.0 kg</div><div class="value" id="c343-mu2" style="font-size:1rem;">—</div></div>
          <div class="stat-card"><div class="label">σ̂₁ 키 표준편차 → 5.5 cm</div><div class="value" id="c343-sd1" style="font-size:1rem;">—</div></div>
          <div class="stat-card"><div class="label">σ̂₂ 몸무게 표준편차 → 8.0 kg</div><div class="value" id="c343-sd2" style="font-size:1rem;">—</div></div>
          <div class="stat-card" style="border:2px solid #3b82f6;"><div class="label">ρ̂ 상관계수 → 0.375</div><div class="value" id="c343-rho" style="font-size:1rem;color:#1d4ed8;">—</div></div>
        </div>
      </div>
    </div>
  </div>
</div>

<div class="section">
  <div class="section-title"><span class="icon">💡</span> 관찰 포인트</div>
  <ul class="content-list">
    <li><strong>N이 작을 때:</strong> 추정량이 실제 값(170.7cm, 62kg)에서 벗어남</li>
    <li><strong>N이 커질수록:</strong> MLE 추정량이 실제 분포 매개변수에 수렴 (일치성)</li>
    <li><strong>피팅 타원:</strong> 빨간선(1σ)이 약 39%, 주황선(2σ)이 약 87%의 데이터를 포함</li>
    <li><strong>다변량 정규분포 = 완전한 생성 모델:</strong> $\hat{\mu}$와 $\hat{\Sigma}$로 새로운 [키, 몸무게] 쌍 생성 가능</li>
  </ul>
  <div class="highlight-box">
    <strong>결론:</strong> 다변량 정규분포는 어떤 다차원 데이터도 평균 벡터와 공분산 행렬만으로 완전히 기술할 수 있는 강력한 생성 모델입니다.
    <a href="#" class="ch-link" onclick="(function(){var s=FLAT_SECTIONS.find(function(x){return x.id==='2.4.1';});if(s)loadSection(s.id,s.ch,s.sub,s.sec);})();return false;">→ Ch.2.4.1 생성 모델로 새 데이터 생성</a>
  </div>
</div>
`;

CHART_INITS['3.4.3'] = function () {
  // 실제 모집단 매개변수 (height, weight)
  const TRUE_MU1 = 170.7, TRUE_MU2 = 62.0;
  const TRUE_S11 = 30.25, TRUE_S12 = 16.5, TRUE_S22 = 64.0; // rho ≈ 0.375

  let N = 200;
  let data = ch3.gen2d(N, TRUE_MU1, TRUE_MU2, TRUE_S11, TRUE_S12, TRUE_S22);

  const ctx = document.getElementById('c343-chart').getContext('2d');
  const chart = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        { label: '샘플 데이터', data: data.map(p => ({ x: p[0], y: p[1] })), backgroundColor: 'rgba(59,130,246,0.22)', pointRadius: 3, order: 5 },
        { label: '1σ 타원 (MLE 피팅)', type: 'line', data: [], borderColor: '#ef4444', borderWidth: 2.5, pointRadius: 0, fill: false, tension: 0, order: 1 },
        { label: '2σ 타원 (MLE 피팅)', type: 'line', data: [], borderColor: '#f97316', borderWidth: 2,   pointRadius: 0, fill: false, tension: 0, order: 2 }
      ]
    },
    options: {
      responsive: true, animation: { duration: 100 },
      plugins: { legend: { labels: { font: { size: 11 } } }, tooltip: { enabled: false } },
      scales: {
        x: { type: 'linear', min: 148, max: 198, title: { display: true, text: '키 (cm)', font: { size: 11 } } },
        y: { min: 30, max: 98, title: { display: true, text: '몸무게 (kg)', font: { size: 11 } } }
      }
    }
  });
  activeChartInstances['c343'] = chart;

  function refit() {
    const est = ch3.mle2d(data);
    chart.data.datasets[0].data = data.map(p => ({ x: p[0], y: p[1] }));
    chart.data.datasets[1].data = ch3.ellipse(est.mu1, est.mu2, est.s11, est.s12, est.s22, 1);
    chart.data.datasets[2].data = ch3.ellipse(est.mu1, est.mu2, est.s11, est.s12, est.s22, 2);
    chart.update();
    const rho = est.s12 / Math.sqrt(est.s11 * est.s22);
    document.getElementById('c343-mu1').textContent = est.mu1.toFixed(1) + ' cm';
    document.getElementById('c343-mu2').textContent = est.mu2.toFixed(1) + ' kg';
    document.getElementById('c343-sd1').textContent = Math.sqrt(est.s11).toFixed(2) + ' cm';
    document.getElementById('c343-sd2').textContent = Math.sqrt(est.s22).toFixed(2) + ' kg';
    document.getElementById('c343-rho').textContent = rho.toFixed(3);
  }

  refit();

  window.ch3_343_update = function () {
    N = parseInt(document.getElementById('c343-n').value);
    document.getElementById('c343-n-val').textContent = N;
    data = ch3.gen2d(N, TRUE_MU1, TRUE_MU2, TRUE_S11, TRUE_S12, TRUE_S22);
    refit();
  };

  window.ch3_343_regen = function () {
    data = ch3.gen2d(N, TRUE_MU1, TRUE_MU2, TRUE_S11, TRUE_S12, TRUE_S22);
    refit();
  };
};
