/**
 * ═══════════════════════════════════════════════════════════════════════
 *  Smart Insole — Diabetic Foot Ulcer Risk Assessment Dashboard
 *  Premium Interactive Script
 * ═══════════════════════════════════════════════════════════════════════
 *
 *  Dashboard data is injected by the HTML template into window.DASHBOARD_DATA
 *
 *  Sections:
 *   1. Live Clock
 *   2. Animated Risk Gauge (SVG)
 *   3. Animated Counters
 *   4. Foot Pressure Map (SVG Interaction)
 *   5. Chart.js Trend Charts
 *   6. Chart Tab Switching
 *   7. Sensor Status
 *   8. Prediction History
 *   9. Initialization
 *  10. Simulated Live Update (Demo)
 */

'use strict';

/* ═══════════════════════════════════════════════════════════════════════
   1. LIVE CLOCK
   ═══════════════════════════════════════════════════════════════════════ */

function startClock() {
  const el = document.getElementById('nav-time');
  if (!el) return;

  function tick() {
    const now = new Date();
    const h = now.getHours();
    const m = String(now.getMinutes()).padStart(2, '0');
    const s = String(now.getSeconds()).padStart(2, '0');
    const ampm = h >= 12 ? 'PM' : 'AM';
    const h12 = String(h % 12 || 12).padStart(2, '0');

    const months = ['Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec'];
    const day = String(now.getDate()).padStart(2, '0');
    const mon = months[now.getMonth()];
    const year = now.getFullYear();

    el.innerHTML =
      `<span class="clock-time">${h12}:${m}:${s} ${ampm}</span>` +
      `<span class="clock-date">${day} ${mon} ${year}</span>`;
  }

  tick();
  setInterval(tick, 1000);
}


/* ═══════════════════════════════════════════════════════════════════════
   2. ANIMATED RISK GAUGE (SVG)
   ═══════════════════════════════════════════════════════════════════════ */

const GAUGE = {
  radius:        120,
  circumference: 2 * Math.PI * 120,         // ≈ 753.98
  arc:           0.75 * (2 * Math.PI * 120), // 270° sweep
  startAngle:    225,                        // degrees, bottom-left
};

/**
 * Easing — cubic ease-out for silky deceleration.
 */
function easeOutCubic(t) {
  return 1 - Math.pow(1 - t, 3);
}

/**
 * Derive the gauge colour from a 0-100 risk score.
 */
function gaugeColor(score) {
  if (score < 30) return '#10B981';
  if (score < 60) return '#F59E0B';
  return '#EF4444';
}

/**
 * Return risk label & CSS class for a given score.
 */
function riskMeta(score) {
  if (score < 30) return { label: 'LOW RISK',    cls: 'low' };
  if (score < 60) return { label: 'MEDIUM RISK', cls: 'medium' };
  return                  { label: 'HIGH RISK',   cls: 'high' };
}

/**
 * Animate the SVG arc gauge from 0 → score over `duration` ms.
 */
function animateGauge(score, duration) {
  duration = duration || 1500;

  const progress = document.getElementById('gauge-progress');
  const valueEl  = document.getElementById('gauge-value');
  const badge    = document.querySelector('.risk-badge');
  if (!progress || !valueEl) return;

  const color = gaugeColor(score);
  const meta  = riskMeta(score);

  // Set stroke colour & optional glow
  progress.style.stroke = color;
  progress.style.filter = `drop-shadow(0 0 8px ${color}66)`;

  // Risk badge
  if (badge) {
    badge.textContent = meta.label;
    badge.className   = 'risk-badge ' + meta.cls;
  }

  // Dash setup — full offset = arc length (empty), target = proportional fill
  const fullOffset   = GAUGE.arc;
  const targetOffset = GAUGE.arc - (score / 100) * GAUGE.arc;

  progress.style.strokeDasharray  = `${GAUGE.arc} ${GAUGE.circumference}`;
  progress.style.strokeDashoffset = fullOffset;

  let start = null;

  function step(ts) {
    if (!start) start = ts;
    const elapsed = ts - start;
    const t       = Math.min(elapsed / duration, 1);
    const ease    = easeOutCubic(t);

    // Animate arc
    const currentOffset = fullOffset - ease * (fullOffset - targetOffset);
    progress.style.strokeDashoffset = currentOffset;

    // Animate number
    valueEl.textContent = Math.round(ease * score);

    if (t < 1) {
      requestAnimationFrame(step);
    }
  }

  requestAnimationFrame(step);
}


/* ═══════════════════════════════════════════════════════════════════════
   3. ANIMATED COUNTERS
   ═══════════════════════════════════════════════════════════════════════ */

/**
 * Count-up animation for a stat card.
 *
 * @param {string}  elementId   – DOM id of the value element
 * @param {number}  target      – target number
 * @param {string}  suffix      – unit suffix ('%', '°C', ' bpm', …)
 * @param {number}  [duration]  – animation length in ms (default 1500)
 * @param {number}  [decimals]  – decimal places (default 0)
 */
function animateCounter(elementId, target, suffix, duration, decimals) {
  duration = duration || 1500;
  decimals = decimals || 0;

  const el = document.getElementById(elementId);
  if (!el) return;

  let start = null;

  function step(ts) {
    if (!start) start = ts;
    const t    = Math.min((ts - start) / duration, 1);
    const ease = easeOutCubic(t);
    const val  = (ease * target).toFixed(decimals);
    el.textContent = val + suffix;

    if (t < 1) requestAnimationFrame(step);
  }

  requestAnimationFrame(step);
}


/* ═══════════════════════════════════════════════════════════════════════
   4. FOOT PRESSURE MAP (SVG INTERACTION)
   ═══════════════════════════════════════════════════════════════════════ */

const PRESSURE_ZONES = [
  { region: 'foot-heel',              text: 'heel-value',    name: 'Heel' },
  { region: 'foot-medial-forefoot',   text: 'medial-value',  name: 'Medial Forefoot' },
  { region: 'foot-lateral-forefoot',  text: 'lateral-value', name: 'Lateral Forefoot' },
  { region: 'foot-toe',              text: 'toe-value',     name: 'Toe' },
];

/**
 * Pressure value → heat colour.
 */
function pressureColor(val) {
  if (val <= 25) return '#10B981';
  if (val <= 50) return '#F59E0B';
  if (val <= 75) return '#F97316';
  return '#EF4444';
}

/**
 * Colour all foot regions and update text labels.
 * @param {number[]} values – [heel, medialForefoot, lateralForefoot, toe] (0-100)
 */
function updateFootPressure(values) {
  PRESSURE_ZONES.forEach((zone, i) => {
    const region = document.getElementById(zone.region);
    const label  = document.getElementById(zone.text);
    const val    = Math.round(Math.max(0, Math.min(100, values[i])));

    if (region) {
      region.style.transition = 'fill 0.6s ease';
      region.style.fill       = pressureColor(val);
    }
    if (label) {
      label.textContent = val + '%';
    }
  });
}

/**
 * Attach hover tooltips to each foot region.
 */
function initFootTooltips() {
  // Create a floating tooltip element
  const tip = document.createElement('div');
  tip.className = 'foot-tooltip';
  Object.assign(tip.style, {
    position: 'fixed', pointerEvents: 'none', opacity: '0',
    transition: 'opacity 0.2s', zIndex: '9999',
    background: 'rgba(15,23,42,0.92)', color: '#fff',
    padding: '6px 14px', borderRadius: '8px',
    fontSize: '13px', fontFamily: 'Poppins, sans-serif',
    boxShadow: '0 4px 14px rgba(0,0,0,0.25)',
  });
  document.body.appendChild(tip);

  PRESSURE_ZONES.forEach((zone) => {
    const region = document.getElementById(zone.region);
    if (!region) return;

    region.style.cursor = 'pointer';

    region.addEventListener('mouseenter', (e) => {
      const label = document.getElementById(zone.text);
      const val   = label ? label.textContent : '—';
      tip.innerHTML = `<strong>${zone.name}</strong><br>Pressure: ${val}`;
      tip.style.opacity = '1';
    });

    region.addEventListener('mousemove', (e) => {
      tip.style.left = (e.clientX + 14) + 'px';
      tip.style.top  = (e.clientY + 14) + 'px';
    });

    region.addEventListener('mouseleave', () => {
      tip.style.opacity = '0';
    });
  });
}


/* ═══════════════════════════════════════════════════════════════════════
   5. CHART.JS TREND CHARTS
   ═══════════════════════════════════════════════════════════════════════ */

// Store chart instances for later updates
const chartInstances = {};

/**
 * Generate `count` timestamps going backwards from now, 1 min apart.
 * @returns {string[]} formatted 'HH:MM'
 */
function generateDummyTimestamps(count) {
  const stamps = [];
  const now = Date.now();
  for (let i = count - 1; i >= 0; i--) {
    const d = new Date(now - i * 60000);
    const h = String(d.getHours()).padStart(2, '0');
    const m = String(d.getMinutes()).padStart(2, '0');
    stamps.push(`${h}:${m}`);
  }
  return stamps;
}

/**
 * Generate smoothly-varying random data.
 */
function generateDummyData(count, min, max) {
  const data = [];
  let prev = (min + max) / 2;
  for (let i = 0; i < count; i++) {
    const jitter = (Math.random() - 0.5) * (max - min) * 0.25;
    prev = Math.max(min, Math.min(max, prev + jitter));
    data.push(parseFloat(prev.toFixed(1)));
  }
  return data;
}

/**
 * Factory — create a premium Chart.js line chart with gradient fill.
 */
function createTrendChart(canvasId, label, data, color) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return null;

  const ctx = canvas.getContext('2d');

  // Gradient fill
  const gradient = ctx.createLinearGradient(0, 0, 0, canvas.clientHeight || 200);
  gradient.addColorStop(0, color + '33'); // semi-transparent
  gradient.addColorStop(1, color + '00'); // fully transparent

  const labels = generateDummyTimestamps(data.length);

  const chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: label,
        data: data,
        borderColor: color,
        backgroundColor: gradient,
        borderWidth: 2.5,
        fill: true,
        tension: 0.4,
        pointRadius: 0,
        pointHoverRadius: 5,
        pointBackgroundColor: color,
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: color,
        pointHoverBorderWidth: 2,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 1500, easing: 'easeOutQuart' },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(15,23,42,0.9)',
          titleFont:  { family: 'Poppins', size: 12 },
          bodyFont:   { family: 'Poppins', size: 13 },
          padding: 12,
          cornerRadius: 10,
          displayColors: false,
        },
      },
      scales: {
        x: {
          grid: { color: '#F1F5F9', drawBorder: false },
          ticks: { font: { family: 'Poppins', size: 11 }, color: '#94A3B8', maxTicksLimit: 8 },
        },
        y: {
          grid: { color: '#F1F5F9', drawBorder: false },
          ticks: { font: { family: 'Poppins', size: 11 }, color: '#94A3B8' },
        },
      },
    },
  });

  chartInstances[canvasId] = chart;
  return chart;
}


/* ═══════════════════════════════════════════════════════════════════════
   6. CHART TAB SWITCHING
   ═══════════════════════════════════════════════════════════════════════ */

function initChartTabs() {
  document.querySelectorAll('.chart-tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      // Deactivate siblings
      const parent = tab.closest('.chart-tabs');
      if (parent) {
        parent.querySelectorAll('.chart-tab').forEach((t) => t.classList.remove('active'));
      }
      tab.classList.add('active');

      // Show matching canvas container, hide the rest
      const target  = tab.dataset.chart;
      const wrapper = tab.closest('.card, .chart-section');
      if (!wrapper) return;

      wrapper.querySelectorAll('.chart-wrap').forEach((c) => {
        c.classList.toggle('active', c.id === target);
      });
    });
  });
}


/* ═══════════════════════════════════════════════════════════════════════
   7. SENSOR STATUS
   ═══════════════════════════════════════════════════════════════════════ */

/**
 * Toggle a sensor dot between online / offline.
 */
function updateSensorStatus(sensorId, isOnline) {
  const el   = document.getElementById('sensor-' + sensorId);
  if (!el) return;
  const dot  = el.querySelector('.sensor-dot');
  const text = el.querySelector('.sensor-label');

  if (dot) {
    dot.classList.toggle('online',  isOnline);
    dot.classList.toggle('offline', !isOnline);
  }
  if (text) {
    text.textContent = isOnline ? 'Connected' : 'Disconnected';
  }
}

function initSensors(sensorData) {
  if (!sensorData) return;
  Object.entries(sensorData).forEach(([id, online]) => {
    updateSensorStatus(id, online);
  });
}


/* ═══════════════════════════════════════════════════════════════════════
   8. PREDICTION HISTORY
   ═══════════════════════════════════════════════════════════════════════ */

/**
 * Populate the history table body.
 */
function populateHistory(historyData) {
  const tbody = document.querySelector('#history-table tbody');
  if (!tbody) return;

  tbody.innerHTML = '';

  historyData.forEach((row) => {
    const tr = document.createElement('tr');

    // Inline micro bar width
    const barWidth = Math.min(row.risk_score, 100);
    const barColor = gaugeColor(row.risk_score);

    tr.innerHTML =
      `<td>${row.timestamp}</td>` +
      `<td>
         <div class="history-score">
           <span>${row.risk_score}</span>
           <div class="micro-bar"><div style="width:${barWidth}%;background:${barColor}"></div></div>
         </div>
       </td>` +
      `<td><span class="status-pill ${row.status_class}">${row.status}</span></td>`;

    tbody.appendChild(tr);
  });
}

/**
 * Generate dummy history entries for demo.
 */
function generateDummyHistory(count) {
  const history = [];
  const now = Date.now();

  for (let i = 0; i < count; i++) {
    const d     = new Date(now - i * 10 * 60000); // 10 min intervals
    const score = Math.round(Math.random() * 80 + 10);
    const meta  = riskMeta(score);

    history.push({
      timestamp:    formatTimestamp(d),
      risk_score:   score,
      status:       meta.label,
      status_class: meta.cls,
    });
  }
  return history;
}

function formatTimestamp(d) {
  const h = String(d.getHours() % 12 || 12).padStart(2, '0');
  const m = String(d.getMinutes()).padStart(2, '0');
  const ampm = d.getHours() >= 12 ? 'PM' : 'AM';
  const day  = String(d.getDate()).padStart(2, '0');
  const mon  = ['Jan','Feb','Mar','Apr','May','Jun',
                'Jul','Aug','Sep','Oct','Nov','Dec'][d.getMonth()];
  return `${day} ${mon} ${h}:${m} ${ampm}`;
}


/* ═══════════════════════════════════════════════════════════════════════
   9. INITIALIZATION
   ═══════════════════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
  // 1. Clock
  startClock();

  // 2. Dashboard data (injected by Flask / Jinja)
  const D = window.DASHBOARD_DATA || {
    risk_score:       42,
    spo2:             97,
    heart_rate:       78,
    temperature_diff: 1.8,
    epti:             64,
    pressure_values:  [35, 62, 48, 71],
    sensors: {
      'sensor-pressure':  true,
      'sensor-temp':      true,
      'sensor-imu':       true,
      'sensor-spo2':      true,
    },
  };

  // 3. Gauge
  animateGauge(D.risk_score);

  // 4. Stat counters
  animateCounter('spo2-value', D.spo2,              '%');
  animateCounter('hr-value',   D.heart_rate,         ' bpm');
  animateCounter('temp-value', D.temperature_diff,   '°C', 1500, 1);
  animateCounter('epti-value', D.epti,               '');

  // 5. Foot pressure map
  // Convert dict {heel, medial_forefoot, lateral_forefoot, toe} → array
  const pv = D.pressure_values;
  const pressureArr = Array.isArray(pv)
    ? pv
    : [pv.heel || 0, pv.medial_forefoot || 0, pv.lateral_forefoot || 0, pv.toe || 0];
  updateFootPressure(pressureArr);
  initFootTooltips();

  // 6. Trend charts
  createTrendChart(
    'pressureTrendChart', 'Avg Pressure (%)',
    generateDummyData(20, 30, 70), '#2563EB'
  );
  createTrendChart(
    'temperatureTrendChart', 'Temperature Δ (°C)',
    generateDummyData(20, 0.5, 3.0), '#F59E0B'
  );
  createTrendChart(
    'riskTrendChart', 'Risk Score',
    generateDummyData(20, 20, 85), '#EF4444'
  );

  // 6b. Regional Pressure bar chart
  const rpCanvas = document.getElementById('regionalPressureChart');
  if (rpCanvas) {
    const rpCtx = rpCanvas.getContext('2d');
    const rpLabels = ['Heel', 'Medial Forefoot', 'Lateral Forefoot', 'Toe'];
    const rpValues = pressureArr;
    const rpColors = rpValues.map(v => pressureColor(v));

    new Chart(rpCtx, {
      type: 'bar',
      data: {
        labels: rpLabels,
        datasets: [{
          label: 'Pressure (%)',
          data: rpValues,
          backgroundColor: rpColors,
          borderRadius: 8,
          barThickness: 28,
        }],
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 1500, easing: 'easeOutQuart' },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: 'rgba(15,23,42,0.9)',
            titleFont: { family: 'Poppins', size: 12 },
            bodyFont: { family: 'Poppins', size: 13 },
            padding: 12,
            cornerRadius: 10,
            callbacks: { label: (ctx) => `${ctx.parsed.x}%` },
          },
        },
        scales: {
          x: {
            beginAtZero: true,
            max: 100,
            grid: { color: '#F1F5F9', drawBorder: false },
            ticks: { font: { family: 'Poppins', size: 11 }, color: '#94A3B8', stepSize: 25 },
          },
          y: {
            grid: { display: false },
            ticks: { font: { family: 'Poppins', size: 12, weight: '500' }, color: '#334155' },
          },
        },
      },
    });
  }

  // 7. Chart tabs
  initChartTabs();

  // 8. Sensors
  initSensors(D.sensors);

  // 9. Prediction history (use server data if available, else dummy)
  const historyData = (D.prediction_history && D.prediction_history.length)
    ? D.prediction_history
    : generateDummyHistory(8);
  populateHistory(historyData);

  // 10. Staggered card entrance animations
  document.querySelectorAll('.card, .stat-card, .chart-section').forEach((card, i) => {
    card.style.opacity   = '0';
    card.style.transform = 'translateY(24px)';
    card.style.transition = 'opacity 0.5s ease, transform 0.5s ease';

    setTimeout(() => {
      card.style.opacity   = '1';
      card.style.transform = 'translateY(0)';
    }, 120 * i);
  });

  // 11. Start live simulation after 3 s
  setTimeout(simulateLiveData, 3000);
});


/* ═══════════════════════════════════════════════════════════════════════
   10. SIMULATED LIVE UPDATE (Demo)
   ═══════════════════════════════════════════════════════════════════════ */

function simulateLiveData() {
  const D = window.DASHBOARD_DATA || {};
  const pvInit = D.pressure_values;
  let pressures = Array.isArray(pvInit)
    ? [...pvInit]
    : pvInit
      ? [pvInit.heel || 35, pvInit.medial_forefoot || 62, pvInit.lateral_forefoot || 48, pvInit.toe || 71]
      : [35, 62, 48, 71];

  setInterval(() => {
    // Jitter pressure values by ±3, clamped 0-100
    pressures = pressures.map((v) => {
      const delta = (Math.random() - 0.5) * 6;
      return Math.round(Math.max(0, Math.min(100, v + delta)));
    });
    updateFootPressure(pressures);

    // Append new point to each chart and trim oldest
    Object.entries(chartInstances).forEach(([id, chart]) => {
      const ds   = chart.data.datasets[0];
      const last = ds.data[ds.data.length - 1];
      const range = id === 'temperatureTrendChart'
        ? { min: 0.5, max: 3.0, scale: 0.3 }
        : id === 'pressureTrendChart'
          ? { min: 30, max: 70, scale: 4 }
          : { min: 20, max: 85, scale: 5 };

      const jitter = (Math.random() - 0.5) * range.scale * 2;
      const next   = Math.max(range.min, Math.min(range.max, last + jitter));

      ds.data.push(parseFloat(next.toFixed(1)));
      ds.data.shift();

      // Shift timestamp labels
      const now = new Date();
      chart.data.labels.push(
        String(now.getHours()).padStart(2, '0') + ':' +
        String(now.getMinutes()).padStart(2, '0')
      );
      chart.data.labels.shift();

      chart.update('none'); // instant update, no re-animation
    });
  }, 5000);
}
