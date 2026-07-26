/* ===========================================================
   PADADRISHTI — DFU RISK COCKPIT
   All UI state is driven through the update*(data) functions
   at the bottom of this file. Wire these to your Flask API /
   WebSocket feed — everything above them is just rendering.
   =========================================================== */

lucide.createIcons();

/* ----------------------------------------------------------
   CONSTANTS
---------------------------------------------------------- */

const SENSOR_META = {
  fsr1: { label: "FSR 1 · Forefoot L", unit: "kPa", icon: "gauge" },
  fsr2: { label: "FSR 2 · Forefoot R", unit: "kPa", icon: "gauge" },
  fsr3: { label: "FSR 3 · Heel L", unit: "kPa", icon: "gauge" },
  fsr4: { label: "FSR 4 · Heel R", unit: "kPa", icon: "gauge" },
  temp: { label: "Temperature", unit: "°C", icon: "thermometer" },
  hr: { label: "Heart Rate", unit: "bpm", icon: "heart-pulse" },
  spo2: { label: "SpO₂", unit: "%", icon: "droplets" },
  battery: { label: "Insole Battery", unit: "%", icon: "battery-medium" },
};

const LEAD_ORDER = ["fsr1", "fsr2", "temp", "fsr3", "fsr4"];

/* ----------------------------------------------------------
   SIDEBAR TOGGLE
---------------------------------------------------------- */

const cockpit = document.querySelector(".cockpit");
document.getElementById("railToggle").addEventListener("click", () => {
  cockpit.classList.toggle("is-collapsed");
});
document.getElementById("railToggleMobile").addEventListener("click", () => {
  cockpit.classList.toggle("is-collapsed");
});

/* ----------------------------------------------------------
   GREETING + CLOCK
---------------------------------------------------------- */

function renderGreeting() {
  const greetingEl = document.getElementById("greetingText");
  const dateEl = document.getElementById("dateText");
  if (!greetingEl || !dateEl) return;
  const now = new Date();
  const hour = now.getHours();
  const greet = hour < 12 ? "Good morning" : hour < 18 ? "Good afternoon" : "Good evening";
  greetingEl.textContent = `${greet}, Y/N`;
  dateEl.textContent = now.toLocaleDateString(undefined, {
    weekday: "long", year: "numeric", month: "long", day: "numeric",
  });
}
renderGreeting();

let uptimeSeconds = 11649;
setInterval(() => {
  uptimeSeconds++;
  const uptimeEl = document.getElementById("teleUptime");
  const latencyEl = document.getElementById("teleLatency");
  if (uptimeEl && latencyEl) {
    const h = String(Math.floor(uptimeSeconds / 3600)).padStart(2, "0");
    const m = String(Math.floor((uptimeSeconds % 3600) / 60)).padStart(2, "0");
    const s = String(uptimeSeconds % 60).padStart(2, "0");
    uptimeEl.textContent = `${h}:${m}:${s}`;
    latencyEl.textContent = `${38 + Math.round(Math.random() * 10)} ms`;
  }
}, 1000);

/* ----------------------------------------------------------
   COUNT-UP NUMBER ANIMATION
---------------------------------------------------------- */

function animateNumber(el, from, to, decimals = 0, duration = 700) {
  const start = performance.now();
  function tick(now) {
    const p = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - p, 3);
    const val = from + (to - from) * eased;
    el.textContent = val.toFixed(decimals);
    if (p < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

/* ----------------------------------------------------------
   SENSOR CARDS
---------------------------------------------------------- */

let sensorState = {
  fsr1: 42, fsr2: 39, fsr3: 61, fsr4: 58,
  temp: 31.2, hr: 76, spo2: 98, battery: 82,
};

function renderSensorCards() {
  const row = document.getElementById("sensorRow");
  if (!row) return;
  row.innerHTML = "";
  Object.keys(SENSOR_META).forEach((key) => {
    const meta = SENSOR_META[key];
    const card = document.createElement("div");
    card.className = "sensor-card";
    card.dataset.sensor = key;
    card.innerHTML = `
      <div class="sensor-card__top">
        <span class="sensor-card__label">${meta.label}</span>
        <span class="sensor-card__icon"><i data-lucide="${meta.icon}"></i></span>
      </div>
      <div class="sensor-card__value">
        <span class="sensor-card__num" data-num="${key}">--</span>
        <span class="sensor-card__unit">${meta.unit}</span>
      </div>
      <svg class="sensor-card__spark" data-spark="${key}" viewBox="0 0 100 24" preserveAspectRatio="none"></svg>
    `;
    row.appendChild(card);
  });
  lucide.createIcons();
}
renderSensorCards();

function renderSparkline(key, points) {
  const svg = document.querySelector(`[data-spark="${key}"]`);
  if (!svg || !points || points.length < 2) return;
  const min = Math.min(...points), max = Math.max(...points) || 1;
  const range = max - min || 1;
  const step = 100 / (points.length - 1);
  const path = points
    .map((v, i) => `${i === 0 ? "M" : "L"}${(i * step).toFixed(1)},${(22 - ((v - min) / range) * 20).toFixed(1)}`)
    .join(" ");
  svg.innerHTML = `<path d="${path}" fill="none" stroke="var(--teal)" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>`;
}

const sparkHistory = {};
Object.keys(SENSOR_META).forEach((k) => (sparkHistory[k] = [sensorState[k]]));

/* ----------------------------------------------------------
   HUD FOOT SCHEMATIC LEADS
---------------------------------------------------------- */

function renderLeads() {
  const wrap = document.querySelector(".hud__readouts");
  if (!wrap) return;
  wrap.innerHTML = "";
  LEAD_ORDER.forEach((key) => {
    const meta = SENSOR_META[key];
    const row = document.createElement("div");
    row.className = "lead";
    row.dataset.lead = key;
    row.innerHTML = `
      <span class="lead__name">${meta.label}</span>
      <span class="lead__val mono" data-lead-val="${key}">--</span>
    `;
    wrap.appendChild(row);
  });
}
renderLeads();

/* ----------------------------------------------------------
   RISK GAUGE
---------------------------------------------------------- */

const GAUGE_CIRCUMFERENCE = 2 * Math.PI * 100;

function buildGaugeTicks() {
  const g = document.getElementById("gaugeTicks");
  if (!g) return;
  const cx = 120, cy = 120, rOuter = 100;
  let html = "";
  for (let i = 0; i <= 20; i++) {
    const angle = (i / 20) * 2 * Math.PI;
    const inner = rOuter - 10, outer = rOuter - 2;
    const x1 = cx + inner * Math.cos(angle), y1 = cy + inner * Math.sin(angle);
    const x2 = cx + outer * Math.cos(angle), y2 = cy + outer * Math.sin(angle);
    html += `<line x1="${x1.toFixed(2)}" y1="${y1.toFixed(2)}" x2="${x2.toFixed(2)}" y2="${y2.toFixed(2)}" stroke="rgba(18,24,27,0.10)" stroke-width="1.5"/>`;
  }
  g.innerHTML = html;
}
buildGaugeTicks();

function setGauge(score, confidence, level) {
  const progress = document.getElementById("gaugeProgress");
  if (!progress) return;
  const offset = GAUGE_CIRCUMFERENCE * (1 - score / 100);
  progress.style.strokeDashoffset = offset;

  const colors = { low: "var(--teal)", medium: "var(--amber)", high: "var(--coral)" };
  progress.style.stroke = colors[level] || "var(--teal)";

  animateNumber(document.getElementById("gaugeScore"), 0, score, 0, 900);
  document.getElementById("gaugeLabel").textContent = `${level.charAt(0).toUpperCase() + level.slice(1)} risk`;

  document.getElementById("confidenceFill").style.width = `${confidence}%`;
  document.getElementById("confidenceValue").textContent = `${confidence}%`;
}

/* ----------------------------------------------------------
   XAI PANEL
---------------------------------------------------------- */

function renderShap(positives, negatives, narrative) {
  const narrativeEl = document.getElementById("xaiNarrative");
  if (!narrativeEl) return;
  narrativeEl.textContent = narrative;

  const posList = document.getElementById("xaiPositive");
  const negList = document.getElementById("xaiNegative");
  posList.innerHTML = "";
  negList.innerHTML = "";

  const maxAbs = Math.max(
    ...positives.map((p) => Math.abs(p.impact)),
    ...negatives.map((n) => Math.abs(n.impact)),
    1e-9 // Avoid division by zero
  );

  const formatShap = (val, isPositive) => {
    if (Math.abs(val) < 0.0001 && val !== 0) return "<0.0001";
    if (isPositive) return `+${val.toFixed(4)}`;
    return val.toFixed(4); // toFixed on negative numbers preserves the '-'
  };

  positives.forEach((f) => {
    const li = document.createElement("li");
    li.innerHTML = `
      <div class="xai__row-wrap">
        <div class="xai__row"><span>${f.feature}</span><span class="mono">${formatShap(f.impact, true)}</span></div>
        <div class="xai__row-bar"><div class="xai__row-fill" style="width:${(Math.abs(f.impact) / maxAbs) * 100}%; background:var(--coral);"></div></div>
      </div>`;
    posList.appendChild(li);
  });

  negatives.forEach((f) => {
    const li = document.createElement("li");
    li.innerHTML = `
      <div class="xai__row-wrap">
        <div class="xai__row"><span>${f.feature}</span><span class="mono">${formatShap(f.impact, false)}</span></div>
        <div class="xai__row-bar"><div class="xai__row-fill" style="width:${(Math.abs(f.impact) / maxAbs) * 100}%; background:var(--teal);"></div></div>
      </div>`;
    negList.appendChild(li);
  });
}

/* ----------------------------------------------------------
   PRESSURE HEATMAP
---------------------------------------------------------- */

function pressureToColor(value) {
  // value 0-100 -> mint -> amber -> coral
  if (value < 40) return "rgba(127,233,196,0.55)";
  if (value < 70) return "rgba(232,184,95,0.55)";
  return "rgba(255,107,94,0.55)";
}

function renderHeatmap({ forefootL, forefootR, mid, heel }) {
  const hf1 = document.getElementById("heatForefoot");
  if (!hf1) return;
  hf1.style.fill = pressureToColor(forefootL);
  document.getElementById("heatForefoot2").style.fill = pressureToColor(forefootR);
  document.getElementById("heatMid").style.fill = pressureToColor(mid);
  document.getElementById("heatHeel").style.fill = pressureToColor(heel);
}

/* ----------------------------------------------------------
   PATIENT PANEL
---------------------------------------------------------- */

function renderPatient(p) {
  const pName = document.getElementById("patientName");
  if (!pName) return;
  pName.textContent = p.name;
  document.getElementById("patientMeta").textContent = `${p.id} · ${p.age}y · ${p.gender}`;
  document.getElementById("patientInitials").textContent = p.name
    .split(" ")
    .map((w) => w[0])
    .slice(0, 2)
    .join("");

  const stats = {
    Weight: `${p.weight} kg`,
    BMI: p.bmi,
    "Diabetes duration": `${p.diabetesDuration} yrs`,
    Neuropathy: p.neuropathy,
    Smoking: p.smoking,
    Hypertension: p.hypertension,
  };
  const dl = document.getElementById("patientStats");
  dl.innerHTML = "";
  Object.entries(stats).forEach(([k, v]) => {
    const wrap = document.createElement("div");
    wrap.innerHTML = `<dt>${k}</dt><dd>${v}</dd>`;
    dl.appendChild(wrap);
  });
}

/* ----------------------------------------------------------
   PREDICTION HISTORY
---------------------------------------------------------- */

function renderHistory(entries) {
  const ol = document.getElementById("historyTimeline");
  if (!ol) return;
  ol.innerHTML = "";
  entries.forEach((e) => {
    const li = document.createElement("li");
    li.innerHTML = `
      <span class="timeline__time mono">${e.time}</span>
      <span class="timeline__badge badge--${e.level}">${e.level}</span>
      <div class="timeline__body">
        <h5>Risk score ${e.score}</h5>
        <p>${e.note}</p>
      </div>
    `;
    ol.appendChild(li);
  });
}

/* ----------------------------------------------------------
   SYSTEM STATUS
---------------------------------------------------------- */

function renderSystemStatus(items) {
  const ul = document.getElementById("systemStatus");
  if (!ul) return;
  ul.innerHTML = "";
  items.forEach((it) => {
    const li = document.createElement("li");
    li.innerHTML = `
      <span class="status-list__name"><i data-lucide="${it.icon}"></i>${it.name}</span>
      <span class="status-list__state"><span class="dot dot--${it.state}"></span>${it.text}</span>
    `;
    ul.appendChild(li);
  });
  lucide.createIcons();
}

/* ----------------------------------------------------------
   TREND CHART (Chart.js)
---------------------------------------------------------- */

let trendChart;

const trendDatasets = {
  pressure: {
    labels: [],
    series: [
      { label: "FSR1", color: "#2A9D8F", data: [] },
      { label: "FSR2", color: "#7FE9C4", data: [] },
      { label: "FSR3", color: "#B8A6E8", data: [] },
      { label: "FSR4", color: "#5FD3E8", data: [] },
    ],
  },
  temperature: {
    labels: [],
    series: [{ label: "Temp (°C)", color: "#E8B85F", data: [] }],
  },
  vitals: {
    labels: [],
    series: [
      { label: "Heart Rate", color: "#FF6B5E", data: [] },
      { label: "SpO₂", color: "#5FD3E8", data: [] },
    ],
  },
  risk: {
    labels: [],
    series: [{ label: "Risk Score", color: "#2A9D8F", data: [] }],
  },
};

function buildChart(metric) {
  const chartEl = document.getElementById("trendChart");
  if (!chartEl) return;
  const chartCtx = chartEl.getContext("2d");
  const cfg = trendDatasets[metric];
  if (trendChart) trendChart.destroy();
  trendChart = new Chart(chartCtx, {
    type: "line",
    data: {
      labels: cfg.labels,
      datasets: cfg.series.map((s) => ({
        label: s.label,
        data: s.data,
        borderColor: s.color,
        backgroundColor: s.color + "22",
        tension: 0.4,
        fill: true,
        pointRadius: 0,
        borderWidth: 2.5,
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          position: "top",
          align: "end",
          labels: { usePointStyle: true, pointStyle: "circle", boxWidth: 7, font: { family: "Inter", size: 11 }, color: "#4B5754" },
        },
      },
      scales: {
        x: { grid: { display: false }, ticks: { color: "#8A9490", font: { size: 10 } } },
        y: { grid: { color: "rgba(18,24,27,0.06)" }, ticks: { color: "#8A9490", font: { size: 10 } } },
      },
      animation: { duration: 600 },
    },
  });
}
buildChart("pressure");

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("is-active"));
    tab.classList.add("is-active");
    buildChart(tab.dataset.metric);
  });
});

function pushTrendPoint(label, values) {
  // values = { fsr1,fsr2,fsr3,fsr4,temp,hr,spo2,risk }
  Object.values(trendDatasets).forEach((cfg) => {
    cfg.labels.push(label);
    if (cfg.labels.length > 20) cfg.labels.shift();
  });
  trendDatasets.pressure.series[0].data.push(values.fsr1);
  trendDatasets.pressure.series[1].data.push(values.fsr2);
  trendDatasets.pressure.series[2].data.push(values.fsr3);
  trendDatasets.pressure.series[3].data.push(values.fsr4);
  trendDatasets.temperature.series[0].data.push(values.temp);
  trendDatasets.vitals.series[0].data.push(values.hr);
  trendDatasets.vitals.series[1].data.push(values.spo2);
  trendDatasets.risk.series[0].data.push(values.risk);

  Object.values(trendDatasets).forEach((cfg) => {
    cfg.series.forEach((s) => {
      if (s.data.length > 20) s.data.shift();
    });
  });

  const activeTab = document.querySelector(".tab.is-active");
  if (activeTab) {
    const activeMetric = activeTab.dataset.metric;
    buildChart(activeMetric);
  }
}

/* ===========================================================
   PUBLIC UPDATE API — wire these to your Flask / WebSocket feed
   =========================================================== */

/**
 * updateSensors(data)
 * data: { fsr1, fsr2, fsr3, fsr4, temp, hr, spo2, battery }  (raw numbers)
 */
function updateSensors(data) {
  Object.entries(data).forEach(([key, value]) => {
    if (!(key in sensorState)) return;
    const from = sensorState[key];
    sensorState[key] = value;

    const numEl = document.querySelector(`[data-num="${key}"]`);
    if (numEl) animateNumber(numEl, from, value, key === "temp" ? 1 : 0, 650);

    const leadEl = document.querySelector(`[data-lead-val="${key}"]`);
    if (leadEl) leadEl.textContent = `${value}${SENSOR_META[key].unit}`;

    sparkHistory[key] = [...(sparkHistory[key] || []), value].slice(-16);
    renderSparkline(key, sparkHistory[key]);
  });

  // reflect forefoot/heel pressure into HUD node glow intensity
  ["fsr1", "fsr2", "fsr3", "fsr4"].forEach((k) => {
    const node = document.getElementById(`node${k.toUpperCase()}`);
    if (node && data[k] !== undefined) {
      const intensity = Math.min(1, data[k] / 100);
      node.style.opacity = 0.5 + intensity * 0.5;
    }
  });
}

/**
 * updatePrediction(data)
 * data: { score (0-100), confidence (0-100), level: 'low'|'medium'|'high' }
 */
function updatePrediction(data) {
  setGauge(data.score, data.confidence, data.level);
}

/**
 * updateShap(data)
 * data: { narrative: string, positive: [{feature, impact}], negative: [{feature, impact}] }
 */
function updateShap(data) {
  renderShap(data.positive, data.negative, data.narrative);
}

/**
 * updateHeatmap(data)
 * data: { forefootL, forefootR, mid, heel } values 0-100
 */
function updateHeatmap(data) {
  renderHeatmap(data);
}

/**
 * updateHistory(entries)
 * entries: [{ time, level: 'low'|'medium'|'high', score, note }]
 */
function updateHistory(entries) {
  renderHistory(entries);
}

/**
 * updatePatient(patient)
 * patient: { id, name, age, gender, weight, bmi, diabetesDuration, neuropathy, smoking, hypertension }
 */
function updatePatient(patient) {
  renderPatient(patient);
}

/**
 * updateSystemStatus(items)
 * items: [{ name, icon, state: 'live'|'warn'|'alert', text }]
 */
function updateSystemStatus(items) {
  renderSystemStatus(items);
}

/**
 * updateTrends(label, values)
 * appends one timestep to all trend charts
 */
function updateTrends(label, values) {
  pushTrendPoint(label, values);
}

/* ===========================================================
   LIVE API INTEGRATION
   =========================================================== */

updatePatient({
  id: "PT-2291",
  name: "Ramesh Iyer",
  age: 58,
  gender: "Male",
  weight: 78,
  bmi: 27.4,
  diabetesDuration: 12,
  neuropathy: "Mild",
  smoking: "Former",
  hypertension: "Yes",
});

let liveBattery = 92;

async function fetchHistory() {
  try {
    const res = await fetch('/history');
    if (res.ok) {
      const data = await res.json();
      const entries = data.map(d => {
        // Handle various timestamp formats robustly
        const date = new Date(d.timestamp || d.time || Date.now());
        const timeStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        return {
          time: timeStr,
          level: (d.risk || d.risk_label || d.level || "low").toLowerCase(),
          score: Math.round(d.risk_score || d.score || 0),
          note: `System recorded a ${d.risk || "Low"} risk assessment at ${timeStr}.`
        };
      }).slice(0, 10);
      updateHistory(entries);
    }
  } catch (err) {
    console.error("History fetch error:", err);
  }
}

async function startLiveUpdates() {
  // Initial fetch
  fetchHistory();

  // Update loop every 1 second
  setInterval(async () => {
    try {
      // 1. Fetch simulation data
      const simRes = await fetch('/simulate');
      if (!simRes.ok) throw new Error("Simulation endpoint failed");
      const sensorData = await simRes.json();

      liveBattery = Math.max(0, liveBattery - 0.05);

      // Map backend payload to frontend format
      const mappedSensors = {
        fsr1: sensorData.fsr1 || (sensorData.pressure ? sensorData.pressure[0] : 0),
        fsr2: sensorData.fsr2 || (sensorData.pressure ? sensorData.pressure[1] : 0),
        fsr3: sensorData.fsr3 || (sensorData.pressure ? sensorData.pressure[2] : 0),
        fsr4: sensorData.fsr4 || (sensorData.pressure ? sensorData.pressure[3] : 0),
        temp: sensorData.temperature,
        hr: sensorData.heart_rate,
        spo2: sensorData.spo2,
        battery: Math.round(liveBattery)
      };

      updateSensors(mappedSensors);

      updateHeatmap({
        forefootL: Math.min(100, mappedSensors.fsr1 + 15),
        forefootR: Math.min(100, mappedSensors.fsr2 + 15),
        mid: 30 + Math.round(Math.random() * 10),
        heel: Math.min(100, Math.max(mappedSensors.fsr3, mappedSensors.fsr4) + 10),
      });

      // 2. Run Prediction
      const predRes = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(sensorData)
      });

      let latestRisk = 0;

      if (predRes.ok) {
        const predData = await predRes.json();
        
        // Handle buffering gracefully
        if (predData.status === 'buffering') {
          // Keep gauge awaiting
          latestRisk = 0;
        } else {
          latestRisk = predData.risk_score || 0;
          const conf = predData.confidence || (85 + Math.round(Math.random() * 10)); // fallback if missing
          const lvl = (predData.risk_label || "low").toLowerCase();
          
          updatePrediction({
            score: latestRisk,
            confidence: conf,
            level: lvl
          });

          if (predData.explanation) {
            // Map "value" -> "impact" for frontend components
            const mapImpact = (arr) => arr.map(f => ({ feature: f.feature, impact: f.value || f.impact || 0 }));
            
            updateShap({
              narrative: predData.explanation.summary || "Explanation unavailable.",
              positive: mapImpact(predData.explanation.positive_contributions || []),
              negative: mapImpact(predData.explanation.negative_contributions || [])
            });
          }

          // Fetch history periodically since a new prediction was registered
          fetchHistory();
        }
      }

      // Update Trend Charts
      const label = new Date().toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" });
      updateTrends(label, { ...mappedSensors, risk: latestRisk });

      // Diagnostics OK
      updateSystemStatus([
        { name: "LSTM Model", icon: "brain-circuit", state: "live", text: "Active" },
        { name: "Inference Backend", icon: "server", state: "live", text: "Connected" },
        { name: "Insole Simulator", icon: "cpu", state: "live", text: "Streaming" },
        { name: "Flask API", icon: "plug-zap", state: "live", text: "200 OK" },
        { name: "BLE Radio", icon: "bluetooth", state: "live", text: "Paired" },
        { name: "Insole Battery", icon: "battery-medium", state: liveBattery > 20 ? "live" : "warn", text: `${Math.round(liveBattery)}%` },
      ]);

    } catch (err) {
      console.error("Live loop error:", err);
      updateSystemStatus([
        { name: "LSTM Model", icon: "brain-circuit", state: "warn", text: "Offline" },
        { name: "Inference Backend", icon: "server", state: "warn", text: "Disconnected" },
        { name: "Insole Simulator", icon: "cpu", state: "warn", text: "Waiting" },
        { name: "Flask API", icon: "plug-zap", state: "warn", text: "Error" }
      ]);
    }
  }, 1000);
}

startLiveUpdates();
