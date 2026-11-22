// app.js - Modern Interactive JavaScript for CyberGuard ML SOC Dashboard

let labelChartInstance = null;
let selectedBatchFile = null;
let cachedAttackSamples = {};
let currentFlowFeatures = {};

document.addEventListener("DOMContentLoaded", () => {
  initNavigation();
  refreshSystemStatus();
  loadEvaluationResults();
  updateModelOptions();
  setupDropzone();
  loadAttackSamples();
});

// ----------------- TOAST NOTIFICATIONS -----------------
function showToast(message, type = "info") {
  const container = document.getElementById("toast-container");
  if (!container) return;
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  const icon = type === "success" ? "circle-check" : (type === "error" ? "triangle-exclamation" : "circle-info");
  toast.innerHTML = `<i class="fa-solid fa-${icon}"></i> <span>${message}</span>`;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateX(50px)";
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

// ----------------- NAVIGATION -----------------
function initNavigation() {
  const navItems = document.querySelectorAll(".nav-item");
  navItems.forEach(item => {
    item.addEventListener("click", () => {
      const tabId = item.getAttribute("data-tab");
      switchTab(tabId);
    });
  });
}

function switchTab(tabId) {
  document.querySelectorAll(".nav-item").forEach(i => i.classList.remove("active"));
  document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));

  const targetNav = document.querySelector(`.nav-item[data-tab="${tabId}"]`);
  const targetPane = document.getElementById(tabId);

  if (targetNav) targetNav.classList.add("active");
  if (targetPane) targetPane.classList.add("active");

  const titles = {
    "tab-dashboard": "Security Operations Center Dashboard",
    "tab-dataset": "CICIDS2017 Dataset Explorer & Distribution",
    "tab-preprocessing": "Data Preprocessing & Cleaning Pipeline",
    "tab-training": "Interactive AI Model Training Center",
    "tab-evaluation": "Model Evaluation, Benchmarks & Confusion Matrices",
    "tab-prediction": "Live Threat Intelligence & Prediction Radar",
    "tab-docs": "Architecture & Mathematical Foundations"
  };
  if (titles[tabId]) {
    document.getElementById("topbar-title").innerText = titles[tabId];
  }

  // Trigger lazy loading
  if (tabId === "tab-dataset" && !labelChartInstance) {
    loadDatasetExplorer();
  } else if (tabId === "tab-evaluation") {
    loadEvaluationResults();
  }
}

// ----------------- SYSTEM STATUS & SYNC -----------------
async function refreshSystemStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.parse ? await res.parse() : await res.json();
    
    document.getElementById("stat-files").innerText = data.dataset_files || 0;
    document.getElementById("top-files-count").innerText = data.dataset_files || 0;
    
    const m = data.models_count || {};
    document.getElementById("stat-l1").innerText = m.level1 || 0;
    document.getElementById("stat-l2").innerText = m.level2 || 0;
    document.getElementById("stat-reg").innerText = m.regression || 0;
    document.getElementById("top-models-count").innerText = m.total || 0;
    document.getElementById("sidebar-models-stat").innerText = `${m.total || 0} Models Loaded`;

    // Populate prediction model dropdowns
    loadModelPickers();
    showToast("System status synchronized successfully.", "info");
  } catch (err) {
    console.error("Status fetch error:", err);
    showToast("Error connecting to CyberGuard backend.", "error");
  }
}

async function loadModelPickers() {
  try {
    const res = await fetch("/api/models");
    const data = await res.json();
    const saved = data.saved || {};

    const fillSelect = (selectId, modelList, defaultText) => {
      const select = document.getElementById(selectId);
      if (!select) return;
      select.innerHTML = "";
      if (modelList.length === 0) {
        select.innerHTML = `<option value="">${defaultText} (None saved)</option>`;
      } else {
        modelList.forEach(m => {
          const opt = document.createElement("option");
          opt.value = m.filename;
          opt.innerText = `${m.metadata.model_name || m.filename} (${m.metadata.timestamp || 'Trained'})`;
          select.appendChild(opt);
        });
      }
    };

    fillSelect("pred-l1-select", saved.level1 || [], "Default L1 Model");
    fillSelect("pred-l2-select", saved.level2 || [], "None (L1 Only)");
    fillSelect("pred-reg-select", saved.regression || [], "None (No Regression)");
  } catch (e) {
    console.error("Error loading model pickers:", e);
  }
}

// ----------------- DATASET EXPLORER -----------------
async function loadDatasetExplorer() {
  const fileContainer = document.getElementById("dataset-file-list");
  const tbody = document.getElementById("preview-body");
  const thead = document.getElementById("preview-headers");

  try {
    const res = await fetch("/api/dataset/preview?rows=100");
    const data = await res.json();

    if (!data.success) {
      fileContainer.innerHTML = `<p style="color:var(--threat-red);">${data.error || 'Failed to load dataset'}</p>`;
      return;
    }

    // Status cards
    document.getElementById("ds-col-count").innerText = data.total_columns;
    document.getElementById("ds-row-count").innerText = data.sample_rows_count;
    document.getElementById("ds-label-col").innerText = data.label_column || "Label";

    // Chart.js Label Breakdown
    const labels = Object.keys(data.label_distribution || {});
    const counts = Object.values(data.label_distribution || {});

    const ctx = document.getElementById("labelChart").getContext("2d");
    if (labelChartInstance) labelChartInstance.destroy();

    labelChartInstance = new Chart(ctx, {
      type: "doughnut",
      data: {
        labels: labels,
        datasets: [{
          data: counts,
          backgroundColor: [
            "#00f5d4", "#a855f7", "#ff3366", "#38bdf8", "#f59e0b", "#ec4899", "#10b981"
          ],
          borderColor: "#0f172a",
          borderWidth: 2
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "bottom",
            labels: { color: "#cbd5e1", font: { size: 11 } }
          }
        }
      }
    });

    // Populate preview table
    thead.innerHTML = "";
    data.columns.slice(0, 12).forEach(c => {
      const th = document.createElement("th");
      th.innerText = c;
      thead.appendChild(th);
    });

    tbody.innerHTML = "";
    data.records.slice(0, 25).forEach(row => {
      const tr = document.createElement("tr");
      data.columns.slice(0, 12).forEach(c => {
        const td = document.createElement("td");
        td.innerText = row[c] !== undefined ? row[c] : "";
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });

    // Files status list
    const statusRes = await fetch("/api/status");
    const statusData = await statusRes.json();
    if (statusData.dataset_details && statusData.dataset_details.length > 0) {
      fileContainer.innerHTML = statusData.dataset_details.map(f => `
        <div style="background:rgba(255,255,255,0.03); border:1px solid var(--border-subtle); border-radius:var(--radius-sm); padding:10px 14px; margin-bottom:8px; display:flex; justify-content:space-between; align-items:center;">
          <div><i class="fa-solid fa-file-csv" style="color:var(--cyber-cyan); margin-right:8px;"></i> <b>${f.name}</b> <span style="font-size:0.75rem; color:var(--text-muted); margin-left:10px;">${f.path}</span></div>
          <span class="badge-chip" style="background:rgba(56,189,248,0.15); color:var(--cyber-cyan);">${f.size_mb} MB</span>
        </div>
      `).join("");
    }
  } catch (err) {
    console.error("Error in loadDatasetExplorer:", err);
  }
}

// ----------------- PREPROCESSING AUDIT -----------------
async function loadPreprocessingStats() {
  showToast("Running preprocessing audit on dataset...", "info");
  try {
    const res = await fetch("/api/preprocessing/stats?sample=1000");
    const data = await res.json();
    if (data.success && data.report) {
      const r = data.report;
      document.getElementById("clean-orig-rows").innerText = Number(r.original_rows).toLocaleString();
      document.getElementById("clean-clean-rows").innerText = Number(r.clean_rows).toLocaleString();
      document.getElementById("clean-dups-dropped").innerText = Number(r.duplicates_removed).toLocaleString();
      document.getElementById("clean-nulls-handled").innerText = Number(r.missing_values_handled).toLocaleString();
      showToast("Data preprocessing audit complete!", "success");
    }
  } catch (err) {
    showToast("Failed to fetch preprocessing stats.", "error");
  }
}

// ----------------- MODEL TRAINING -----------------
function updateModelOptions() {
  const level = document.getElementById("train-level").value;
  const modelSelect = document.getElementById("train-model-name");
  const regGroup = document.getElementById("reg-target-group");

  modelSelect.innerHTML = "";
  if (level === "regression") {
    regGroup.style.display = "block";
    const models = ["Linear Regression", "Ridge Regression", "Decision Tree Regressor", "Random Forest Regressor"];
    models.forEach(m => {
      const opt = document.createElement("option");
      opt.value = m;
      opt.innerText = m;
      modelSelect.appendChild(opt);
    });
  } else {
    regGroup.style.display = "none";
    const models = ["Random Forest", "Decision Tree", "Logistic Regression", "K-Nearest Neighbors (KNN)", "Support Vector Machine (SVM)"];
    models.forEach(m => {
      const opt = document.createElement("option");
      opt.value = m;
      opt.innerText = m;
      modelSelect.appendChild(opt);
    });
  }
}

async function handleTrainSubmit(event) {
  event.preventDefault();
  const btn = document.getElementById("train-submit-btn");
  const term = document.getElementById("train-terminal");
  const resCard = document.getElementById("train-result-card");
  const metricsBox = document.getElementById("train-metrics-container");

  const level = document.getElementById("train-level").value;
  const modelName = document.getElementById("train-model-name").value;
  const sampleSize = parseInt(document.getElementById("train-sample-size").value);
  const testSplit = parseFloat(document.getElementById("train-test-split").value);
  const targetCol = document.getElementById("train-target-col").value;

  btn.disabled = true;
  btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Training ${modelName}...`;

  term.innerHTML += `\n[${new Date().toLocaleTimeString()}] Initializing training pipeline for ${level.toUpperCase()}...`;
  term.innerHTML += `\n> Algorithm: ${modelName} | Sample Size: ${sampleSize.toLocaleString()} | Test Split: ${(testSplit*100).toFixed(0)}%`;
  term.scrollTop = term.scrollHeight;

  try {
    const res = await fetch("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        level: level,
        model_name: modelName,
        sample_size: sampleSize,
        test_size: testSplit,
        target_col: targetCol
      })
    });

    const data = await res.json();
    if (!data.success) {
      term.innerHTML += `\n❌ ERROR: ${data.error || 'Training failed'}`;
      term.scrollTop = term.scrollHeight;
      showToast(data.error || "Training failed", "error");
      btn.disabled = false;
      btn.innerHTML = `<i class="fa-solid fa-bolt"></i> Start Model Training`;
      return;
    }

    term.innerHTML += `\n✅ Completed in ${data.duration} seconds! Model saved as '${data.saved_model}'.`;
    term.scrollTop = term.scrollHeight;
    showToast(`Successfully trained ${modelName}!`, "success");

    // Display result card
    resCard.style.display = "block";
    metricsBox.innerHTML = "";

    const m = data.metrics;
    if (level === "regression") {
      metricsBox.innerHTML = `
        <div class="stat-card"><div class="stat-info"><div class="stat-value" style="color:var(--amber-reg)">${m['R2-Score']}</div><div class="stat-label">R² Score</div></div></div>
        <div class="stat-card"><div class="stat-info"><div class="stat-value">${m['MAE']}</div><div class="stat-label">MAE</div></div></div>
        <div class="stat-card"><div class="stat-info"><div class="stat-value">${m['RMSE']}</div><div class="stat-label">RMSE</div></div></div>
        <div class="stat-card"><div class="stat-info"><div class="stat-value">${data.duration}s</div><div class="stat-label">Training Duration</div></div></div>
      `;
    } else {
      metricsBox.innerHTML = `
        <div class="stat-card"><div class="stat-info"><div class="stat-value" style="color:var(--cyber-green)">${(m.Accuracy * 100).toFixed(1)}%</div><div class="stat-label">Accuracy</div></div></div>
        <div class="stat-card"><div class="stat-info"><div class="stat-value">${(m['F1-Score'] * 100).toFixed(1)}%</div><div class="stat-label">F1-Score</div></div></div>
        <div class="stat-card"><div class="stat-info"><div class="stat-value">${(m.Precision * 100).toFixed(1)}%</div><div class="stat-label">Precision</div></div></div>
        <div class="stat-card"><div class="stat-info"><div class="stat-value">${data.duration}s</div><div class="stat-label">Duration</div></div></div>
      `;
    }

    refreshSystemStatus();
    loadEvaluationResults();
    updateModelOptions();
    loadAttackSamples();
  } catch (err) {
    term.innerHTML += `\n❌ Network Error: ${err.message}`;
    showToast("Network error during training", "error");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<i class="fa-solid fa-bolt"></i> Start Model Training`;
  }
}

// ----------------- EVALUATION & BENCHMARKS -----------------
function showEvalSubTab(sub) {
  document.getElementById("eval-sub-l1").style.display = sub === "l1" ? "block" : "none";
  document.getElementById("eval-sub-l2").style.display = sub === "l2" ? "block" : "none";
  document.getElementById("eval-sub-reg").style.display = sub === "reg" ? "block" : "none";

  document.getElementById("eval-tab-l1-btn").classList.toggle("active", sub === "l1");
  document.getElementById("eval-tab-l2-btn").classList.toggle("active", sub === "l2");
  document.getElementById("eval-tab-reg-btn").classList.toggle("active", sub === "reg");
}

async function loadEvaluationResults() {
  try {
    const res = await fetch("/api/results");
    const data = await res.json();
    if (!data.success) return;

    // Render tables
    const renderTable = (tbodyId, rows, cols) => {
      const tbody = document.getElementById(tbodyId);
      if (!tbody) return;
      tbody.innerHTML = "";
      if (!rows || rows.length === 0) {
        tbody.innerHTML = `<tr><td colspan="${cols.length}" style="text-align:center; color:var(--text-muted)">No models evaluated yet.</td></tr>`;
        return;
      }
      rows.forEach(r => {
        const tr = document.createElement("tr");
        cols.forEach(c => {
          const td = document.createElement("td");
          td.innerText = r[c] !== undefined ? r[c] : "-";
          tr.appendChild(td);
        });
        tbody.appendChild(tr);
      });
    };

    renderTable("table-eval-l1", data.metrics_tables.level1, ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "Training_Time_s"]);
    renderTable("table-eval-l2", data.metrics_tables.level2, ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "Training_Time_s"]);
    renderTable("table-eval-reg", data.metrics_tables.regression, ["Model", "Target_Column", "MAE", "MSE", "RMSE", "R2-Score", "Training_Time_s"]);

    // Render plot cards
    const gallery = document.getElementById("plots-gallery");
    if (gallery) {
      gallery.innerHTML = "";
      if (data.plots && data.plots.length > 0) {
        data.plots.forEach(p => {
          const card = document.createElement("div");
          card.style.background = "rgba(15,23,42,0.7)";
          card.style.border = "1px solid var(--border-subtle)";
          card.style.borderRadius = "var(--radius-md)";
          card.style.padding = "14px";
          card.style.cursor = "pointer";
          card.innerHTML = `
            <div style="font-size:0.82rem; font-weight:600; margin-bottom:8px; color:#cbd5e1; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${p.name}</div>
            <img src="${p.url}" alt="${p.name}" style="width:100%; height:160px; object-fit:cover; border-radius:var(--radius-sm);">
          `;
          card.addEventListener("click", () => openImageModal(p.url, p.name));
          gallery.appendChild(card);
        });
      } else {
        gallery.innerHTML = `<p style="color:var(--text-muted)">No plots generated yet. Train models to generate confusion matrices and regression charts.</p>`;
      }
    }
  } catch (err) {
    console.error("Evaluation results error:", err);
  }
}

function openImageModal(src, title) {
  document.getElementById("modal-img").src = src;
  document.getElementById("modal-title").innerText = title;
  document.getElementById("image-modal").classList.add("active");
}

function closeImageModal() {
  document.getElementById("image-modal").classList.remove("active");
}

// ----------------- PREDICTIONS & RADAR -----------------
function togglePredictionMode(mode) {
  document.getElementById("pred-single-container").style.display = mode === "single" ? "block" : "none";
  document.getElementById("pred-batch-container").style.display = mode === "batch" ? "block" : "none";

  document.getElementById("pred-mode-single-btn").className = mode === "single" ? "btn-cyber" : "btn-secondary";
  document.getElementById("pred-mode-batch-btn").className = mode === "batch" ? "btn-cyber" : "btn-secondary";
}

// ----------------- PREDEFINED ATTACK SAMPLES & TESTING -----------------
async function loadAttackSamples() {
  const select = document.getElementById("pred-attack-sample-select");
  if (!select) return;

  try {
    const res = await fetch("/api/attack-samples");
    const data = await res.json();
    if (data.success && data.attack_types && data.attack_types.length > 0) {
      cachedAttackSamples = data.samples || {};
      select.innerHTML = '<option value="">-- Select Predefined Attack Type to Populate Sample Values --</option>';

      data.attack_types.forEach(atk => {
        const opt = document.createElement("option");
        opt.value = atk;
        const icon = atk === "BENIGN" ? "🟢" : 
                     (atk.toLowerCase().includes("dos") || atk.toLowerCase().includes("ddos") ? "🌊" : 
                     (atk.toLowerCase().includes("patator") ? "🚨" : 
                     (atk.toLowerCase().includes("web") ? "💉" : 
                     (atk.toLowerCase().includes("bot") ? "🤖" : 
                     (atk.toLowerCase().includes("heartbleed") ? "💔" : "⚠️")))));
        opt.innerText = `${icon} ${atk}`;
        select.appendChild(opt);
      });
    } else {
      select.innerHTML = '<option value="">-- No Attack Types Found in Dataset --</option>';
    }
  } catch (err) {
    console.error("Failed to load attack samples:", err);
    if (select) select.innerHTML = '<option value="">-- Error Loading Attack Types --</option>';
  }
}

function onAttackSampleSelected(attackType) {
  if (!attackType || !cachedAttackSamples[attackType]) {
    return;
  }

  const sample = cachedAttackSamples[attackType];
  const ui = sample.ui_values || {};

  // Store full sample features in order
  currentFlowFeatures = Object.assign({}, sample.full_values || {});

  // Populate all existing input fields with realistic sample feature values
  const fields = [
    { id: "feat-port", key: "Destination Port", fallback: 80 },
    { id: "feat-duration", key: "Flow Duration", fallback: 100000 },
    { id: "feat-fwd-pkts", key: "Total Fwd Packets", fallback: 10 },
    { id: "feat-bwd-pkts", key: "Total Backward Packets", fallback: 8 },
    { id: "feat-fwd-len", key: "Total Length of Fwd Packets", fallback: 500 },
    { id: "feat-bwd-len", key: "Total Length of Bwd Packets", fallback: 800 },
    { id: "feat-bytes-sec", key: "Flow Bytes/s", fallback: 13000 },
    { id: "feat-pkts-sec", key: "Flow Packets/s", fallback: 180 },
    { id: "feat-syn-count", key: "SYN Flag Count", fallback: 0 }
  ];

  fields.forEach(f => {
    const el = document.getElementById(f.id);
    if (el) {
      const val = (ui[f.id] !== undefined) ? ui[f.id] : ((ui[f.key] !== undefined) ? ui[f.key] : f.fallback);
      el.value = val;
      currentFlowFeatures[f.key] = parseFloat(val);
    }
  });

  showToast(`Populated realistic sample values for: ${attackType}`, "info");
}

function clearManualFields() {
  const ids = [
    "feat-port", "feat-duration", "feat-fwd-pkts", "feat-bwd-pkts",
    "feat-fwd-len", "feat-bwd-len", "feat-bytes-sec", "feat-pkts-sec",
    "feat-syn-count"
  ];
  ids.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.value = "";
  });
  const sel = document.getElementById("pred-attack-sample-select");
  if (sel) sel.value = "";
  currentFlowFeatures = {};

  const verdictContainer = document.getElementById("verdict-container");
  if (verdictContainer) verdictContainer.style.display = "none";

  showToast("All input fields cleared for custom manual data entry.", "info");
}

function resetManualFields() {
  if (cachedAttackSamples["BENIGN"]) {
    const sel = document.getElementById("pred-attack-sample-select");
    if (sel) sel.value = "BENIGN";
    onAttackSampleSelected("BENIGN");
    showToast("Reset fields to baseline BENIGN traffic sample.", "info");
    return;
  }
  
  const defaults = {
    "feat-port": 80,
    "feat-duration": 100000,
    "feat-fwd-pkts": 10,
    "feat-bwd-pkts": 8,
    "feat-fwd-len": 500,
    "feat-bwd-len": 800,
    "feat-bytes-sec": 13000,
    "feat-pkts-sec": 180,
    "feat-syn-count": 1
  };
  for (const [id, val] of Object.entries(defaults)) {
    const el = document.getElementById(id);
    if (el) el.value = val;
  }
  const sel = document.getElementById("pred-attack-sample-select");
  if (sel) sel.value = "";
  currentFlowFeatures = {};
  showToast("Fields reset to baseline network flow defaults.", "info");
}

function loadPreset(preset) {
  // Backward compatibility alias for any preset trigger
  const map = {
    benign: "BENIGN",
    ddos: "DDoS",
    ssh: "SSH-Patator",
    portscan: "PortScan"
  };
  const target = map[preset.toLowerCase()] || preset;
  const sel = document.getElementById("pred-attack-sample-select");
  if (sel && cachedAttackSamples[target]) {
    sel.value = target;
    onAttackSampleSelected(target);
  } else {
    onAttackSampleSelected(target);
  }
}

async function analyzeSingleFlow() {
  const verdictCard = document.getElementById("verdict-card");
  const verdictContainer = document.getElementById("verdict-container");

  // Read current input values directly from manual input fields
  const port = parseFloat(document.getElementById("feat-port").value);
  const duration = parseFloat(document.getElementById("feat-duration").value);
  const fwdPkts = parseFloat(document.getElementById("feat-fwd-pkts").value);
  const bwdPkts = parseFloat(document.getElementById("feat-bwd-pkts").value);
  const fwdLen = parseFloat(document.getElementById("feat-fwd-len").value);
  const bwdLen = parseFloat(document.getElementById("feat-bwd-len").value);
  const bytesSec = parseFloat(document.getElementById("feat-bytes-sec").value);
  const pktsSec = parseFloat(document.getElementById("feat-pkts-sec").value);
  const syn = parseFloat(document.getElementById("feat-syn-count").value);

  // Fallback defaults for missing or invalid inputs
  const manualFeatures = {
    "Destination Port": isNaN(port) ? 80 : port,
    "Flow Duration": isNaN(duration) ? 100000 : duration,
    "Total Fwd Packets": isNaN(fwdPkts) ? 1 : fwdPkts,
    "Total Backward Packets": isNaN(bwdPkts) ? 0 : bwdPkts,
    "Total Length of Fwd Packets": isNaN(fwdLen) ? 0 : fwdLen,
    "Total Length of Bwd Packets": isNaN(bwdLen) ? 0 : bwdLen,
    "Flow Bytes/s": isNaN(bytesSec) ? 0 : bytesSec,
    "Flow Packets/s": isNaN(pktsSec) ? 0 : pktsSec,
    "SYN Flag Count": isNaN(syn) ? 0 : syn
  };

  // Combine with full features from current sample if selected, letting user's manual edits override
  const combinedFeatures = Object.assign({}, currentFlowFeatures, manualFeatures);

  const payload = {
    features: combinedFeatures,
    l1_model: document.getElementById("pred-l1-select").value || null,
    l2_model: document.getElementById("pred-l2-select").value || null,
    reg_model: document.getElementById("pred-reg-select").value || null
  };

  showToast("Scanning connection flow characteristics...", "info");

  try {
    const res = await fetch("/api/predict/single", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const data = await res.json();
    if (!data.success) {
      showToast(data.error || "Inference error", "error");
      return;
    }

    const p = data.prediction;
    const isBenign = p.level1 === "BENIGN";

    verdictContainer.style.display = "block";
    verdictCard.className = `threat-verdict-card ${isBenign ? 'benign' : 'attack'}`;

    // Verdict column 1: Level 1 Triage
    const icon = document.getElementById("verdict-icon");
    icon.className = `verdict-icon ${isBenign ? 'benign' : 'attack'}`;
    icon.innerHTML = isBenign ? `<i class="fa-solid fa-shield-check"></i>` : `<i class="fa-solid fa-skull-crossbones"></i>`;
    document.getElementById("verdict-l1-text").innerText = p.level1;
    document.getElementById("verdict-l1-badge").innerText = isBenign ? "Normal Traffic (Secure)" : "Malicious Activity Detected";

    // Verdict column 2: Level 2 Attack Classification & Confidence
    document.getElementById("verdict-l2-type").innerText = p.attack_type;
    const confVal = Math.min(100, Math.max(1, p.confidence));
    document.getElementById("verdict-conf-text").innerText = `${confVal.toFixed(1)}%`;
    const confBar = document.getElementById("verdict-conf-bar");
    confBar.className = `confidence-fill ${isBenign ? 'benign' : 'attack'}`;
    confBar.style.width = `${confVal}%`;

    // Verdict column 3: Level 3 Regression
    if (p.regression_val !== null) {
      document.getElementById("verdict-reg-title").innerText = `Predicted ${p.regression_target}`;
      document.getElementById("verdict-reg-val").innerText = `${Number(p.regression_val).toLocaleString()} μs`;
    } else {
      document.getElementById("verdict-reg-val").innerText = "N/A";
    }

    showToast(`Flow Analysis Verdict: ${p.level1} | Threat: ${p.attack_type} (${confVal.toFixed(1)}% confidence)`, isBenign ? "success" : "error");
  } catch (err) {
    showToast("Error executing threat prediction", "error");
  }
}

// ----------------- BATCH CSV UPLOAD -----------------
function setupDropzone() {
  const dropzone = document.getElementById("csv-dropzone");
  if (!dropzone) return;

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFiles(files[0]);
    }
  });
}

function handleFileSelected(event) {
  if (event.target.files.length > 0) {
    handleFiles(event.target.files[0]);
  }
}

function handleFiles(file) {
  if (!file.name.endsWith(".csv")) {
    showToast("Please upload a valid .csv file", "error");
    return;
  }
  selectedBatchFile = file;
  document.getElementById("file-chosen-name").innerText = `Selected File: ${file.name} (${(file.size / (1024*1024)).toFixed(2)} MB)`;
  document.getElementById("batch-analyze-btn").disabled = false;
  showToast(`File ${file.name} ready for analysis.`, "info");
}

async function executeBatchAnalysis() {
  if (!selectedBatchFile) return;

  const btn = document.getElementById("batch-analyze-btn");
  btn.disabled = true;
  btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Processing Batch Traffic Streams...`;

  const formData = new FormData();
  formData.append("file", selectedBatchFile);
  formData.append("l1_model", document.getElementById("pred-l1-select").value || "");
  formData.append("l2_model", document.getElementById("pred-l2-select").value || "");
  formData.append("reg_model", document.getElementById("pred-reg-select").value || "");

  try {
    const res = await fetch("/api/predict/batch", {
      method: "POST",
      body: formData
    });

    const data = await res.json();
    if (!data.success) {
      showToast(data.error || "Batch prediction failed", "error");
      btn.disabled = false;
      btn.innerHTML = `<i class="fa-solid fa-play"></i> Run Batch Security Analysis`;
      return;
    }

    // Populate batch results
    document.getElementById("batch-results-panel").style.display = "block";
    document.getElementById("batch-total-rows").innerText = Number(data.total_rows).toLocaleString();
    document.getElementById("batch-benign-count").innerText = Number(data.benign_count).toLocaleString();
    document.getElementById("batch-attack-count").innerText = Number(data.attack_count).toLocaleString();
    document.getElementById("batch-avg-conf").innerText = `${data.avg_confidence}%`;

    // Download link
    const dlLink = document.getElementById("batch-download-link");
    dlLink.href = `/api/predict/download/${data.download_filename}`;

    // Table preview
    const tbody = document.querySelector("#batch-preview-table tbody");
    tbody.innerHTML = "";
    data.preview.forEach(row => {
      const isBenign = row.Level_1_Prediction === "BENIGN";
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><span class="badge-chip" style="background:${isBenign ? 'rgba(0,245,212,0.15)' : 'rgba(255,51,102,0.15)'}; color:${isBenign ? 'var(--cyber-green)' : 'var(--threat-red)'}">${row.Level_1_Prediction}</span></td>
        <td>${row['Level_1_Confidence_%']}%</td>
        <td><b>${row.Predicted_Attack_Type}</b></td>
        <td>${row['Attack_Confidence_%']}%</td>
        <td>${row[`Predicted_${data.regression_target}`] !== undefined ? row[`Predicted_${data.regression_target}`] : '-'}</td>
      `;
      tbody.appendChild(tr);
    });

    showToast(`Batch Analysis complete! ${data.attack_count} attacks identified.`, "success");
  } catch (err) {
    showToast("Error processing batch upload", "error");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<i class="fa-solid fa-play"></i> Run Batch Security Analysis`;
  }
}
