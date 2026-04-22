const form = document.getElementById("prediction-form");
const panel = document.getElementById("result-panel");
const resultStatus = document.getElementById("result-status");
const resultSummary = document.getElementById("result-summary");
const resultProbability = document.getElementById("result-probability");
const resultConfidence = document.getElementById("result-confidence");
const resultRisk = document.getElementById("result-risk");
const resultFactors = document.getElementById("result-factors");
const resultMeter = document.getElementById("result-meter");
const resultBadge = document.getElementById("result-badge");
const sidebarToggle = document.getElementById("sidebar-toggle");
const mobileSidebarOpen = document.getElementById("mobile-sidebar-open");
const mobileSidebarBackdrop = document.getElementById("mobile-sidebar-backdrop");
const body = document.body;
const DESKTOP_SIDEBAR_BREAKPOINT = 900;

function syncSidebarState(collapsed) {
  body.classList.toggle("sidebar-collapsed", collapsed);
  if (sidebarToggle) {
    sidebarToggle.setAttribute("aria-expanded", String(!collapsed));
  }
}

function syncMobileSidebarState(open) {
  body.classList.toggle("mobile-nav-open", open);
  if (mobileSidebarOpen) {
    mobileSidebarOpen.setAttribute("aria-expanded", String(open));
  }
  if (mobileSidebarBackdrop) {
    mobileSidebarBackdrop.hidden = !open;
  }
}

function animateValue(element, target, options = {}) {
  if (!element || Number.isNaN(Number(target))) return;

  const decimals = Number(options.decimals || 0);
  const prefix = options.prefix || "";
  const suffix = options.suffix || "";
  const duration = options.duration || 700;
  const startTime = performance.now();
  const start = 0;

  function frame(now) {
    const progress = Math.min((now - startTime) / duration, 1);
    const eased = 1 - (1 - progress) ** 3;
    const value = start + (Number(target) - start) * eased;
    element.textContent = `${prefix}${value.toFixed(decimals)}${suffix}`;
    if (progress < 1) {
      requestAnimationFrame(frame);
    }
  }

  requestAnimationFrame(frame);
}

function animateStaticCounters() {
  document.querySelectorAll(".count-up").forEach((element) => {
    const target = Number(element.dataset.count);
    const decimals = Number(element.dataset.decimals || 0);
    const prefix = element.dataset.prefix || "";
    const suffix = element.dataset.suffix || "";
    animateValue(element, target, { decimals, prefix, suffix, duration: 900 });
  });
}

function populateForm(values) {
  Object.entries(values).forEach(([key, value]) => {
    const field = form.elements.namedItem(key);
    if (field) {
      field.value = value;
    }
  });
}

function setRiskClasses(level) {
  if (!panel || !resultBadge) return;
  panel.classList.remove("risk-low", "risk-medium", "risk-high", "loading");
  resultBadge.classList.remove("low", "medium", "high", "neutral");

  if (level === "LOW") {
    panel.classList.add("risk-low");
    resultBadge.classList.add("low");
  } else if (level === "MEDIUM") {
    panel.classList.add("risk-medium");
    resultBadge.classList.add("medium");
  } else if (level === "HIGH") {
    panel.classList.add("risk-high");
    resultBadge.classList.add("high");
  } else {
    resultBadge.classList.add("neutral");
  }
}

function renderResult(payload) {
  setRiskClasses(payload.risk_level);

  resultStatus.textContent = `${payload.risk_level} Risk`;
  resultSummary.textContent = payload.message;
  resultRisk.textContent = payload.risk_level;
  resultBadge.textContent = `${payload.risk_level} Signal`;

  animateValue(resultProbability, payload.fraud_score, { decimals: 2, suffix: "%" });
  animateValue(resultConfidence, payload.confidence, { decimals: 2, suffix: "%" });
  resultMeter.style.width = `${payload.fraud_score}%`;

  resultFactors.innerHTML = "";
  payload.reasons.forEach((factor) => {
    const item = document.createElement("li");
    item.textContent = factor;
    resultFactors.appendChild(item);
  });
}

function showLoadingState() {
  if (!panel) return;
  panel.classList.add("loading");
  resultStatus.textContent = "Analyzing transaction";
  resultSummary.textContent = "Running the transaction through the hybrid fraud engine.";
  resultProbability.textContent = "--";
  resultConfidence.textContent = "--";
  resultRisk.textContent = "--";
  resultBadge.textContent = "Processing";
  resultFactors.innerHTML = [
    "<li>Preparing behavioral features...</li>",
    "<li>Running global anomaly model...</li>",
  ].join("");
}

function showErrorState(message) {
  setRiskClasses("HIGH");
  resultStatus.textContent = "Input error";
  resultSummary.textContent = message;
  resultProbability.textContent = "--";
  resultConfidence.textContent = "--";
  resultRisk.textContent = "--";
  resultBadge.textContent = "Check input";
  resultMeter.style.width = "0%";
  resultFactors.innerHTML = "<li>Please correct the form values and try again.</li>";
}

async function submitPrediction(event) {
  event.preventDefault();
  const data = Object.fromEntries(new FormData(form).entries());

  showLoadingState();

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    const payload = await response.json();

    if (!response.ok || payload.status !== "ok") {
      throw new Error(payload.message || "Prediction failed");
    }

    renderResult(payload);
  } catch (error) {
    showErrorState(error.message);
  }
}

function resetPredictionPanel() {
  if (!panel) return;
  panel.classList.remove("risk-low", "risk-medium", "risk-high", "loading");
  resultStatus.textContent = "Ready for analysis";
  resultSummary.textContent =
    "Run a prediction to display the fraud score, model confidence, and the strongest reasons behind the final decision.";
  resultProbability.textContent = "--";
  resultConfidence.textContent = "--";
  resultRisk.textContent = "--";
  resultBadge.textContent = "Awaiting input";
  resultBadge.className = "risk-badge neutral";
  resultMeter.style.width = "0%";
  resultFactors.innerHTML = "<li>Explanation factors will appear here after prediction.</li>";
}

function initSidebar() {
  if (!sidebarToggle) return;

  function applyResponsiveSidebarState() {
    const saved = window.localStorage.getItem("sidebar-collapsed");
    const allowCollapsed = window.innerWidth > DESKTOP_SIDEBAR_BREAKPOINT;
    const shouldCollapse = allowCollapsed && saved === "true";
    syncSidebarState(shouldCollapse);
    if (allowCollapsed) {
      syncMobileSidebarState(false);
    }
  }

  applyResponsiveSidebarState();

  sidebarToggle.addEventListener("click", () => {
    if (window.innerWidth <= DESKTOP_SIDEBAR_BREAKPOINT) {
      syncMobileSidebarState(false);
      return;
    }
    const next = !body.classList.contains("sidebar-collapsed");
    syncSidebarState(next);
    window.localStorage.setItem("sidebar-collapsed", String(next));
  });

  if (mobileSidebarOpen) {
    mobileSidebarOpen.addEventListener("click", () => {
      if (window.innerWidth > DESKTOP_SIDEBAR_BREAKPOINT) return;
      const next = !body.classList.contains("mobile-nav-open");
      syncMobileSidebarState(next);
    });
  }

  if (mobileSidebarBackdrop) {
    mobileSidebarBackdrop.addEventListener("click", () => {
      syncMobileSidebarState(false);
    });
  }

  document.querySelectorAll(".sidebar-link").forEach((link) => {
    link.addEventListener("click", () => {
      if (window.innerWidth <= DESKTOP_SIDEBAR_BREAKPOINT) {
        syncMobileSidebarState(false);
      }
    });
  });

  window.addEventListener("resize", applyResponsiveSidebarState);
}

if (form) {
  form.addEventListener("submit", submitPrediction);

  document.querySelectorAll("[data-scenario]").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll("[data-scenario]").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      const key = button.dataset.scenario;
      const scenario = window.APP_DATA?.[key];
      if (!scenario) return;
      populateForm(scenario.values);
      form.requestSubmit();
    });
  });

  form.addEventListener("reset", () => {
    window.setTimeout(() => {
      document.querySelectorAll("[data-scenario]").forEach((item) => item.classList.remove("active"));
      resetPredictionPanel();
    }, 0);
  });
}

document.addEventListener("DOMContentLoaded", () => {
  animateStaticCounters();
  initSidebar();
});
