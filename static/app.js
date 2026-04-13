const form = document.getElementById("prediction-form");
const panel = document.getElementById("result-panel");
const resultStatus = document.getElementById("result-status");
const resultSummary = document.getElementById("result-summary");
const resultProbability = document.getElementById("result-probability");
const resultConfidence = document.getElementById("result-confidence");
const resultRisk = document.getElementById("result-risk");
const resultFactors = document.getElementById("result-factors");
const resultMeter = document.getElementById("result-meter");

function populateForm(values) {
  Object.entries(values).forEach(([key, value]) => {
    const field = form.elements.namedItem(key);
    if (field) {
      field.value = value;
    }
  });
}

function renderResult(payload) {
  const risky = payload.risk_level !== "LOW";
  panel.classList.toggle("risky", risky);
  panel.classList.toggle("safe", !risky);

  resultStatus.textContent = `${payload.risk_level} Risk`;
  resultSummary.textContent = payload.message;
  resultProbability.textContent = `${payload.fraud_score}%`;
  resultConfidence.textContent = `${payload.confidence}%`;
  resultRisk.textContent = payload.risk_level;
  resultMeter.style.width = `${payload.fraud_score}%`;

  resultFactors.innerHTML = "";
  payload.reasons.forEach((factor) => {
    const item = document.createElement("li");
    item.textContent = factor;
    resultFactors.appendChild(item);
  });
}

async function submitPrediction(event) {
  event.preventDefault();
  const data = Object.fromEntries(new FormData(form).entries());

  resultStatus.textContent = "Analyzing...";
  resultSummary.textContent = "Running the transaction through the fraud classifier.";

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
    panel.classList.remove("safe");
    panel.classList.add("risky");
    resultStatus.textContent = "Input error";
    resultSummary.textContent = error.message;
    resultProbability.textContent = "--";
    resultConfidence.textContent = "--";
    resultRisk.textContent = "--";
    resultMeter.style.width = "0%";
    resultFactors.innerHTML = "<li>Please correct the form values and try again.</li>";
  }
}

form.addEventListener("submit", submitPrediction);

document.querySelectorAll("[data-scenario]").forEach((button) => {
  button.addEventListener("click", () => {
    const key = button.dataset.scenario;
    const scenario = window.APP_DATA[key];
    if (!scenario) return;
    populateForm(scenario.values);
    form.requestSubmit();
  });
});

form.addEventListener("reset", () => {
  window.setTimeout(() => {
    panel.classList.remove("safe", "risky");
    resultStatus.textContent = "Waiting for analysis";
    resultSummary.textContent =
      "Submit the transaction to see fraud probability, confidence, and the top reasons behind the decision.";
    resultProbability.textContent = "--";
    resultConfidence.textContent = "--";
    resultRisk.textContent = "--";
    resultMeter.style.width = "0%";
    resultFactors.innerHTML =
      "<li>Model explanation will appear here after prediction.</li>";
  }, 0);
});
