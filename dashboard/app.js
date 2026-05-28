let events = [];
let selectedView = "overview";

const severityFilter = document.querySelector("#severityFilter");
const timeline = document.querySelector("#timeline");
const mapCanvas = document.querySelector("#mapCanvas");
const riskBar = document.querySelector("#riskBar");
const aiSummary = document.querySelector("#aiSummary");
const summaryTime = document.querySelector("#summaryTime");
const autoRefreshToggle = document.querySelector("#autoRefresh");

const scenarioHints = [
  { scene_hint: "overflowing bin near market entrance", audio_anomaly_score: 0.15, crowd_density: 0.3, thermal_hotspot_score: 0.1, weather_risk: 0.25 },
  { scene_hint: "vehicle stopped near roadside drain at night dumping bags", lighting: "night", audio_anomaly_score: 0.25, crowd_density: 0.2, thermal_hotspot_score: 0.15, weather_risk: 0.2 },
  { scene_hint: "smoke and fire near municipal waste accumulation area with crowd nearby", lighting: "night", audio_anomaly_score: 0.7, crowd_density: 0.75, thermal_hotspot_score: 0.95, weather_risk: 0.25 },
  { scene_hint: "plastic garbage blocking drain stagnant waterlogging", lighting: "day", audio_anomaly_score: 0.2, crowd_density: 0.25, thermal_hotspot_score: 0.2, weather_risk: 0.7 },
  { scene_hint: "possible toxic spill in industrial waste area with crowd nearby", lighting: "day", audio_anomaly_score: 0.5, crowd_density: 0.6, thermal_hotspot_score: 0.35, weather_risk: 0.4 },
];

document.querySelectorAll(".nav button").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".nav button").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    selectedView = button.dataset.view;
    document.querySelector("#viewTitle").textContent = button.textContent;
    render();
  });
});

severityFilter.addEventListener("change", render);

document.querySelector("#injectEvent").addEventListener("click", async () => {
  await injectRandomScenario();
});

async function injectRandomScenario() {
  const payload = scenarioHints[Math.floor(Math.random() * scenarioHints.length)];
  try {
    await fetch("/simulate/frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (err) {
    console.error("simulate/frame failed:", err);
  }
}

function filteredEvents() {
  const severity = severityFilter.value;
  return severity === "all" ? events : events.filter((event) => event.severity === severity);
}

async function refreshFromApi() {
  try {
    const eventsRes = await fetch("/events");
    const eventsData = await eventsRes.json();
    events = eventsData.events || [];
  } catch (err) {
    console.error("Failed to refresh from API:", err);
  }
  render();
}

let autoTimer = null;
function startAutoRefresh() {
  stopAutoRefresh();
  autoTimer = setInterval(() => {
    if (!autoRefreshToggle.checked) return;
    refreshFromApi();
  }, 2000);
}

function stopAutoRefresh() {
  if (autoTimer) clearInterval(autoTimer);
  autoTimer = null;
}

autoRefreshToggle.addEventListener("change", () => {
  startAutoRefresh();
});

function render() {
  const current = filteredEvents();
  const critical = current.filter((event) => event.environmental_risk_level === "critical");
  const dispatch = current.filter((event) => event.escalation_status === "auto_dispatch");

  document.querySelector("#activeEvents").textContent = current.length;
  document.querySelector("#criticalEvents").textContent = critical.length;
  document.querySelector("#dispatchEvents").textContent = dispatch.length;
  document.querySelector("#hotspotScore").textContent = Math.min(99, current.length * 16 + critical.length * 12);

  renderMap(current);
  renderTimeline(current);
  renderSummary(current);
}

function renderMap(current) {
  mapCanvas.querySelectorAll(".pin").forEach((pin) => pin.remove());
  current.forEach((event, index) => {
    const pin = document.createElement("div");
    pin.className = `pin ${severityClass(event.severity)}`;
    pin.style.left = `${14 + ((event.gps.longitude * 10000 + index * 17) % 72)}%`;
    pin.style.top = `${12 + ((event.gps.latitude * 10000 + index * 23) % 68)}%`;
    pin.innerHTML = `<span>${event.scene_description}<br>${event.drone_id} | ${(event.confidence * 100).toFixed(0)}%</span>`;
    mapCanvas.appendChild(pin);
  });
}

function renderTimeline(current) {
  timeline.innerHTML = "";
  current.forEach((event) => {
    const row = document.createElement("article");
    row.className = "event-row";
    row.innerHTML = `
      <div>
        <strong>${formatTime(event.timestamp)}</strong>
        <p>${event.drone_id} at ${event.drone_altitude_m}m</p>
      </div>
      <div>
        <strong>${humanize(event.event_type)}</strong>
        <p>${event.scene_description}</p>
      </div>
      <span class="badge ${severityClass(event.severity)}">${event.severity.replace(" Trigger", "")}</span>
      <div>
        <strong>${event.environmental_risk_level}</strong>
        <p>${event.escalation_status}</p>
      </div>
    `;
    timeline.appendChild(row);
  });
}

function renderSummary(current) {
  const major = current.filter((event) => event.severity === "Major Trigger").length;
  const moderate = current.filter((event) => event.severity === "Moderate Trigger").length;
  const repeated = current.filter((event) => event.historical_relevance_score >= 0.7).length;
  const riskScore = Math.min(100, major * 34 + moderate * 16 + repeated * 8);

  riskBar.style.width = `${riskScore}%`;
  summaryTime.textContent = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  aiSummary.textContent =
    current.length === 0
      ? "No active incidents match the current filter."
      : `${current.length} events are active. ${major} major triggers require immediate command attention, ${moderate} moderate triggers remain on watch, and ${repeated} incidents have historical relevance in their GPS zones.`;
}

function severityClass(severity) {
  if (severity === "Major Trigger") return "major";
  if (severity === "Moderate Trigger") return "moderate";
  return "minor";
}

function humanize(value) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function formatTime(value) {
  return new Date(value).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

refreshFromApi().then(async () => {
  if (events.length === 0) {
    await injectRandomScenario();
    await refreshFromApi();
  }
});
startAutoRefresh();
