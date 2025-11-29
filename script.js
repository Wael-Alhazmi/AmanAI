let map;
let trafficLayer = null;
let incidentsLayer = null;
let heatLayer = null;
let patrolLayer = null;

const TOMTOM_KEY = "OLiJPFwlldEw398ZSUmRCKuAkUn3lLEb";

// رابط الخادم الأساسي (Render)
const API = "https://amanai-26b5.onrender.com";

/* ===========================
   INIT
=========================== */
window.onload = () => {
    setupThemeToggle();
    initMap();
    loadIncidents();
    updateDashboardStats();
    loadHeatmap();
};

/* ===========================
   THEME TOGGLE
=========================== */
function setupThemeToggle() {
    const btn = document.getElementById("themeToggle");
    if (!btn) return;

    const saved = localStorage.getItem("amanai-theme");

    if (saved === "dark") {
        document.body.classList.add("dark-mode");
        btn.textContent = "☀️ وضع النهار";
    } else {
        document.body.classList.remove("dark-mode");
        btn.textContent = "🌙 الوضع الليلي";
    }

    btn.addEventListener("click", () => {
        const isDark = document.body.classList.toggle("dark-mode");
        btn.textContent = isDark ? "☀️ وضع النهار" : "🌙 الوضع الليلي";
        localStorage.setItem("amanai-theme", isDark ? "dark" : "light");
    });
}

/* ===========================
   MAP INIT
=========================== */
function initMap() {
    map = L.map("map").setView([24.47, 39.61], 13);

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 19
    }).addTo(map);
}

function showMapMessage(msg) {
    const box = document.getElementById("map-notify");
    if (!box) return;
    box.innerHTML = msg;
    box.style.opacity = 1;
    setTimeout(() => (box.style.opacity = 0), 2600);
}

/* ===========================
   🔥 تحليل الازدحام (محاكاة)
=========================== */
function detectTrafficAutomatically() {
    fetch(`${API}/detect-traffic`)
        .then(r => r.json())
        .then(res => {
            showMapMessage("🔥 " + res.msg);
            loadIncidents();
            updateDashboardStats();
            loadHeatmap();
        })
        .catch(() => showMapMessage("⚠ فشل الاتصال بخادم التحليل"));
}

/* ===========================
   🚦 طبقة المرور من TomTom
=========================== */
function toggleTrafficLayer() {
    if (trafficLayer) {
        map.removeLayer(trafficLayer);
        trafficLayer = null;
        showMapMessage("❌ تم إخفاء طبقة المرور");
        return;
    }

    const url = `https://api.tomtom.com/traffic/map/4/tile/flow/absolute/{z}/{x}/{y}.png?key=${TOMTOM_KEY}`;

    trafficLayer = L.tileLayer(url, {
        opacity: 0.8,
        maxZoom: 18,
        crossOrigin: true
    });

    trafficLayer.on("tileerror", () => {
        showMapMessage("⚠ فشل تحميل طبقة المرور من TomTom");
    });

    trafficLayer.addTo(map);
    showMapMessage("🚦 تم تفعيل طبقة المرور (TomTom)");
}

/* ===========================
   🔴 طبقة الحوادث من TomTom
=========================== */
function toggleIncidentsLayer() {
    if (incidentsLayer) {
        map.removeLayer(incidentsLayer);
        incidentsLayer = null;
        showMapMessage("❌ تم إخفاء طبقة الحوادث");
        return;
    }

    const bbox = "24.40,39.50,24.55,39.70";
    const url =
        `https://api.tomtom.com/traffic/services/5/incidentDetails?bbox=${bbox}&fields=incidents&key=${TOMTOM_KEY}`;

    fetch(url)
        .then(r => r.json())
        .then(res => {
            incidentsLayer = L.layerGroup();

            if (!res.incidents || res.incidents.length === 0) {
                showMapMessage("🔴 لا توجد حوادث من TomTom حاليًا");
                return;
            }

            res.incidents.forEach(inc => {
                const p = inc.point;
                const loc = [p.latitude, p.longitude];

                L.circle(loc, {
                    radius: 90,
                    color: "#d00000",
                    fillColor: "#d00000",
                    fillOpacity: 0.45
                })
                .bindPopup(`
                    <strong>حادث مروري</strong><br>
                    ${inc.properties?.description || ""}
                `)
                .addTo(incidentsLayer);
            });

            incidentsLayer.addTo(map);
            showMapMessage("🔴 تم عرض طبقة الحوادث من TomTom");
        })
        .catch(() => showMapMessage("⚠ فشل جلب الحوادث من TomTom"));
}

/* ===========================
   ⭐ الخريطة الحرارية
=========================== */
function loadHeatmap() {
    fetch(`${API}/heatmap`)
        .then(r => r.json())
        .then(data => {
            if (heatLayer) map.removeLayer(heatLayer);
            heatLayer = L.layerGroup();

            data.points.forEach(p => {
                let levelClass = "pulse-low";
                if (p.weight === 2) levelClass = "pulse-med";
                if (p.weight === 3) levelClass = "pulse-high";

                const icon = L.divIcon({
                    className: `pulse-marker ${levelClass}`,
                    iconSize: [22, 22]
                });

                L.marker([p.lat, p.lng], { icon })
                 .addTo(heatLayer)
                 .bindPopup(
                    p.weight === 3 ? "🔴 خطورة مرتفعة" :
                    p.weight === 2 ? "🟠 خطورة متوسطة" :
                                      "🟢 خطورة منخفضة"
                 );
            });

            if (data.points.length > 0) {
                heatLayer.addTo(map);
            }
        })
        .catch(() => console.warn("Heatmap fetch error"));
}

/* ===========================
   🚔 تمركز الدوريات – نبضة Pulse
=========================== */
function forecastPatrolZones() {
    fetch(`${API}/patrol-forecast`)
        .then(r => r.json())
        .then(zones => {

            if (patrolLayer) map.removeLayer(patrolLayer);
            patrolLayer = L.layerGroup();

            zones.forEach(z => {

                L.circle([z.lat, z.lng], {
                    radius: 160,
                    color: "#0ea5e9",
                    fillColor: "#0ea5e9",
                    fillOpacity: 0.18
                }).addTo(patrolLayer);

                const pulseIcon = L.divIcon({
                    className: "",
                    html: `<div class="patrol-pulse"></div>`,
                    iconSize: [30, 30],
                    iconAnchor: [15, 15]
                });

                L.marker([z.lat, z.lng], { icon: pulseIcon })
                .bindPopup(`
                    <div style="text-align:center;">
                        🚔 <strong>موقع تمركز متوقع</strong><br>
                        خلال 24 ساعة القادمة
                    </div>
                `)
                .addTo(patrolLayer);
            });

            patrolLayer.addTo(map);
            showMapMessage("🚔 تم عرض تمركز الدوريات المتوقع");
        })
        .catch(() => showMapMessage("⚠ فشل حساب تمركز الدوريات"));
}

/* ===========================
   تسجيل بلاغ يدوي
=========================== */
function logIncident() {
    const body = {
        incident_type: incident_type.value,
        observed_risk: observed_risk.value,
        recommendation: manual_recommendation.value,
        lat: Number(manual_lat.value),
        lng: Number(manual_lng.value)
    };

    fetch(`${API}/save-incident`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
    })
    .then(r => r.json())
    .then(() => {
        log_output.innerHTML = "✔ تم تسجيل البلاغ بنجاح";
        loadIncidents();
        updateDashboardStats();
        loadHeatmap();
    })
    .catch(() => {
        log_output.innerHTML = "⚠ فشل حفظ البلاغ – تأكد من تشغيل الخادم";
    });
}

/* ===========================
   عرض البلاغات
=========================== */
function loadIncidents() {
    fetch(`${API}/incidents`)
        .then(r => r.json())
        .then(data => {
            const tbody = document.getElementById("incident_table");
            tbody.innerHTML = "";

            data.forEach(i => {
                const tr = document.createElement("tr");
                const lat = i.lat != null ? i.lat.toFixed(4) : "-";
                const lng = i.lng != null ? i.lng.toFixed(4) : "-";

                tr.innerHTML = `
                    <td>${i.incident_type || ""}</td>
                    <td>${i.predicted_risk || ""}</td>
                    <td>${i.observed_risk || ""}</td>
                    <td>${i.recommendation || ""}</td>
                    <td>(${lat}, ${lng})</td>
                    <td>${i.time || ""}</td>
                    <td><button class="delete-btn" onclick="deleteIncident(${i.id})">مسح</button></td>
                `;
                tbody.appendChild(tr);
            });

            updateCharts(data);
        })
        .catch(() => console.warn("Incidents fetch error"));
}

/* ===========================
   حذف بلاغ
=========================== */
function deleteIncident(id) {
    fetch(`${API}/delete-incident`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id })
    }).then(() => {
        loadIncidents();
        updateDashboardStats();
        loadHeatmap();
    });
}

/* ===========================
   مسح كل البلاغات
=========================== */
function clearIncidents() {
    fetch(`${API}/clear-incidents`, {
        method: "POST"
    }).then(() => {
        loadIncidents();
        updateDashboardStats();
        loadHeatmap();
    });
}

/* ===========================
   تصدير PDF
=========================== */
function exportPDF() {
    window.open(`${API}/export-pdf`, "_blank");
}

/* ===========================
   Dashboard Stats
=========================== */
function updateDashboardStats() {
    fetch(`${API}/dashboard-stats`)
        .then(r => r.json())
        .then(s => {
            stat_total.innerText = s.total;
            stat_high.innerText = s.high;
            stat_last_hour.innerText = s.last_hour;
            stat_high_pct.innerText = s.high_pct + "%";
        })
        .catch(() => console.warn("Stats fetch error"));
}

/* ===========================
   Charts
=========================== */
let barChart, pieChart;

function updateCharts(data) {
    const low = data.filter(d => d.predicted_risk === "منخفض").length;
    const med = data.filter(d => d.predicted_risk === "متوسط").length;
    const high = data.filter(d => d.predicted_risk === "مرتفع").length;

    const ctxBar = document.getElementById("riskBarChart");
    const ctxPie = document.getElementById("riskChart");

    if (!ctxBar || !ctxPie) return;

    if (barChart) barChart.destroy();
    if (pieChart) pieChart.destroy();

    // PIE
    pieChart = new Chart(ctxPie, {
        type: "pie",
        data: {
            labels: ["منخفض", "متوسط", "مرتفع"],
            datasets: [{
                data: [low, med, high],
                backgroundColor: ["#22c55e", "#f59e0b", "#ef4444"],
                borderColor: "#fff",
                borderWidth: 3
            }]
        },
        options: {
            plugins: {
                legend: {
                    display: true,
                    position: "bottom",
                    labels: {
                        font: { size: 13 },
                        color: "#000000"
                    }
                }
            }
        }
    });

    // BAR
    barChart = new Chart(ctxBar, {
        type: "bar",
        data: {
            labels: ["منخفض", "متوسط", "مرتفع"],
            datasets: [{
                data: [low, med, high],
                backgroundColor: ["#22c55e", "#f59e0b", "#ef4444"],
                borderRadius: 8,
                barThickness: 55
            }]
        },
        options: {
            scales: {
                x: {
                    ticks: { color: "#000000" },
                    grid: { display: false }
                },
                y: {
                    beginAtZero: true,
                    ticks: { color: "#000000" },
                    grid: { color: "#ddd" }
                }
            },
            plugins: { legend: { display: false } }
        }
    });
}
