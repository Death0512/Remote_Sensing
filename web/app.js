function escapeHtml(str) {
    if (str === null || str === undefined) return "";
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

let inMemoryAuthToken = "";
function getStoredToken() {
    if (inMemoryAuthToken) return inMemoryAuthToken;
    try {
        if (typeof sessionStorage !== "undefined") {
            const token = sessionStorage.getItem("tactical_token");
            if (token) {
                inMemoryAuthToken = token;
                return token;
            }
        }
    } catch (e) {
    }
    return "";
}

function setStoredToken(token) {
    inMemoryAuthToken = token || "";
    try {
        if (typeof sessionStorage !== "undefined") {
            if (token) {
                sessionStorage.setItem("tactical_token", token);
            } else {
                sessionStorage.removeItem("tactical_token");
            }
        }
    } catch (e) {
    }
}

let map = null;
let alertsData = [];
let markersMap = {};
let selectedAlertId = null;
let activeBbox = null;
let isDrawingAoi = false;
let drawStart = null;
let currentDate = "2026-10-22";
let currentChipTab = "preview";
let currentAuthToken = getStoredToken();

const BASE_TIMESTAMP = Date.UTC(2024, 0, 1);
const MAX_DAYS = 1025;

const ZONE_COLORS = {
    "1": "#ef4444",
    "2": "#f59e0b",
    "3": "#3b82f6"
};

const PRESETS = {
    "danang": {
        name: "Da Nang Bay & Naval Base",
        center: [108.22, 16.12],
        bbox: [108.10, 16.02, 108.32, 16.22],
        zoom: 12
    },
    "camranh": {
        name: "Cam Ranh Bay & Base",
        center: [109.15, 11.90],
        bbox: [109.05, 11.80, 109.28, 12.02],
        zoom: 12
    },
    "truongsa": {
        name: "Spratly Islands / Truong Sa",
        center: [111.92, 9.65],
        bbox: [111.75, 9.50, 112.10, 9.80],
        zoom: 11
    },
    "hoangsa": {
        name: "Paracel Islands / Hoang Sa",
        center: [112.33, 16.83],
        bbox: [112.20, 16.70, 112.45, 16.95],
        zoom: 11
    },
    "malacca": {
        name: "Malacca Strait Chokepoint",
        center: [102.50, 2.00],
        bbox: [102.20, 1.80, 102.80, 2.20],
        zoom: 10
    }
};

if (typeof document !== "undefined") {
    document.addEventListener("DOMContentLoaded", () => {
        setupAuth();
        initMap();
        setupFilters();
        setupLayerSwitcher();
        setupTimeline();
        setupScanner();
        setupDelegatedEvents();
        setInterval(fetchAlerts, 10000);
    });
}

async function setupAuth() {
    const usernameInput = document.getElementById("auth-username");
    const passwordInput = document.getElementById("auth-password");
    const loginBtn = document.getElementById("auth-login-btn");
    const indicator = document.getElementById("auth-status-indicator");

    async function loginWithCredentials(username, password) {
        if (!username || !password) return;
        try {
            const resp = await fetch("/api/auth/token", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ username, password })
            });
            if (resp.ok) {
                const res = await resp.json();
                const payload = res.data || res;
                currentAuthToken = payload.access_token;
                setStoredToken(currentAuthToken);
                const role = (payload.user && payload.user.role) ? payload.user.role.toUpperCase() : "AUTHENTICATED";
                if (indicator) {
                    indicator.textContent = `AUTHENTICATED (${role})`;
                    indicator.style.color = "var(--accent-green)";
                }
                if (passwordInput) passwordInput.value = "";
                await fetchAlerts();
            } else {
                if (indicator) {
                    indicator.textContent = "AUTH FAILED";
                    indicator.style.color = "var(--accent-red)";
                }
            }
        } catch (e) {
            if (indicator) {
                indicator.textContent = "AUTH ERROR";
                indicator.style.color = "var(--accent-red)";
            }
        }
    }

    if (loginBtn) {
        loginBtn.addEventListener("click", () => {
            const u = usernameInput ? usernameInput.value.trim() : "";
            const p = passwordInput ? passwordInput.value : "";
            loginWithCredentials(u, p);
        });
    }

    if (passwordInput) {
        passwordInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                const u = usernameInput ? usernameInput.value.trim() : "";
                const p = passwordInput ? passwordInput.value : "";
                loginWithCredentials(u, p);
            }
        });
    }

    const existingToken = getStoredToken();
    if (existingToken && indicator) {
        indicator.textContent = "AUTHENTICATED";
        indicator.style.color = "var(--accent-green)";
    }
}

function getAuthHeaders() {
    const headers = { "Content-Type": "application/json" };
    const token = inMemoryAuthToken || currentAuthToken || getStoredToken();
    if (token) {
        headers["Authorization"] = `Bearer ${token}`;
    }
    return headers;
}

function getGibsUrl(dateStr) {
    return `https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/VIIRS_SNPP_CorrectedReflectance_TrueColor/default/${dateStr}/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg`;
}

function initMap() {
    const styleObj = {
        version: 8,
        sources: {
            "carto-dark": {
                type: "raster",
                tiles: [
                    "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
                    "https://b.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
                    "https://c.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png"
                ],
                tileSize: 256
            },
            "esri-satellite": {
                type: "raster",
                tiles: [
                    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                ],
                tileSize: 256,
                maxzoom: 19
            },
            "opentopo": {
                type: "raster",
                tiles: [
                    "https://tile.opentopomap.org/{z}/{x}/{y}.png"
                ],
                tileSize: 256,
                maxzoom: 17
            },
            "arcgis-ocean": {
                type: "raster",
                tiles: [
                    "https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}"
                ],
                tileSize: 256,
                maxzoom: 16
            },
            "openseamap": {
                type: "raster",
                tiles: [
                    "https://tiles.openseamap.org/seamark/{z}/{x}/{y}.png"
                ],
                tileSize: 256,
                maxzoom: 18
            },
            "nasa-gibs": {
                type: "raster",
                tiles: [
                    getGibsUrl(currentDate)
                ],
                tileSize: 256,
                maxzoom: 9
            }
        },
        layers: [
            { id: "layer-carto-dark", type: "raster", source: "carto-dark", layout: { visibility: "none" } },
            { id: "layer-arcgis-ocean", type: "raster", source: "arcgis-ocean", layout: { visibility: "none" } },
            { id: "layer-opentopo", type: "raster", source: "opentopo", layout: { visibility: "none" } },
            { id: "layer-esri-satellite", type: "raster", source: "esri-satellite", layout: { visibility: "visible" } },
            { id: "layer-nasa-gibs", type: "raster", source: "nasa-gibs", layout: { visibility: "none" } },
            { id: "layer-openseamap", type: "raster", source: "openseamap", layout: { visibility: "visible" } }
        ]
    };

    map = new maplibregl.Map({
        container: "map",
        style: styleObj,
        center: [111.5, 12.0],
        zoom: 5.5,
        attributionControl: false
    });

    map.addControl(new maplibregl.NavigationControl(), "top-right");
    map.addControl(new maplibregl.ScaleControl({ maxWidth: 200, unit: "metric" }), "bottom-right");

    map.on("load", () => {
        fetchZones();
        fetchAlerts();
        initAoiDrawLayers();
    });
}

function initAoiDrawLayers() {
    map.addSource("aoi-draw", {
        type: "geojson",
        data: {
            type: "FeatureCollection",
            features: []
        }
    });

    map.addLayer({
        id: "aoi-draw-fill",
        type: "fill",
        source: "aoi-draw",
        paint: {
            "fill-color": "#06b6d4",
            "fill-opacity": 0.15
        }
    });

    map.addLayer({
        id: "aoi-draw-line",
        type: "line",
        source: "aoi-draw",
        paint: {
            "line-color": "#06b6d4",
            "line-width": 2,
            "line-dasharray": [3, 2]
        }
    });
}

function setAoiBox(bbox) {
    activeBbox = bbox;
    const [w, s, e, n] = bbox;
    const geom = {
        type: "Polygon",
        coordinates: [[
            [w, n],
            [e, n],
            [e, s],
            [w, s],
            [w, n]
        ]]
    };
    if (map.getSource("aoi-draw")) {
        map.getSource("aoi-draw").setData({
            type: "FeatureCollection",
            features: [{ type: "Feature", properties: {}, geometry: geom }]
        });
    }
}

function setupScanner() {
    const presetSelect = document.getElementById("scan-preset");
    if (presetSelect) {
        presetSelect.addEventListener("change", (e) => {
            const key = e.target.value;
            if (key === "none") {
                activeBbox = null;
                if (map.getSource("aoi-draw")) {
                    map.getSource("aoi-draw").setData({ type: "FeatureCollection", features: [] });
                }
                updateScanStatus("Preset cleared. Ready to scan current view.");
                return;
            }
            const preset = PRESETS[key];
            if (preset) {
                map.flyTo({ center: preset.center, zoom: preset.zoom });
                setAoiBox(preset.bbox);
                updateScanStatus(`Preset selected: ${preset.name}`);
            }
        });
    }

    const drawBtn = document.getElementById("btn-draw-aoi");
    if (drawBtn) {
        drawBtn.addEventListener("click", () => {
            isDrawingAoi = !isDrawingAoi;
            if (isDrawingAoi) {
                drawBtn.classList.add("active");
                drawBtn.textContent = "CLICK & DRAG AOI";
                map.getCanvas().style.cursor = "crosshair";
                map.dragPan.disable();
            } else {
                drawBtn.classList.remove("active");
                drawBtn.textContent = "DRAW AOI BOX";
                map.getCanvas().style.cursor = "";
                map.dragPan.enable();
            }
        });
    }

    map.on("mousedown", (e) => {
        if (!isDrawingAoi) return;
        drawStart = [e.lngLat.lng, e.lngLat.lat];
    });

    map.on("mousemove", (e) => {
        if (!isDrawingAoi || !drawStart) return;
        const current = [e.lngLat.lng, e.lngLat.lat];
        const minX = Math.min(drawStart[0], current[0]);
        const maxX = Math.max(drawStart[0], current[0]);
        const minY = Math.min(drawStart[1], current[1]);
        const maxY = Math.max(drawStart[1], current[1]);
        setAoiBox([minX, minY, maxX, maxY]);
    });

    map.on("mouseup", (e) => {
        if (!isDrawingAoi || !drawStart) return;
        const current = [e.lngLat.lng, e.lngLat.lat];
        const minX = Math.min(drawStart[0], current[0]);
        const maxX = Math.max(drawStart[0], current[0]);
        const minY = Math.min(drawStart[1], current[1]);
        const maxY = Math.max(drawStart[1], current[1]);
        if (Math.abs(maxX - minX) > 0.005 && Math.abs(maxY - minY) > 0.005) {
            setAoiBox([minX, minY, maxX, maxY]);
            updateScanStatus(`AOI Box: [${minX.toFixed(3)}, ${minY.toFixed(3)}, ${maxX.toFixed(3)}, ${maxY.toFixed(3)}]`);
            if (presetSelect) presetSelect.value = "none";
        }
        isDrawingAoi = false;
        drawStart = null;
        drawBtn.classList.remove("active");
        drawBtn.textContent = "DRAW AOI BOX";
        map.getCanvas().style.cursor = "";
        map.dragPan.enable();
    });

    document.getElementById("btn-scan-aoi")?.addEventListener("click", triggerAoiScan);
}

function updateScanStatus(msg) {
    const el = document.getElementById("scan-status");
    if (el) el.textContent = msg;
}

async function triggerAoiScan() {
    let bbox = activeBbox;
    if (!bbox) {
        const bounds = map.getBounds();
        bbox = [bounds.getWest(), bounds.getSouth(), bounds.getEast(), bounds.getNorth()];
    }

    const source = document.getElementById("scan-source").value;
    const date = currentDate;
    const scanBtn = document.getElementById("btn-scan-aoi");
    if (scanBtn) {
        scanBtn.classList.add("scanning");
        scanBtn.disabled = true;
    }
    updateScanStatus(`Scanning ${source.toUpperCase()} tiles: [${bbox[0].toFixed(2)}, ${bbox[1].toFixed(2)}, ${bbox[2].toFixed(2)}, ${bbox[3].toFixed(2)}]...`);

    try {
        const resp = await fetch("/api/scan-map-aoi", {
            method: "POST",
            headers: getAuthHeaders(),
            body: JSON.stringify({ bbox, source, date })
        });
        const resJson = await resp.json().catch(() => ({}));
        if (!resp.ok) {
            const errMsg = resJson.errors && resJson.errors.length > 0 ? resJson.errors[0].message : (resJson.detail || resp.statusText);
            updateScanStatus(`Scan failed: ${errMsg}`);
            return;
        }
        const data = resJson.data || resJson;
        updateScanStatus(`Scan complete: ${data.detections_count} contacts detected across ${data.tiles_fetched} tiles.`);
        await fetchAlerts();
        if (data.alerts && data.alerts.length > 0) {
            selectAlert(data.alerts[0].alert_id);
        }
    } catch (err) {
        updateScanStatus(`Scan error: ${err.message}`);
    } finally {
        if (scanBtn) {
            scanBtn.classList.remove("scanning");
            scanBtn.disabled = false;
        }
    }
}

function setupLayerSwitcher() {
    const radios = document.querySelectorAll('input[name="baselayer"]');
    radios.forEach(radio => {
        radio.addEventListener("change", (e) => {
            setBaseLayer(e.target.value);
        });
    });

    const seamarksToggle = document.getElementById("toggle-seamarks");
    if (seamarksToggle) {
        seamarksToggle.addEventListener("change", (e) => {
            const visibility = e.target.checked ? "visible" : "none";
            if (map.getLayer("layer-openseamap")) {
                map.setLayoutProperty("layer-openseamap", "visibility", visibility);
            }
        });
    }
}

function setBaseLayer(layerKey) {
    const layerIds = {
        "satellite": "layer-esri-satellite",
        "dark": "layer-carto-dark",
        "ocean": "layer-arcgis-ocean",
        "topo": "layer-opentopo",
        "gibs": "layer-nasa-gibs"
    };

    Object.entries(layerIds).forEach(([key, id]) => {
        if (map.getLayer(id)) {
            map.setLayoutProperty(id, "visibility", key === layerKey ? "visible" : "none");
        }
    });

    const timelineBar = document.getElementById("gibs-timeline-bar");
    if (timelineBar) {
        timelineBar.style.display = (layerKey === "gibs") ? "flex" : "none";
    }

    const scanSource = document.getElementById("scan-source");
    if (scanSource) {
        if (layerKey === "gibs") {
            scanSource.value = "nasa_gibs";
        } else if (layerKey === "satellite") {
            scanSource.value = "satellite_hd";
        }
    }
}

function setupTimeline() {
    const slider = document.getElementById("gibs-slider");
    const picker = document.getElementById("gibs-date-picker");
    const badge = document.getElementById("gibs-date-badge");

    function dateFromOffset(offset) {
        const d = new Date(BASE_TIMESTAMP + offset * 86400000);
        return d.toISOString().split("T")[0];
    }

    function offsetFromDate(dateStr) {
        const d = new Date(dateStr + "T00:00:00Z");
        const diff = Math.round((d.getTime() - BASE_TIMESTAMP) / 86400000);
        return Math.max(0, Math.min(MAX_DAYS, diff));
    }

    function applyDate(dateStr) {
        currentDate = dateStr;
        if (badge) badge.textContent = dateStr;
        if (picker && picker.value !== dateStr) picker.value = dateStr;
        const offset = offsetFromDate(dateStr);
        if (slider && Number(slider.value) !== offset) slider.value = String(offset);

        const newUrl = getGibsUrl(dateStr);
        if (map && map.getSource("nasa-gibs")) {
            const isVisible = map.getLayoutProperty("layer-nasa-gibs", "visibility") || "none";
            map.removeLayer("layer-nasa-gibs");
            map.removeSource("nasa-gibs");
            map.addSource("nasa-gibs", {
                type: "raster",
                tiles: [newUrl],
                tileSize: 256,
                maxzoom: 9
            });
            map.addLayer({
                id: "layer-nasa-gibs",
                type: "raster",
                source: "nasa-gibs",
                layout: { visibility: isVisible }
            }, "layer-openseamap");
        }
    }

    if (slider) {
        slider.addEventListener("input", (e) => {
            const dateStr = dateFromOffset(Number(e.target.value));
            applyDate(dateStr);
        });
    }

    if (picker) {
        picker.addEventListener("change", (e) => {
            if (e.target.value) applyDate(e.target.value);
        });
    }

    document.getElementById("btn-time-today")?.addEventListener("click", () => {
        applyDate("2026-10-22");
    });

    document.getElementById("btn-time-prev1")?.addEventListener("click", () => {
        const off = Math.max(0, offsetFromDate(currentDate) - 1);
        applyDate(dateFromOffset(off));
    });

    document.getElementById("btn-time-next1")?.addEventListener("click", () => {
        const off = Math.min(MAX_DAYS, offsetFromDate(currentDate) + 1);
        applyDate(dateFromOffset(off));
    });

    document.getElementById("btn-time-prev7")?.addEventListener("click", () => {
        const off = Math.max(0, offsetFromDate(currentDate) - 7);
        applyDate(dateFromOffset(off));
    });

    document.getElementById("btn-time-next7")?.addEventListener("click", () => {
        const off = Math.min(MAX_DAYS, offsetFromDate(currentDate) + 7);
        applyDate(dateFromOffset(off));
    });
}

async function fetchZones() {
    try {
        const resp = await fetch("/api/zones", { headers: getAuthHeaders() });
        if (!resp.ok) return;
        const resJson = await resp.json();
        const geojson = resJson.data || resJson;

        if (map.getSource("zones")) {
            map.getSource("zones").setData(geojson);
            return;
        }

        map.addSource("zones", {
            type: "geojson",
            data: geojson
        });

        const zoneColorExpression = [
            "match",
            ["to-string", ["coalesce", ["get", "zone_id"], ["get", "zone"], "3"]],
            "1", ZONE_COLORS["1"],
            "2", ZONE_COLORS["2"],
            "3", ZONE_COLORS["3"],
            ZONE_COLORS["3"]
        ];

        map.addLayer({
            id: "zones-fill",
            type: "fill",
            source: "zones",
            paint: {
                "fill-color": zoneColorExpression,
                "fill-opacity": [
                    "match",
                    ["to-string", ["coalesce", ["get", "zone_id"], ["get", "zone"], "3"]],
                    "1", 0.12,
                    "2", 0.08,
                    0.05
                ]
            }
        });

        map.addLayer({
            id: "zones-line",
            type: "line",
            source: "zones",
            paint: {
                "line-color": zoneColorExpression,
                "line-width": 2,
                "line-opacity": 0.8
            }
        });

        const popup = new maplibregl.Popup({
            closeButton: false,
            closeOnClick: false
        });

        map.on("mousemove", "zones-fill", (e) => {
            map.getCanvas().style.cursor = "pointer";
            if (!e.features || !e.features.length) return;
            const props = e.features[0].properties || {};
            const name = escapeHtml(props.name || "Surveillance Zone");
            const radius = props.max_radius_km ? ` (${escapeHtml(props.max_radius_km)} km)` : "";
            popup.setLngLat(e.lngLat).setHTML(`<strong>${name}</strong>${radius}`).addTo(map);
        });

        map.on("mouseleave", "zones-fill", () => {
            map.getCanvas().style.cursor = "";
            popup.remove();
        });
    } catch (e) {
    }
}

async function fetchAlerts() {
    const zoneFilter = document.getElementById("filter-zone").value;
    const levelFilter = document.getElementById("filter-level").value;
    const statusFilter = document.getElementById("filter-status").value;

    const params = new URLSearchParams();
    if (zoneFilter !== "ALL") params.append("zone", zoneFilter);
    if (levelFilter !== "ALL") params.append("level", levelFilter);
    if (statusFilter !== "ALL") params.append("status", statusFilter);

    try {
        const resp = await fetch(`/api/alerts?${params.toString()}`, { headers: getAuthHeaders() });
        if (!resp.ok) return;
        const resJson = await resp.json();
        alertsData = resJson.data !== undefined ? resJson.data : resJson;
        renderAlertList();
        renderMapMarkers();
        updateMetrics();
    } catch (e) {
    }
}

function updateMetrics() {
    const total = alertsData.length;
    const pending = alertsData.filter(a => a.status === "PENDING" || !a.status).length;
    document.getElementById("active-alert-count").textContent = String(total);
    document.getElementById("pending-audit-count").textContent = String(pending);
}

function renderAlertList() {
    const container = document.getElementById("alert-list");
    container.innerHTML = "";

    alertsData.forEach(alert => {
        const item = document.createElement("div");
        const safeLevel = escapeHtml(alert.level || 'UNKNOWN');
        const safeTarget = escapeHtml(alert.target_class || 'TARGET');
        const safeStatus = escapeHtml(alert.status || 'PENDING');
        const safeId = escapeHtml(alert.alert_id);
        const safeZone = escapeHtml(alert.zone);
        const scoreVal = (alert.score || 0).toFixed(1);

        item.className = `alert-item level-${safeLevel} ${alert.alert_id === selectedAlertId ? 'selected' : ''}`;
        item.dataset.id = safeId;

        item.innerHTML = `
            <div class="alert-item-header">
                <span class="alert-target">${safeTarget}</span>
                <span class="badge badge-${safeLevel}">${safeLevel}</span>
            </div>
            <div class="alert-item-body">
                <span>Zone ${safeZone} | Score: ${scoreVal}</span>
                <span class="badge badge-${safeStatus}">${safeStatus}</span>
            </div>
        `;

        item.addEventListener("click", () => selectAlert(alert.alert_id));
        container.appendChild(item);
    });
}

function renderMapMarkers() {
    Object.values(markersMap).forEach(entry => {
        if (entry.marker) entry.marker.remove();
    });
    markersMap = {};

    alertsData.forEach(alert => {
        if (alert.lat == null || alert.lon == null) return;

        const el = document.createElement("div");
        el.className = "custom-marker";
        const dot = document.createElement("div");
        dot.className = `marker-dot ${escapeHtml(alert.level)}`;
        el.appendChild(dot);

        const safeTarget = escapeHtml(alert.target_class || 'TARGET');
        const safeId = escapeHtml(alert.alert_id);
        const safeLevel = escapeHtml(alert.level);
        const safeStatus = escapeHtml(alert.status || 'PENDING');
        const safeScore = (alert.score || 0).toFixed(1);
        const alertIdJson = JSON.stringify(alert.alert_id || '');

        const popupContent = `
            <div style="font-size: 11px; color: #111; min-width: 140px;">
                <strong>${safeTarget}</strong><br/>
                ID: ${safeId}<br/>
                Severity: <b>${safeLevel}</b> (Score: ${safeScore})<br/>
                Status: <b>${safeStatus}</b>
                <div style="margin-top: 6px;">
                    <img src="/api/chips/${encodeURIComponent(alert.alert_id)}" style="width: 100%; height: 60px; object-fit: cover; border-radius: 2px; display: block; background: #000;" onerror="this.style.display='none'"/>
                </div>
                <div style="margin-top: 6px; display: flex; gap: 4px;">
                    <button style="flex: 1; padding: 2px 4px; font-size: 9px; background: #dc2626; color: #fff; border: none; border-radius: 2px; cursor: pointer;" data-action="CONFIRM" data-alert-id="${safeId}">CONFIRM</button>
                    <button style="flex: 1; padding: 2px 4px; font-size: 9px; background: #475569; color: #fff; border: none; border-radius: 2px; cursor: pointer;" data-action="DISMISS" data-alert-id="${safeId}">DISMISS</button>
                    <button style="flex: 1; padding: 2px 4px; font-size: 9px; background: #d97706; color: #fff; border: none; border-radius: 2px; cursor: pointer;" data-action="FLAG_UNCERTAIN" data-alert-id="${safeId}">UNCERTAIN</button>
                </div>
            </div>
        `;

        const popup = new maplibregl.Popup({ offset: 12 }).setHTML(popupContent);

        const marker = new maplibregl.Marker({ element: el })
            .setLngLat([alert.lon, alert.lat])
            .setPopup(popup)
            .addTo(map);

        el.addEventListener("click", () => selectAlert(alert.alert_id));

        markersMap[alert.alert_id] = { marker, popup };
    });
}

function selectAlert(alertId) {
    selectedAlertId = alertId;
    const alert = alertsData.find(a => a.alert_id === alertId);
    if (!alert) return;

    document.querySelectorAll(".alert-item").forEach(el => {
        el.classList.toggle("selected", el.dataset.id === alertId);
    });

    if (alert.lat != null && alert.lon != null && map) {
        map.flyTo({ center: [alert.lon, alert.lat], zoom: Math.max(map.getZoom(), 8) });
        if (markersMap[alertId]) {
            markersMap[alertId].marker.togglePopup();
        }
    }

    renderInspectionDrawer(alert);
}

function setChipTab(tab) {
    currentChipTab = tab;
    if (selectedAlertId) {
        const alert = alertsData.find(a => a.alert_id === selectedAlertId);
        if (alert) renderInspectionDrawer(alert);
    }
}

function renderInspectionDrawer(alert) {
    const statusBadge = document.getElementById("inspect-status-badge");
    const st = alert.status || "PENDING";
    statusBadge.textContent = st;
    statusBadge.className = `badge badge-${escapeHtml(st)}`;

    const drawerBody = document.getElementById("drawer-body");
    const previewUrl = alert.chips?.preview || `/api/chips/${encodeURIComponent(alert.alert_id)}`;
    const analysisUrl = alert.chips?.analysis || `/api/chips/${encodeURIComponent(alert.alert_id)}/analysis.tif`;
    const rawUrl = alert.chips?.raw || `/api/chips/${encodeURIComponent(alert.alert_id)}/raw.tif`;

    const safeTarget = escapeHtml(alert.target_class || 'UNKNOWN');
    const safeLevel = escapeHtml(alert.level || 'UNKNOWN');
    const safeSensor = escapeHtml(alert.sensor || 'SAR');
    const safeAlertId = escapeHtml(alert.alert_id);
    const safeBundleUri = escapeHtml(alert.bundle_uri || `file://data/outputs/alert_${alert.alert_id}`);

    let chipViewHtml = "";
    if (currentChipTab === "preview") {
        chipViewHtml = `
            <img class="chip-preview" src="${escapeHtml(previewUrl)}" onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'256\\' height=\\'256\\'><rect width=\\'256\\' height=\\'256\\' fill=\\'%23111\\'/><text x=\\'50%\\' y=\\'50%\\' fill=\\'%23666\\' dominant-baseline=\\'middle\\' text-anchor=\\'middle\\'>CHIP PREVIEW</text></svg>'" alt="Chip preview"/>
        `;
    } else if (currentChipTab === "analysis") {
        chipViewHtml = `
            <div class="chip-meta-box">
URI: ${escapeHtml(analysisUrl)}
Type: Float32 Normalized Tensor
Channels: Single / Multiband
Dynamic Range: [0.000, 1.000]
Interpolation: Sub-pixel Gated
Resolution: Metric GSD Gated
Status: Radiometrically Calibrated
            </div>
        `;
    } else {
        chipViewHtml = `
            <div class="chip-meta-box">
URI: ${escapeHtml(rawUrl)}
Bundle: ${safeBundleUri}
Sensor: ${safeSensor}
Target: ${safeTarget}
Coordinates: ${alert.lat != null ? alert.lat.toFixed(5) : '0.00000'}°N, ${alert.lon != null ? alert.lon.toFixed(5) : '0.00000'}°E
Format: 16-bit GeoTIFF / Raw Window
Radiometry: Unaltered Digital Numbers (DN)
Lineage: Walk intact to source asset
            </div>
        `;
    }

    const hyp = alert.hypothesis || {};
    const conf = alert.confidence || {};
    const supp = alert.evidence?.supporting || [];
    const refut = alert.evidence?.refuting || [];
    const prov = alert.provenance || {};

    const safeStatement = escapeHtml(hyp.statement || 'UNVERIFIED_SURFACE_TARGET');
    const safeExplanation = escapeHtml(hyp.primary_explanation || hyp.explanation || 'No primary explanation provided');
    const safeRationale = escapeHtml(hyp.detailed_rationale || '');
    const safeCalibState = escapeHtml(conf.calibration_state || 'NOT_CALIBRATED');
    const safePipeline = escapeHtml(prov.pipeline_version || '3.0.0');
    const safeAssetId = escapeHtml(prov.source_asset_id || ('ASSET_' + alert.alert_id));
    const safeRunId = escapeHtml(prov.processing_run_id || 'live');

    const rawScoreDisplay = conf.raw_score != null ? conf.raw_score.toFixed(2) : (alert.score || 0).toFixed(2);
    const modelScoreDisplay = conf.model_score != null ? conf.model_score.toFixed(2) : 'N/A';
    const calibDisplay = conf.calibrated_probability != null
        ? `<span class="conf-calibrated">${(conf.calibrated_probability * 100).toFixed(1)}%</span>`
        : `<span class="conf-uncalibrated">N/A (Heuristic / Uncalibrated)</span>`;

    drawerBody.innerHTML = `
        <div class="chip-switcher">
            <button class="chip-tab ${currentChipTab === 'preview' ? 'active' : ''}" data-tab="preview">Preview (8-bit)</button>
            <button class="chip-tab ${currentChipTab === 'analysis' ? 'active' : ''}" data-tab="analysis">Analysis (Norm)</button>
            <button class="chip-tab ${currentChipTab === 'raw' ? 'active' : ''}" data-tab="raw">Raw Metadata</button>
        </div>

        ${chipViewHtml}

        <div class="drawer-detail-grid">
            <div class="detail-cell">
                <div class="detail-label">TARGET CLASS</div>
                <div class="detail-val">${safeTarget}</div>
            </div>
            <div class="detail-cell">
                <div class="detail-label">SEVERITY / SCORE</div>
                <div class="detail-val">${safeLevel} (${(alert.score || 0).toFixed(1)})</div>
            </div>
            <div class="detail-cell">
                <div class="detail-label">SENSOR</div>
                <div class="detail-val">${safeSensor}</div>
            </div>
            <div class="detail-cell">
                <div class="detail-label">DARK VESSEL</div>
                <div class="detail-val" style="color: ${alert.is_dark ? '#ef4444' : '#10b981'}">${alert.is_dark ? 'YES (UNTRACKED)' : 'NO (CORRELATED)'}</div>
            </div>
            <div class="detail-cell">
                <div class="detail-label">COORDINATES</div>
                <div class="detail-val">${alert.lat != null ? alert.lat.toFixed(4) : '0.0000'}°N, ${alert.lon != null ? alert.lon.toFixed(4) : '0.0000'}°E</div>
            </div>
            <div class="detail-cell">
                <div class="detail-label">SPEED / HEADING</div>
                <div class="detail-val">${alert.speed_knots != null ? alert.speed_knots.toFixed(1) + ' kn' : 'N/A'} / ${alert.heading_deg != null ? alert.heading_deg.toFixed(0) + '°' : 'N/A'}</div>
            </div>
        </div>

        <div class="section-divider">EXPLAINABLE HYPOTHESIS</div>
        <div class="explain-box">
            <div class="explain-title">${safeStatement}</div>
            <div class="explain-desc">${safeExplanation}</div>
            <div class="explain-rationale">${safeRationale}</div>
        </div>

        <div class="section-divider">HONEST CONFIDENCE VECTOR</div>
        <div class="confidence-vector">
            <div class="conf-cell">
                <div class="conf-label">RAW HEURISTIC SCORE</div>
                <div class="conf-val">${rawScoreDisplay}</div>
            </div>
            <div class="conf-cell">
                <div class="conf-label">MODEL SCORE</div>
                <div class="conf-val">${modelScoreDisplay}</div>
            </div>
            <div class="conf-cell" style="grid-column: span 2;">
                <div class="conf-label">CALIBRATED PROBABILITY</div>
                <div class="conf-val">${calibDisplay}</div>
            </div>
            <div class="conf-cell" style="grid-column: span 2;">
                <div class="conf-label">DATA QUALITY &amp; CALIBRATION STATE</div>
                <div class="conf-val">${(conf.quality_score ?? 1.0).toFixed(2)} [${safeCalibState}]</div>
            </div>
        </div>

        <div class="section-divider">SUPPORTING EVIDENCE (${supp.length})</div>
        <div class="evidence-list">
            ${supp.length > 0 ? supp.map(ev => `
                <div class="evidence-card supporting">
                    <div class="evidence-head">
                        <span>${escapeHtml(ev.kind || 'EVIDENCE')}</span>
                        <span style="color: #10b981;">+${Number(ev.value ?? 1).toFixed(2)}</span>
                    </div>
                    <div class="evidence-body">
                        Source: <b>${escapeHtml(ev.source || 'Sensor')}</b> | Supports: <b>${escapeHtml(ev.supports || 'TARGET')}</b><br/>
                        ${(ev.references || []).map(r => escapeHtml(r)).join(' | ')}
                    </div>
                </div>
            `).join('') : '<div style="color: #64748b; font-size: 10px; font-style: italic;">No supporting evidence</div>'}
        </div>

        <div class="section-divider">REFUTING EVIDENCE (${refut.length})</div>
        <div class="evidence-list">
            ${refut.length > 0 ? refut.map(ev => `
                <div class="evidence-card refuting">
                    <div class="evidence-head">
                        <span>${escapeHtml(ev.kind || 'REFUTING')}</span>
                        <span style="color: #ef4444;">-${Number(ev.value ?? 1).toFixed(2)}</span>
                    </div>
                    <div class="evidence-body">
                        Source: <b>${escapeHtml(ev.source || 'Sensor')}</b> | Counter: <b>${escapeHtml(ev.supports || 'TARGET')}</b><br/>
                        ${(ev.references || []).map(r => escapeHtml(r)).join(' | ')}
                    </div>
                </div>
            `).join('') : '<div style="color: #64748b; font-size: 10px; font-style: italic;">No refuting evidence observed.</div>'}
        </div>

        <div class="section-divider">PROVENANCE &amp; AUDIT TRAIL</div>
        <div class="chip-meta-box">
Pipeline: ${safePipeline}
Asset: ${safeAssetId}
Run: ${safeRunId}
Lineage: Walk intact to source asset
        </div>

        <div class="section-divider">OPERATOR REVIEW ACTION</div>
        <div class="action-buttons">
            <button class="btn btn-confirm" data-action="CONFIRM" data-alert-id="${safeAlertId}">CONFIRM</button>
            <button class="btn btn-dismiss" data-action="DISMISS" data-alert-id="${safeAlertId}">DISMISS</button>
            <button class="btn btn-uncertain" data-action="FLAG_UNCERTAIN" data-alert-id="${safeAlertId}">UNCERTAIN</button>
        </div>
    `;
}

async function performAlertAction(alertId, action) {
    try {
        const resp = await fetch(`/api/alerts/${alertId}/action`, {
            method: "POST",
            headers: getAuthHeaders(),
            body: JSON.stringify({ action: action, notes: `Operator action ${action} from Tactical Command UI` })
        });
        if (resp.ok) {
            await fetchAlerts();
            selectAlert(alertId);
        } else {
            const err = await resp.json().catch(() => ({}));
            const msg = err.errors && err.errors.length > 0 ? err.errors[0].message : (err.detail || "Action forbidden or failed");
            alert(`Review action rejected: ${msg}`);
        }
    } catch (e) {
    }
}

function setupDelegatedEvents() {
    document.addEventListener("click", (e) => {
        const tabBtn = e.target.closest("[data-tab]");
        if (tabBtn) {
            const tab = tabBtn.getAttribute("data-tab");
            if (tab) {
                setChipTab(tab);
                return;
            }
        }

        const actionBtn = e.target.closest("[data-action]");
        if (actionBtn) {
            const action = actionBtn.getAttribute("data-action");
            const alertId = actionBtn.getAttribute("data-alert-id");
            if (action && alertId) {
                performAlertAction(alertId, action);
                return;
            }
        }
    });
}

function auditAlert(alertId, status) {
    return performAlertAction(alertId, status);
}

function setupFilters() {
    document.getElementById("filter-zone").addEventListener("change", fetchAlerts);
    document.getElementById("filter-level").addEventListener("change", fetchAlerts);
    document.getElementById("filter-status").addEventListener("change", fetchAlerts);
}

if (typeof window !== "undefined") {
    window.escapeHtml = escapeHtml;
}
if (typeof module !== "undefined" && module.exports) {
    module.exports = { escapeHtml };
}
