let map = null;
let alertsData = [];
let markersMap = {};
let selectedAlertId = null;
let activeBbox = null;
let isDrawingAoi = false;
let drawStart = null;
let currentDate = "2026-10-22";

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

document.addEventListener("DOMContentLoaded", () => {
    initMap();
    setupFilters();
    setupLayerSwitcher();
    setupTimeline();
    setupScanner();
    setInterval(fetchAlerts, 10000);
});

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
            {
                id: "layer-carto-dark",
                type: "raster",
                source: "carto-dark",
                layout: { visibility: "none" }
            },
            {
                id: "layer-esri-satellite",
                type: "raster",
                source: "esri-satellite",
                layout: { visibility: "visible" }
            },
            {
                id: "layer-opentopo",
                type: "raster",
                source: "opentopo",
                layout: { visibility: "none" }
            },
            {
                id: "layer-arcgis-ocean",
                type: "raster",
                source: "arcgis-ocean",
                layout: { visibility: "none" }
            },
            {
                id: "layer-nasa-gibs",
                type: "raster",
                source: "nasa-gibs",
                layout: { visibility: "none" }
            },
            {
                id: "layer-openseamap",
                type: "raster",
                source: "openseamap",
                layout: { visibility: "visible" }
            }
        ]
    };

    map = new maplibregl.Map({
        container: "map",
        style: styleObj,
        center: [108.0, 16.0],
        zoom: 5
    });

    map.addControl(new maplibregl.NavigationControl(), "top-right");

    map.on("load", () => {
        initAoiBoxLayer();
        fetchZones();
        fetchAlerts();
    });
}

function initAoiBoxLayer() {
    if (map.getSource("aoi-box")) return;

    map.addSource("aoi-box", {
        type: "geojson",
        data: {
            type: "FeatureCollection",
            features: []
        }
    });

    map.addLayer({
        id: "aoi-box-fill",
        type: "fill",
        source: "aoi-box",
        paint: {
            "fill-color": "#06b6d4",
            "fill-opacity": 0.15
        }
    });

    map.addLayer({
        id: "aoi-box-line",
        type: "line",
        source: "aoi-box",
        paint: {
            "line-color": "#22d3ee",
            "line-width": 2,
            "line-dasharray": [2, 2]
        }
    });
}

function updateAoiBoxGeojson(bbox) {
    if (!map || !map.getSource("aoi-box")) return;
    if (!bbox) {
        clearAoiBoxGeojson();
        return;
    }
    const [minX, minY, maxX, maxY] = bbox;
    const feature = {
        type: "Feature",
        geometry: {
            type: "Polygon",
            coordinates: [[
                [minX, minY],
                [maxX, minY],
                [maxX, maxY],
                [minX, maxY],
                [minX, minY]
            ]]
        }
    };
    map.getSource("aoi-box").setData({
        type: "FeatureCollection",
        features: [feature]
    });
}

function clearAoiBoxGeojson() {
    if (!map || !map.getSource("aoi-box")) return;
    map.getSource("aoi-box").setData({
        type: "FeatureCollection",
        features: []
    });
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

    document.getElementById("btn-time-prev7")?.addEventListener("click", () => {
        const cur = offsetFromDate(currentDate);
        applyDate(dateFromOffset(Math.max(0, cur - 7)));
    });

    document.getElementById("btn-time-prev1")?.addEventListener("click", () => {
        const cur = offsetFromDate(currentDate);
        applyDate(dateFromOffset(Math.max(0, cur - 1)));
    });

    document.getElementById("btn-time-next1")?.addEventListener("click", () => {
        const cur = offsetFromDate(currentDate);
        applyDate(dateFromOffset(Math.min(MAX_DAYS, cur + 1)));
    });

    document.getElementById("btn-time-next7")?.addEventListener("click", () => {
        const cur = offsetFromDate(currentDate);
        applyDate(dateFromOffset(Math.min(MAX_DAYS, cur + 7)));
    });

    document.getElementById("btn-time-today")?.addEventListener("click", () => {
        applyDate("2026-10-22");
    });
}

function updateScanStatus(msg) {
    const el = document.getElementById("scan-status");
    if (el) el.textContent = msg;
}

function setupScanner() {
    const presetSelect = document.getElementById("scan-preset");
    presetSelect?.addEventListener("change", (e) => {
        const key = e.target.value;
        if (key === "none") {
            activeBbox = null;
            clearAoiBoxGeojson();
            updateScanStatus("Current map viewport will be scanned.");
            return;
        }
        const preset = PRESETS[key];
        if (preset) {
            activeBbox = [...preset.bbox];
            map.flyTo({ center: preset.center, zoom: preset.zoom });
            updateAoiBoxGeojson(activeBbox);
            updateScanStatus(`Preset selected: ${preset.name}`);
        }
    });

    const drawBtn = document.getElementById("btn-draw-aoi");
    drawBtn?.addEventListener("click", () => {
        isDrawingAoi = !isDrawingAoi;
        if (isDrawingAoi) {
            drawBtn.classList.add("active");
            drawBtn.textContent = "CANCEL DRAW";
            map.getCanvas().style.cursor = "crosshair";
            map.dragPan.disable();
            updateScanStatus("Click and drag on map to define custom AOI.");
        } else {
            drawBtn.classList.remove("active");
            drawBtn.textContent = "DRAW AOI BOX";
            map.getCanvas().style.cursor = "";
            map.dragPan.enable();
            drawStart = null;
        }
    });

    map.on("mousedown", (e) => {
        if (!isDrawingAoi) return;
        drawStart = [e.lngLat.lng, e.lngLat.lat];
    });

    map.on("mousemove", (e) => {
        if (!isDrawingAoi || !drawStart) return;
        const cur = [e.lngLat.lng, e.lngLat.lat];
        const minX = Math.min(drawStart[0], cur[0]);
        const maxX = Math.max(drawStart[0], cur[0]);
        const minY = Math.min(drawStart[1], cur[1]);
        const maxY = Math.max(drawStart[1], cur[1]);
        updateAoiBoxGeojson([minX, minY, maxX, maxY]);
    });

    map.on("mouseup", (e) => {
        if (!isDrawingAoi || !drawStart) return;
        const cur = [e.lngLat.lng, e.lngLat.lat];
        const minX = Math.min(drawStart[0], cur[0]);
        const maxX = Math.max(drawStart[0], cur[0]);
        const minY = Math.min(drawStart[1], cur[1]);
        const maxY = Math.max(drawStart[1], cur[1]);
        if (Math.abs(maxX - minX) > 0.005 && Math.abs(maxY - minY) > 0.005) {
            activeBbox = [minX, minY, maxX, maxY];
            updateAoiBoxGeojson(activeBbox);
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
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ bbox, source, date })
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            updateScanStatus(`Scan failed: ${err.detail || resp.statusText}`);
            return;
        }
        const data = await resp.json();
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

async function fetchZones() {
    try {
        const resp = await fetch("/api/zones");
        if (!resp.ok) return;
        const geojson = await resp.json();

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
            const name = props.name || "Surveillance Zone";
            const radius = props.max_radius_km ? ` (${props.max_radius_km} km)` : "";
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
        const resp = await fetch(`/api/alerts?${params.toString()}`);
        if (!resp.ok) return;
        alertsData = await resp.json();
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
        item.className = `alert-item level-${alert.level} ${alert.alert_id === selectedAlertId ? 'selected' : ''}`;
        item.dataset.id = alert.alert_id;

        item.innerHTML = `
            <div class="alert-item-header">
                <span class="alert-target">${alert.target_class || 'TARGET'}</span>
                <span class="badge badge-${alert.level}">${alert.level}</span>
            </div>
            <div class="alert-item-body">
                <span>Zone ${alert.zone} | Score: ${(alert.score || 0).toFixed(1)}</span>
                <span class="badge badge-${alert.status || 'PENDING'}">${alert.status || 'PENDING'}</span>
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
        dot.className = `marker-dot ${alert.level}`;
        el.appendChild(dot);

        const popupContent = `
            <div style="font-size: 11px; color: #111; min-width: 140px;">
                <strong>${alert.target_class || 'TARGET'}</strong><br/>
                ID: ${alert.alert_id}<br/>
                Severity: <b>${alert.level}</b> (Score: ${(alert.score || 0).toFixed(1)})<br/>
                Status: <b>${alert.status || 'PENDING'}</b>
                <div style="margin-top: 6px;">
                    <img src="/api/chips/${alert.alert_id}" style="width: 100%; height: 60px; object-fit: cover; border-radius: 2px; display: block; background: #000;" onerror="this.style.display='none'"/>
                </div>
                <div style="margin-top: 6px; display: flex; gap: 4px;">
                    <button style="flex: 1; padding: 2px 4px; font-size: 9px; background: #dc2626; color: #fff; border: none; border-radius: 2px; cursor: pointer;" onclick="auditAlert('${alert.alert_id}', 'CONFIRMED')">CONFIRM</button>
                    <button style="flex: 1; padding: 2px 4px; font-size: 9px; background: #475569; color: #fff; border: none; border-radius: 2px; cursor: pointer;" onclick="auditAlert('${alert.alert_id}', 'DISMISSED')">DISMISS</button>
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

function renderInspectionDrawer(alert) {
    const statusBadge = document.getElementById("inspect-status-badge");
    statusBadge.textContent = alert.status || "PENDING";
    statusBadge.className = `badge badge-${alert.status || 'PENDING'}`;

    const drawerBody = document.getElementById("drawer-body");
    const chipSrc = `/api/chips/${alert.alert_id}`;

    drawerBody.innerHTML = `
        <img class="chip-preview" src="${chipSrc}" onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'256\\' height=\\'256\\'><rect width=\\'256\\' height=\\'256\\' fill=\\'%23111\\'/><text x=\\'50%\\' y=\\'50%\\' fill=\\'%23666\\' dominant-baseline=\\'middle\\' text-anchor=\\'middle\\'>CHIP PREVIEW</text></svg>'" alt="Chip preview"/>
        <div class="drawer-detail-grid">
            <div class="detail-cell">
                <div class="detail-label">TARGET CLASS</div>
                <div class="detail-val">${alert.target_class || 'UNKNOWN'}</div>
            </div>
            <div class="detail-cell">
                <div class="detail-label">SEVERITY / SCORE</div>
                <div class="detail-val">${alert.level} (${(alert.score || 0).toFixed(1)})</div>
            </div>
            <div class="detail-cell">
                <div class="detail-label">SENSOR</div>
                <div class="detail-val">${alert.sensor || 'SAR'}</div>
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
        <div class="action-buttons">
            <button class="btn btn-confirm" onclick="auditAlert('${alert.alert_id}', 'CONFIRMED')">CONFIRM THREAT</button>
            <button class="btn btn-dismiss" onclick="auditAlert('${alert.alert_id}', 'DISMISSED')">DISMISS</button>
        </div>
    `;
}

async function auditAlert(alertId, status) {
    try {
        const resp = await fetch(`/api/alerts/${alertId}/audit`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ status, note: `Audit action ${status} from Tactical Command UI` })
        });
        if (resp.ok) {
            await fetchAlerts();
            selectAlert(alertId);
        }
    } catch (e) {
    }
}

function setupFilters() {
    document.getElementById("filter-zone").addEventListener("change", fetchAlerts);
    document.getElementById("filter-level").addEventListener("change", fetchAlerts);
    document.getElementById("filter-status").addEventListener("change", fetchAlerts);
}
