// Debounce utility — delays fn execution until after `wait` ms of inactivity
function debounce(fn, wait) {
  let timer;
  return function(...args) {
    clearTimeout(timer);
    timer = setTimeout(() => fn.apply(this, args), wait);
  };
}

// Demo mode initialization — fetches /api/health and configures UI
async function initDemoMode() {
  try {
    const r = await fetch('/api/health');
    const h = await r.json();
    window.__DEMO__ = !!h.demo;
    if (h.demo) {
      const banner = document.getElementById('demoBanner');
      const range = h.snapshot_range || {};
      const text = (range.start && range.end)
        ? `Demo mode — frozen snapshot of Portugal fires, ${range.start} to ${range.end}.`
        : 'Demo mode — read-only snapshot.';
      banner.textContent = text;
      banner.style.display = 'block';
      // Hide nav links to admin pages
      document.querySelectorAll('.navbar-nav .nav-link').forEach(a => {
        if (a.getAttribute('href') === '/static/dataset.html') {
          a.parentElement.style.display = 'none';
        }
      });
      // Disable AOI write actions
      const drawBtn = document.getElementById('aoiDrawBtn');
      if (drawBtn) {
        drawBtn.classList.add('demo-disabled');
        drawBtn.title = 'Disabled in demo mode';
      }
    }
  } catch (e) {
    // Health unreachable — assume not demo, proceed normally
    window.__DEMO__ = false;
  }
}

// FIRMS response cache keyed by "product:days:bbox"
const _firmsCache = new Map();

// Region configurations with bbox [lon_min, lat_min, lon_max, lat_max]
const REGIONS = {
  mainland: { 
    center: [39.5, -8.0], 
    zoom: 7, 
    name: 'Mainland Portugal',
    bbox: [-9.5, 36.8, -6.0, 42.3]
  },
  madeira: { 
    center: [32.65, -16.9], 
    zoom: 10, 
    name: 'Madeira',
    bbox: [-17.3, 32.3, -16.2, 33.2]
  },
  azores: { 
    center: [38.7, -27.2], 
    zoom: 8, 
    name: 'Açores',
    bbox: [-31.3, 36.9, -24.7, 39.7]
  },
  all: { 
    center: [39.5, -15.0], 
    zoom: 6, 
    name: 'All Portugal Regions',
    bbox: [-31.3, 32.3, -6.0, 42.3]
  },
  california: {
    center: [37.0, -119.5],
    zoom: 6,
    name: 'California (Test Region)',
    bbox: [-124.5, 32.5, -114.0, 42.0]
  },
  western_us: {
    center: [40.0, -115.0],
    zoom: 5,
    name: 'Western USA',
    bbox: [-125.0, 31.0, -102.0, 49.0]
  }
};

// Current active region
let currentRegion = 'mainland';

// Initialize map centered on Portugal
const map = L.map('map').setView([39.5, -8.0], 7);
const osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 19,
  attribution: '© OpenStreetMap contributors'
}).addTo(map);

// Layer groups for different data sources
let viirsLayer = L.layerGroup().addTo(map);
let modisLayer = L.layerGroup();
let frpOverlay = null;
let heatmapLayer = null;
let s3Layer = L.layerGroup();
let allFireData = []; // Store all fire data for heatmap and CSV

// Supercluster for fire point clustering
let fireClusterIndex = null;
let clusterMarkers = [];
let _mlPopupCounter = 0;

// Fire severity colors (mirror CSS tokens --fire-high / --fire-medium / --fire-low)
const FIRE_HIGH_COLOR   = 'hsl(0, 82%, 52%)';
const FIRE_MEDIUM_COLOR = 'hsl(25, 90%, 46%)';
const FIRE_LOW_COLOR    = 'hsl(45, 92%, 40%)';

function fireSeverityColor(conf) {
  return conf >= 80 ? FIRE_HIGH_COLOR : conf >= 50 ? FIRE_MEDIUM_COLOR : FIRE_LOW_COLOR;
}

// Status tracking — oldest message at top, newest at bottom
const STATUS_ICONS = { success: '✓', warning: '⚠', error: '✕', info: '' };
function updateStatus(message, type = 'info') {
  const statusBox = document.querySelector('#statusMessages');
  const timestamp = new Date().toLocaleTimeString();
  const className = type === 'success' ? 'success' : type === 'warning' ? 'warning' : type === 'error' ? 'error' : '';
  const icon = STATUS_ICONS[type] || '';
  // Strip any leading icon already in the message to avoid doubling
  const cleanMsg = message.replace(/^[✓✗✕⚠]\s*/, '');
  const div = document.createElement('div');
  if (className) div.className = className;
  // Note: message comes from internal code only (no user input), safe for innerHTML
  div.innerHTML = `${icon ? `<span class="status-icon">${icon}</span> ` : ''}[${timestamp}] ${cleanMsg}`;
  statusBox.appendChild(div);
  // Keep only last 5 messages
  while (statusBox.children.length > 5) {
    statusBox.removeChild(statusBox.firstChild);
  }
  statusBox.scrollTop = statusBox.scrollHeight;
  console.log(`[${type.toUpperCase()}] ${cleanMsg}`);
}

function updateBadge(elementId, count, status = '') {
  const badge = document.getElementById(elementId);
  if (status) {
    badge.textContent = status;
    badge.className = 'badge';
  } else {
    badge.textContent = count;
    badge.className = count > 0 ? 'badge active' : 'badge empty';
  }
}

// Load FIRMS fire data (VIIRS or MODIS)
async function loadFIRMS(product, layer, days = 1, bbox = null) {
  const productName = product.toUpperCase();
  const badgeId = product === 'viirs' ? 'viirsCount' : 'modisCount';

  try {
    updateStatus(`Loading ${productName} data (${days} days)...`);
    updateBadge(badgeId, 0, '...');

    // Build URL with optional bbox parameter
    let url = `/api/firms/${product}?days=${days}`;
    if (bbox) {
      url += `&bbox=${bbox.join(',')}`;
    }

    // Use cached response if available for same (product, days, bbox) key
    const cacheKey = `${product}:${days}:${bbox ? bbox.join(',') : ''}`;
    let data;
    if (_firmsCache.has(cacheKey)) {
      data = _firmsCache.get(cacheKey);
    } else {
      const res = await fetch(url);
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }
      data = await res.json();
      _firmsCache.set(cacheKey, data);
    }
    layer.clearLayers();
    
    if (data.features && data.features.length > 0) {
      data.features.forEach(f => {
        const [lon, lat] = f.geometry.coordinates;
        const conf = f.properties.confidence ?? 0;
        const frp = f.properties.frp ?? 'N/A';
        const color = fireSeverityColor(conf);

        const marker = L.circleMarker([lat, lon], {
          radius: 5,
          color,
          fillColor: color,
          fillOpacity: 0.7,
          weight: 2
        });

        const popupId = `mlp${++_mlPopupCounter}`;
        marker.bindPopup(`
          <strong>${productName}</strong><br/>
          Time: ${f.properties.acq_time}<br/>
          Confidence: ${conf.toFixed(0)}%<br/>
          FRP: ${frp} MW
          <hr class="popup-divider"/>
          <div id="${popupId}">
            <button class="ml-btn"
              data-lat="${lat}" data-lon="${lon}"
              data-popup-id="${popupId}" data-live="false">
              ML Check
            </button>
            <button class="ml-btn ml-btn-live"
              data-lat="${lat}" data-lon="${lon}"
              data-popup-id="${popupId}" data-live="true"
              title="Fetch a fresh Sentinel-2 tile for this location">
              Live ML Check
            </button>
          </div>
        `);

        // C3: keyboard accessibility for fire markers
        marker.on('add', () => {
          const el = marker.getElement();
          if (!el) return;
          el.setAttribute('tabindex', '0');
          el.setAttribute('role', 'button');
          el.setAttribute('aria-label', `${productName} fire: confidence ${conf.toFixed(0)}%, FRP ${frp} MW`);
          el.addEventListener('keydown', (evt) => {
            if (evt.key === 'Enter' || evt.key === ' ') {
              evt.preventDefault();
              marker.openPopup();
            }
          });
        });

        marker.addTo(layer);
      });
      
      // Store data for heatmap and CSV export
      if (product === 'viirs') {
        allFireData = data.features;
        updateHeatmap();
      }
      
      updateStatus(`✓ ${productName}: ${data.features.length} fire detection(s)`, 'success');
      updateBadge(badgeId, data.features.length);
    } else {
      updateStatus(`${productName}: No fires detected (good news!)`, 'warning');
      updateBadge(badgeId, 0);
    }
  } catch (error) {
    console.error(`Error loading ${productName}:`, error);
    updateStatus(`✗ Error loading ${productName}: ${error.message}`, 'error');
    updateBadge(badgeId, 0, 'ERR');
  }
}

// Load Meteosat FRP overlay
async function loadFRP() {
  try {
    updateStatus('Loading Meteosat FRP overlay...');
    updateBadge('frpStatus', 0, '...');
    
    const res = await fetch('/api/eumetsat/frp/latest');
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    }
    
    const meta = await res.json();
    
    if (frpOverlay) {
      map.removeLayer(frpOverlay);
      frpOverlay = null;
    }
    
    if (meta && meta.error) {
      updateStatus(`Meteosat FRP: ${meta.error}`, 'warning');
      updateBadge('frpStatus', 0, 'N/A');
      console.warn('EUMETSAT FRP not available:', meta.error);
    } else if (meta && meta.path && meta.bbox) {
      const bbox = meta.bbox; // lon_min, lat_min, lon_max, lat_max
      const sw = [bbox[1], bbox[0]];
      const ne = [bbox[3], bbox[2]];
      
      frpOverlay = L.imageOverlay(`/api/images/eumetsat/${meta.id}`, [sw, ne], {
        opacity: 0.6,
        alt: 'Meteosat FRP'
      });
      
      if (document.querySelector('#frpToggle').checked) {
        frpOverlay.addTo(map);
      }
      updateStatus('✓ Meteosat FRP overlay loaded', 'success');
      updateBadge('frpStatus', 0, 'OK');
    } else {
      updateStatus('Meteosat FRP: Not configured (optional feature)', 'warning');
      updateBadge('frpStatus', 0, 'N/A');
    }
  } catch (error) {
    console.error('Error loading FRP:', error);
    updateStatus(`✗ Error loading Meteosat FRP: ${error.message}`, 'error');
    updateBadge('frpStatus', 0, 'ERR');
  }
}

async function loadS3FirePixels() {
  try {
    updateStatus('Loading Sentinel-3 fire pixels...');
    updateBadge('s3Count', 0, '...');
    const res = await fetch('/api/sentinel3/fire-pixels');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    s3Layer.clearLayers();
    if (!data.features || data.features.length === 0) {
      updateStatus('Sentinel-3: No fire pixels — <a href="/static/dataset.html">download via Dataset page</a>', 'warning');
      updateBadge('s3Count', 0);
      return;
    }
    data.features.forEach(f => {
      const [lon, lat] = f.geometry.coordinates;
      const p = f.properties;
      const marker = L.marker([lat, lon], {
        icon: L.divIcon({
          html: '<svg class="s3-marker" width="14" height="14" viewBox="0 0 14 14"><polygon class="s3-marker-poly" points="7,1 13,5 11,13 3,13 1,5"/></svg>',
          className: '', iconSize: [14, 14], iconAnchor: [7, 7],
        }),
      });
      const satLabel = p.satellite ? `${p.satellite} (SLSTR)` : 'Sentinel-3';
      const instrLabel = p.instrument ? `<br/>Instrument: ${p.instrument}` : '';
      const dateLabel = p.date ? `<br/>Date: ${p.date}` : '';
      marker.bindPopup(`<strong>${satLabel}</strong>${dateLabel}${instrLabel}<br/>Tile: ${p.tile_id || 'N/A'}<br/>Lat: ${lat.toFixed(4)}, Lon: ${lon.toFixed(4)}`);
      marker.addTo(s3Layer);
    });
    if (document.querySelector('#s3Toggle').checked) s3Layer.addTo(map);
    updateStatus(`✓ Sentinel-3: ${data.features.length} fire pixel(s)`, 'success');
    updateBadge('s3Count', data.features.length);
  } catch (err) {
    console.error('S3 fire pixels error:', err);
    updateStatus(`✗ Sentinel-3 error: ${err.message}`, 'error');
    updateBadge('s3Count', 0, 'ERR');
  }
}

async function loadS3SceneCount() {
  try {
    const res = await fetch('/api/sentinel3/scenes');
    if (!res.ok) return;
    const scenes = await res.json();
    const label = scenes.length === 0 ? '0 scenes' : `${scenes.length} scenes`;
    if (!document.querySelector('#s3Toggle').checked) updateBadge('s3Count', scenes.length, label);
  } catch (_) { /* non-critical */ }
}

// Update heatmap layer with Supercluster-based clustering
function updateHeatmap() {
  if (!allFireData || allFireData.length === 0) {
    updateBadge('heatmapCount', 0);
    if (heatmapLayer && map.hasLayer(heatmapLayer)) {
      map.removeLayer(heatmapLayer);
    }
    fireClusterIndex = null;
    return;
  }
  
  // Initialize Supercluster with fire data
  fireClusterIndex = new Supercluster({
    radius: 60,  // Cluster radius in pixels
    maxZoom: 16, // Max zoom to cluster points on
    minZoom: 0,
    minPoints: 2 // Minimum points to form a cluster
  });
  
  // Convert fire features to GeoJSON points for Supercluster
  const points = allFireData.map(f => ({
    type: 'Feature',
    geometry: f.geometry,
    properties: f.properties
  }));
  
  fireClusterIndex.load(points);
  
  updateBadge('heatmapCount', allFireData.length);
  
  // Initial render of clusters
  if (document.querySelector('#heatmapToggle').checked) {
    renderClusters();
  }
}

// Render clustered fire points on the map
function renderClusters() {
  if (!fireClusterIndex) return;
  
  // Remove old heatmap/cluster layer
  if (heatmapLayer && map.hasLayer(heatmapLayer)) {
    map.removeLayer(heatmapLayer);
  }
  heatmapLayer = L.layerGroup();
  
  const bounds = map.getBounds();
  const bbox = [
    bounds.getWest(),
    bounds.getSouth(),
    bounds.getEast(),
    bounds.getNorth()
  ];
  const zoom = map.getZoom();
  
  // Get clusters for current map view
  const clusters = fireClusterIndex.getClusters(bbox, Math.floor(zoom));
  
  clusters.forEach(cluster => {
    const [lon, lat] = cluster.geometry.coordinates;
    const props = cluster.properties;
    
    if (cluster.properties.cluster) {
      // This is a cluster
      const count = props.point_count;
      const size = count < 10 ? 30 : count < 50 ? 40 : count < 100 ? 50 : 60;
      
      const marker = L.marker([lat, lon], {
        icon: L.divIcon({
          html: `<div class="fire-cluster-icon" style="width:${size}px;height:${size}px;">${count}</div>`,
          className: 'fire-cluster-marker',
          iconSize: [size, size]
        })
      });

      // Zoom in on cluster click or keyboard activation
      marker.on('click', () => {
        const expansionZoom = fireClusterIndex.getClusterExpansionZoom(cluster.id);
        map.setView([lat, lon], expansionZoom);
      });
      marker.on('add', () => {
        const el = marker.getElement();
        if (!el) return;
        el.setAttribute('tabindex', '0');
        el.setAttribute('role', 'button');
        el.setAttribute('aria-label', `Fire cluster: ${count} detections`);
        el.addEventListener('keydown', (evt) => {
          if (evt.key === 'Enter' || evt.key === ' ') {
            evt.preventDefault();
            const expansionZoom = fireClusterIndex.getClusterExpansionZoom(cluster.id);
            map.setView([lat, lon], expansionZoom);
          }
        });
      });

      marker.addTo(heatmapLayer);
    } else {
      // Individual fire point
      const conf = props.confidence ?? 50;
      const frp = props.frp ?? 'N/A';
      const color = fireSeverityColor(conf);

      const marker = L.circleMarker([lat, lon], {
        radius: 8,
        color,
        fillColor: color,
        fillOpacity: 0.7,
        weight: 2
      });
      
      marker.bindPopup(`
        <strong>${props.source || 'VIIRS'}</strong><br/>
        Time: ${props.acq_time}<br/>
        Confidence: ${conf.toFixed(0)}%<br/>
        FRP: ${frp} MW
      `);
      
      marker.addTo(heatmapLayer);
    }
  });
  
  heatmapLayer.addTo(map);
}

// CSV export function with enhanced metadata
function exportToCSV() {
  if (!allFireData || allFireData.length === 0) {
    updateStatus('No fire data to export', 'warning');
    return;
  }
  
  const region = REGIONS[currentRegion];
  const days = parseInt(document.querySelector('#fireDays').value);
  const timestamp = new Date().toISOString();
  
  // Create CSV content with metadata header
  const metadata = [
    `# Fire Detection Data Export`,
    `# Generated: ${timestamp}`,
    `# Region: ${region.name}`,
    `# Time Period: ${days} day(s)`,
    `# Total Detections: ${allFireData.length}`,
    `# Data Sources: NASA FIRMS (VIIRS/MODIS NRT)`,
    `#`,
  ].join('\n');
  
  const headers = ['ID', 'Date/Time', 'Latitude', 'Longitude', 'Confidence (%)', 'FRP (MW)', 'Source', 'Region'];
  const rows = allFireData.map(f => {
    const [lon, lat] = f.geometry.coordinates;
    const props = f.properties;
    return [
      `"${props.id || 'N/A'}"`,
      `"${props.acq_time || 'N/A'}"`,
      lat.toFixed(6),
      lon.toFixed(6),
      (props.confidence || 0).toFixed(1),
      props.frp !== null && props.frp !== undefined ? props.frp.toFixed(2) : 'N/A',
      `"${props.source || 'VIIRS'}"`,
      `"${region.name}"`
    ].join(',');
  });
  
  const csv = metadata + '\n' + [headers.join(','), ...rows].join('\n');
  
  // Download CSV
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `fire-detections-${region.name.toLowerCase().replace(/\s+/g, '-')}-${days}d-${new Date().toISOString().split('T')[0]}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  
  updateStatus(`✓ Exported ${allFireData.length} fire detections to CSV`, 'success');
}

// Region switching function
async function switchRegion(regionKey) {
  const region = REGIONS[regionKey];
  if (!region) return;
  
  // Update current region
  currentRegion = regionKey;
  
  // Center and zoom map
  map.setView(region.center, region.zoom);
  updateStatus(`Switching to: ${region.name}...`);
  
  // Clear all fire layers
  viirsLayer.clearLayers();
  modisLayer.clearLayers();
  if (heatmapLayer) {
    map.removeLayer(heatmapLayer);
    heatmapLayer = null;
  }
  allFireData = [];
  fireClusterIndex = null;
  
  // Reset badges
  updateBadge('viirsCount', 0, '...');
  updateBadge('modisCount', 0, '...');
  updateBadge('heatmapCount', 0);
  
  // Get current time period
  const days = parseInt(document.querySelector('#fireDays').value);
  
  // Reload fire data for new region
  updateStatus(`Loading fire data for ${region.name}...`);
  
  // Load VIIRS if checked
  if (document.querySelector('#viirsToggle').checked) {
    await loadFIRMS('viirs', viirsLayer, days, region.bbox);
  }
  
  // Load MODIS if checked
  if (document.querySelector('#modisToggle').checked && map.hasLayer(modisLayer)) {
    await loadFIRMS('modis', modisLayer, days, region.bbox);
  }
  
  updateStatus(`✓ Switched to ${region.name}`, 'success');
}

// Helper function to refresh fire data
function refreshFireData() {
  const days = parseInt(document.querySelector('#fireDays').value);
  const region = REGIONS[currentRegion];
  updateStatus(`Refreshing fire data (${days} days)...`);
  
  if (document.querySelector('#viirsToggle').checked) {
    loadFIRMS('viirs', viirsLayer, days, region.bbox);
  }
  if (document.querySelector('#modisToggle').checked && map.hasLayer(modisLayer)) {
    loadFIRMS('modis', modisLayer, days, region.bbox);
  }
}

// Layer toggle handlers
document.querySelector('#viirsToggle').addEventListener('change', (e) => {
  if (e.target.checked) {
    viirsLayer.addTo(map);
  } else {
    map.removeLayer(viirsLayer);
  }
});

document.querySelector('#modisToggle').addEventListener('change', (e) => {
  if (e.target.checked) {
    if (!map.hasLayer(modisLayer)) {
      const days = parseInt(document.querySelector('#fireDays').value);
      const region = REGIONS[currentRegion];
      loadFIRMS('modis', modisLayer, days, region.bbox);
    }
    modisLayer.addTo(map);
  } else {
    map.removeLayer(modisLayer);
  }
});

document.querySelector('#frpToggle').addEventListener('change', (e) => {
  if (frpOverlay) {
    if (e.target.checked) {
      frpOverlay.addTo(map);
    } else {
      map.removeLayer(frpOverlay);
    }
  } else if (e.target.checked) {
    loadFRP();
  }
});

// Time period change handler
document.querySelector('#fireDays').addEventListener('change', () => {
  refreshFireData();
});

// Refresh button handler
document.querySelector('#refreshBtn').addEventListener('click', () => {
  refreshFireData();
  if (document.querySelector('#frpToggle').checked || frpOverlay) {
    loadFRP();
  }
});

// CSV export button handler
document.querySelector('#exportCsvBtn').addEventListener('click', () => {
  exportToCSV();
});

// Region selector handler
document.querySelector('#regionSelect').addEventListener('change', (e) => {
  switchRegion(e.target.value);
});

// Heatmap toggle handler
document.querySelector('#heatmapToggle').addEventListener('change', (e) => {
  if (e.target.checked) {
    if (fireClusterIndex) {
      renderClusters();
    } else {
      updateHeatmap();
    }
  } else {
    if (heatmapLayer) {
      map.removeLayer(heatmapLayer);
    }
  }
});

document.querySelector('#s3Toggle').addEventListener('change', (e) => {
  if (e.target.checked) {
    if (s3Layer.getLayers().length === 0) loadS3FirePixels();
    else s3Layer.addTo(map);
  } else {
    map.removeLayer(s3Layer);
  }
});

// Delegate ML button clicks via popupopen to avoid inline onclick (XSS fix)
map.on('popupopen', function(e) {
  const container = e.popup.getElement();
  if (!container) return;
  container.addEventListener('click', function(evt) {
    const btn = evt.target.closest('[data-popup-id]');
    if (!btn) return;
    const lat = parseFloat(btn.dataset.lat);
    const lon = parseFloat(btn.dataset.lon);
    const popupId = btn.dataset.popupId;
    const live = btn.dataset.live === 'true';
    checkMLFire(lat, lon, popupId, live);
  });
});

// Map move/zoom handlers to update clusters dynamically (debounced — 50ms)
const _debouncedRenderClusters = debounce(() => {
  if (fireClusterIndex && document.querySelector('#heatmapToggle').checked) {
    renderClusters();
  }
}, 50);

map.on('moveend', _debouncedRenderClusters);
map.on('zoomend', _debouncedRenderClusters);

// Sentinel-2 search
document.querySelector('#s2SearchBtn').addEventListener('click', async () => {
  const days = document.querySelector('#s2Days').value;
  const cloud = document.querySelector('#s2Cloud').value;
  const box = document.querySelector('#s2Results');
  
  box.innerHTML = '<p class="ml-loading">Searching Sentinel-2 imagery\u2026</p>';

  try {
    const res = await fetch(`/api/sentinel2/search?days=${days}&cloud_pct=${cloud}&limit=8`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    box.innerHTML = '';

    if (data.features && data.features.length > 0) {
      // Fetch and display thumbnails for each item
      for (const item of data.features.slice(0, 4)) {
        try {
          const thumbRes = await fetch(`/api/sentinel2/thumbnail?days=${days}&cloud_pct=${cloud}`);
          const tn = await thumbRes.json();

          if (tn.path) {
            const img = document.createElement('img');
            img.src = `/api/images/sentinel2/${tn.path.split('/').pop()}`;
            img.alt = `Sentinel-2 ${item.id}`;
            img.title = `${item.properties?.datetime || 'Unknown date'} - Cloud: ${item.properties?.['eo:cloud_cover']?.toFixed(1) || 'N/A'}%`;
            box.appendChild(img);
          }
        } catch (err) {
          console.error('Error fetching thumbnail:', err);
        }
      }

      if (box.children.length === 0) {
        box.innerHTML = '<p class="ml-loading">No thumbnails available for this search.</p>';
      }
    } else {
      box.innerHTML = '<p class="ml-loading">No imagery found \u2014 try extending the date range or increasing cloud cover threshold.</p>';
    }
  } catch (error) {
    console.error('Error searching Sentinel-2:', error);
    box.innerHTML = '<p class="ml-error">Search failed \u2014 check your network connection and try again.</p>';
  }
});

// Sidebar collapse toggle — persists state in localStorage
(function initSidebarToggle() {
  const toggle = document.getElementById('sidebar-toggle');
  const app = document.getElementById('app');
  if (!toggle || !app) return;

  function setCollapsed(collapsed) {
    app.classList.toggle('sidebar-collapsed', collapsed);
    toggle.textContent = collapsed ? '\u203a' : '\u2039';
    toggle.setAttribute('aria-expanded', String(!collapsed));
    toggle.setAttribute('aria-label', collapsed ? 'Expand sidebar' : 'Collapse sidebar');
    toggle.setAttribute('title', collapsed ? 'Expand sidebar' : 'Collapse sidebar');
    localStorage.setItem('sidebar-collapsed', String(collapsed));
  }

  // Restore persisted state
  const savedCollapsed = localStorage.getItem('sidebar-collapsed') === 'true';
  if (savedCollapsed) setCollapsed(true);

  toggle.addEventListener('click', () => {
    setCollapsed(!app.classList.contains('sidebar-collapsed'));
  });
})();

// Initial data load - fetch config and center map
async function initializeApp() {
  updateStatus('Application started. Loading configuration...');
  
  try {
    // Fetch server config to get current bbox/center from .env
    const configRes = await fetch('/api/config');
    const config = await configRes.json();
    
    // Default to Portugal mainland
    let detectedRegion = 'mainland';
    
    // Set the region selector to match
    document.querySelector('#regionSelect').value = detectedRegion;
    currentRegion = detectedRegion;
    const region = REGIONS[currentRegion];
    
    // Center map on detected region
    map.setView(region.center, region.zoom);
    updateStatus(`Detected region: ${region.name}`);
    
    // Load initial fire data with detected region's bbox
    const initialDays = parseInt(document.querySelector('#fireDays').value);
    loadFIRMS('viirs', viirsLayer, initialDays, region.bbox);
    loadFRP();
    loadS3SceneCount();
    loadMLStatus();

    // Add helpful info
    setTimeout(() => {
      updateStatus('Tip: Use the region selector to switch between different Portugal regions', 'info');
    }, 3000);
  } catch (error) {
    console.error('Error initializing app:', error);
    updateStatus('Error loading configuration, using default map view', 'warning');
    
    // Fallback to default behavior
    const region = REGIONS[currentRegion];
    const initialDays = parseInt(document.querySelector('#fireDays').value);
    loadFIRMS('viirs', viirsLayer, initialDays, region.bbox);
    loadFRP();
  }
}

// Load ML model status into sidebar — always shows the card
async function loadMLStatus() {
  const text = document.getElementById('mlStatusText');
  if (!text) return;
  try {
    const res = await fetch('/api/inference/status');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (data.loaded) {
      // Also fetch test metrics if available
      let metricsNote = '';
      try {
        const mr = await fetch('/api/inference/metrics');
        if (mr.ok) {
          const m = await mr.json();
          metricsNote = `<br/>Test F1: <strong class="ml-pct">${(m.f1 * 100).toFixed(1)}%</strong> &middot; AUC: <strong class="ml-pct">${(m.auc * 100).toFixed(1)}%</strong>`;
        }
      } catch (_) {}
      text.innerHTML = `
        <span class="ml-status-ok">&#10003; Model loaded</span><br/>
        ${data.tiles_indexed} tiles indexed &middot; threshold ${(data.threshold * 100).toFixed(0)}%${metricsNote}<br/>
        <span class="ml-status-hint">Click any fire marker &rarr; ML / Live ML Check</span>
      `;
    } else {
      text.innerHTML = `<span class="ml-status-hint">Model not loaded &mdash; restart app with <code>--checkpoint</code> flag or check <a href="/static/dataset.html">training status</a>.</span>`;
    }
  } catch (e) {
    text.innerHTML = `<span class="ml-status-hint">ML status unavailable &mdash; check <a href="/static/dataset.html">dataset page</a>.</span>`;
  }
}

// Render ML prediction result HTML into element el.
// statusLabel: "Preliminary (training tile)" | "Live (Sentinel-2)" | null
function _renderMLResult(el, data, statusLabel) {
  const source = data.source || 'training_tile';
  const pct = (data.fire_probability * 100).toFixed(1);
  const severityClass = data.fire_probability >= 0.8 ? 'ml-fire-high'
    : data.fire_probability >= 0.5 ? 'ml-fire-medium' : 'ml-fire-none';
  const label = data.is_fire ? 'FIRE CONFIRMED' : 'No fire detected';
  const severityText = data.fire_probability >= 0.8 ? '▲ High'
    : data.fire_probability >= 0.5 ? '◆ Medium' : '● Low';

  let distNote = '';
  if (source === 'live') {
    const sceneDate = data.scene_date ? ` · scene ${data.scene_date}` : '';
    distNote = `<br/><span class="ml-source-live">Live S2 tile${sceneDate}</span>`;
  } else if (data.tile_lat != null && data.tile_lon != null) {
    const lat = data._req_lat, lon = data._req_lon;
    let distKm = null;
    if (lat != null && lon != null) {
      const latRad = lat * Math.PI / 180;
      const dlat = (data.tile_lat - lat) * 111;
      const dlon = (data.tile_lon - lon) * 111 * Math.cos(latRad);
      distKm = Math.sqrt(dlat * dlat + dlon * dlon);
    }
    if (distKm !== null) {
      const warnClass = distKm > 50 ? 'ml-dist-warn' : 'ml-dist-ok';
      const warnIcon = distKm > 50 ? ' ⚠' : '';
      distNote = `<br/><span class="${warnClass}">Nearest tile: ${distKm.toFixed(1)} km away${warnIcon}</span>`;
      if (distKm > 50) {
        distNote += `<br/><span class="ml-dist-warn">Result may be unreliable at this distance</span>`;
      }
    }
  }

  const statusBadge = statusLabel
    ? `<span class="ml-status-badge">${statusLabel}</span><br/>`
    : '';

  el.innerHTML = `
    <div class="ml-result ${severityClass}" data-status="${statusLabel || 'final'}">
      ${statusBadge}
      <strong class="ml-label">ML: ${label}</strong>
      <span class="ml-severity-text">${severityText}</span><br/>
      Probability: <strong class="ml-pct">${pct}%</strong>
      <div class="ml-prob-bar" role="progressbar" aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100" aria-label="Fire probability ${pct}%">
        <div class="ml-prob-bar-fill"></div>
      </div>
      ${distNote}
    </div>`;
  requestAnimationFrame(() => {
    const fill = el.querySelector('.ml-prob-bar-fill');
    if (fill) fill.style.transform = `scaleX(${pct / 100})`;
  });
}

// ML fire inference check
async function checkMLFire(lat, lon, popupId, live = false) {
  const el = document.getElementById(popupId);
  if (!el) return;

  if (!live) {
    // Fast path: training tile only
    el.innerHTML = '<span class="ml-loading">Running model\u2026</span>';
    try {
      const res = await fetch('/api/inference/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lat, lon, live: false }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        el.innerHTML = `<span class="ml-error">ML error: ${err.detail || res.status}</span>`;
        return;
      }
      const data = await res.json();
      data._req_lat = lat; data._req_lon = lon;
      _renderMLResult(el, data, null);
    } catch (e) {
      el.innerHTML = `<span class="ml-error">ML error: ${e.message}</span>`;
    }
    return;
  }

  // Live path: show spinner, run training tile immediately as preliminary,
  // then kick off live fetch and replace result when done.
  el.innerHTML = '<span class="ml-loading"><span class="ml-spinner" aria-hidden="true"></span> Fetching S2 tile\u2026</span>';

  // Step 1: get preliminary training-tile result (fast, ~1s)
  let prelimData = null;
  try {
    const r = await fetch('/api/inference/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ lat, lon, live: false }),
    });
    if (r.ok) {
      prelimData = await r.json();
      prelimData._req_lat = lat; prelimData._req_lon = lon;
      _renderMLResult(el, prelimData, 'Preliminary (training tile)');
      // Append live-fetch spinner below preliminary result
      const spinner = document.createElement('div');
      spinner.className = 'ml-live-pending';
      spinner.innerHTML = '<span class="ml-spinner" aria-hidden="true"></span> Fetching live S2 tile\u2026';
      el.appendChild(spinner);
    }
  } catch (_) { /* non-critical — proceed to live fetch anyway */ }

  // Step 2: live fetch
  try {
    const res = await fetch('/api/inference/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ lat, lon, live: true }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      if (prelimData) {
        // Keep preliminary result, show live error below
        const existingResult = el.querySelector('.ml-result');
        if (existingResult) {
          const errSpan = document.createElement('div');
          errSpan.className = 'ml-live-error';
          errSpan.innerHTML = `<span class="ml-error">Live fetch failed: ${err.detail || res.status}</span>`;
          el.innerHTML = '';
          el.appendChild(existingResult);
          el.appendChild(errSpan);
        }
      } else {
        el.innerHTML = `<span class="ml-error">ML error: ${err.detail || res.status}</span>`;
      }
      return;
    }
    const data = await res.json();
    data._req_lat = lat; data._req_lon = lon;
    _renderMLResult(el, data, 'Live (Sentinel-2)');
  } catch (e) {
    if (prelimData) {
      // Live fetch failed but we have a preliminary — show it without the live-pending spinner
      _renderMLResult(el, prelimData, 'Preliminary (training tile)');
      const errDiv = document.createElement('div');
      errDiv.innerHTML = `<span class="ml-error">Live fetch error: ${e.message}</span>`;
      el.appendChild(errDiv);
    } else {
      el.innerHTML = `<span class="ml-error">ML error: ${e.message}</span>`;
    }
  }
}

// ─── Historical Fire Timeline ─────────────────────────────────
let historicalLayer = L.layerGroup();

function _ageColor(detectedAt) {
  const now = Date.now();
  const age = now - new Date(detectedAt).getTime();
  const days = age / (1000 * 60 * 60 * 24);
  // Newer = bright orange-red, older = faded gold
  const opacity = Math.max(0.2, 1 - days / 30);
  return { color: 'hsl(28,90%,48%)', opacity };
}

async function loadHistoricalFires() {
  const start = document.getElementById('histStart').value;
  const end = document.getElementById('histEnd').value;
  const source = document.getElementById('histSource').value;
  const statusEl = document.getElementById('histStatus');

  if (!start || !end) {
    statusEl.textContent = 'Select both start and end dates.';
    return;
  }

  statusEl.textContent = 'Loading historical data\u2026';
  historicalLayer.clearLayers();
  if (map.hasLayer(historicalLayer)) map.removeLayer(historicalLayer);

  let url = `/api/firms/history?start=${start}&end=${end}`;
  if (source) url += `&source=${source}`;
  const region = REGIONS[currentRegion];
  if (region && region.bbox) url += `&bbox=${region.bbox.join(',')}`;

  try {
    const res = await fetch(url);
    if (res.status === 503) {
      statusEl.textContent = 'Historical data unavailable (database not configured).';
      return;
    }
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    if (!data.features || data.features.length === 0) {
      statusEl.textContent = 'No historical fires found for this period.';
      return;
    }

    data.features.forEach(f => {
      const [lon, lat] = f.geometry.coordinates;
      const p = f.properties;
      const { color, opacity } = _ageColor(p.detected_at);
      const marker = L.circleMarker([lat, lon], {
        radius: 4,
        color,
        fillColor: color,
        fillOpacity: opacity,
        weight: 1,
        opacity,
      });
      marker.bindPopup(
        `<strong>Historical: ${p.source}</strong><br/>` +
        `Date: ${p.detected_at ? p.detected_at.slice(0, 10) : 'N/A'}<br/>` +
        `FRP: ${p.frp != null ? p.frp.toFixed(1) + ' MW' : 'N/A'}<br/>` +
        `Confidence: ${p.confidence || 'N/A'}`
      );
      marker.addTo(historicalLayer);
    });

    historicalLayer.addTo(map);
    statusEl.textContent = `${data.features.length} detections from ${start} to ${end}.`;
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
  }
}

// Set sensible defaults: last 7 days
(function initHistoryDates() {
  const today = new Date();
  const prior = new Date(today);
  prior.setDate(today.getDate() - 7);
  const fmt = d => d.toISOString().slice(0, 10);
  const s = document.getElementById('histStart');
  const e = document.getElementById('histEnd');
  if (s) s.value = fmt(prior);
  if (e) e.value = fmt(today);
})();

document.getElementById('histSearchBtn').addEventListener('click', loadHistoricalFires);
document.getElementById('histClearBtn').addEventListener('click', () => {
  historicalLayer.clearLayers();
  if (map.hasLayer(historicalLayer)) map.removeLayer(historicalLayer);
  document.getElementById('histStatus').textContent = '';
});

// ─── Area of Interest (AOI) ───────────────────────────────────
let _aoiDrawLayer = null;
let _drawControl = null;

async function loadAOIList() {
  const listEl = document.getElementById('aoiList');
  if (!listEl) return;
  try {
    const res = await fetch('/api/aoi');
    if (res.status === 503) {
      listEl.innerHTML = '<p class="ml-loading" style="font-size:11px">Database not configured.</p>';
      return;
    }
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const aois = await res.json();
    if (!aois.length) {
      listEl.innerHTML = '<p class="ml-loading" style="font-size:11px">No saved areas yet.</p>';
      return;
    }
    listEl.innerHTML = aois.map(a => `
      <div class="aoi-item" data-id="${a.id}">
        <span class="aoi-name">${a.name || 'Unnamed AOI'}</span>
        <span class="aoi-email">${a.email ? '\uD83D\uDCE7 ' + a.email : ''}</span>
        <div class="aoi-actions">
          <button class="aoi-fires-btn outline" data-id="${a.id}" title="Show fires in this area">Fires</button>
          <button class="aoi-delete-btn outline" data-id="${a.id}" title="Delete AOI">\u00D7</button>
        </div>
      </div>`).join('');
  } catch (e) {
    listEl.innerHTML = `<p class="ml-error" style="font-size:11px">Failed to load AOIs: ${e.message}</p>`;
  }
}

async function deleteAOI(id) {
  try {
    const res = await fetch(`/api/aoi/${id}`, { method: 'DELETE' });
    if (!res.ok && res.status !== 204) throw new Error(`HTTP ${res.status}`);
    await loadAOIList();
    updateStatus('AOI deleted.', 'success');
  } catch (e) {
    updateStatus(`Failed to delete AOI: ${e.message}`, 'error');
  }
}

async function showAOIFires(id) {
  updateStatus('Loading fires for AOI\u2026');
  try {
    const res = await fetch(`/api/aoi/${id}/fires`);
    if (res.status === 503) { updateStatus('Database not configured.', 'warning'); return; }
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    historicalLayer.clearLayers();
    if (map.hasLayer(historicalLayer)) map.removeLayer(historicalLayer);
    if (!data.features || !data.features.length) {
      updateStatus('No recent fires in this AOI.', 'info'); return;
    }
    data.features.forEach(f => {
      const [lon, lat] = f.geometry.coordinates;
      const p = f.properties;
      const { color, opacity } = _ageColor(p.detected_at);
      L.circleMarker([lat, lon], { radius: 4, color, fillColor: color, fillOpacity: opacity, weight: 1 })
        .bindPopup(`<strong>${p.source}</strong><br/>${p.detected_at ? p.detected_at.slice(0,10) : ''}`)
        .addTo(historicalLayer);
    });
    historicalLayer.addTo(map);
    updateStatus(`\u2713 ${data.count} fire(s) in AOI`, 'success');
  } catch (e) {
    updateStatus(`AOI fires error: ${e.message}`, 'error');
  }
}

// Delegate AOI button clicks
document.getElementById('aoiList').addEventListener('click', e => {
  const del = e.target.closest('.aoi-delete-btn');
  if (del) { deleteAOI(del.dataset.id); return; }
  const fires = e.target.closest('.aoi-fires-btn');
  if (fires) { showAOIFires(fires.dataset.id); }
});

document.getElementById('aoiDrawBtn').addEventListener('click', () => {
  if (!window.L || !L.Draw) { updateStatus('Leaflet.draw not loaded.', 'warning'); return; }

  if (_drawControl) {
    map.removeControl(_drawControl);
    _drawControl = null;
  }
  if (_aoiDrawLayer) {
    map.removeLayer(_aoiDrawLayer);
    _aoiDrawLayer = null;
  }

  _aoiDrawLayer = new L.FeatureGroup().addTo(map);
  _drawControl = new L.Control.Draw({
    draw: { polygon: true, polyline: false, rectangle: false, circle: false, marker: false, circlemarker: false },
    edit: { featureGroup: _aoiDrawLayer },
  });
  map.addControl(_drawControl);
  updateStatus('Draw a polygon on the map to create an AOI.', 'info');
});

map.on(L.Draw ? L.Draw.Event.CREATED : 'draw:created', async (e) => {
  if (!_aoiDrawLayer) return;
  _aoiDrawLayer.clearLayers();
  _aoiDrawLayer.addLayer(e.layer);

  const geojson = e.layer.toGeoJSON().geometry;
  const name = prompt('Name for this AOI (optional):') || '';
  const email = prompt('Email for fire alerts (optional):') || '';

  try {
    const res = await fetch('/api/aoi', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: name || null, email: email || null, boundary: geojson }),
    });
    if (res.status === 503) { updateStatus('Database not configured — AOI not saved.', 'warning'); return; }
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    updateStatus('\u2713 AOI saved.', 'success');
    await loadAOIList();
    if (_drawControl) { map.removeControl(_drawControl); _drawControl = null; }
  } catch (err) {
    updateStatus(`Failed to save AOI: ${err.message}`, 'error');
  }
});

// Start the application
(async () => {
  await initDemoMode();
  initializeApp();
  loadAOIList();
})();
