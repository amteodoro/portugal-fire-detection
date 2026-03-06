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

// Status tracking
function updateStatus(message, type = 'info') {
  const statusBox = document.querySelector('#statusMessages');
  const timestamp = new Date().toLocaleTimeString();
  const className = type === 'success' ? 'success' : type === 'warning' ? 'warning' : type === 'error' ? 'error' : '';
  statusBox.innerHTML = `<div class="${className}">[${timestamp}] ${message}</div>` + statusBox.innerHTML;
  // Keep only last 5 messages
  const messages = statusBox.querySelectorAll('div');
  if (messages.length > 5) {
    messages[messages.length - 1].remove();
  }
  console.log(`[${type.toUpperCase()}] ${message}`);
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
    
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    }
    
    const data = await res.json();
    layer.clearLayers();
    
    if (data.features && data.features.length > 0) {
      data.features.forEach(f => {
        const [lon, lat] = f.geometry.coordinates;
        const conf = f.properties.confidence ?? 0;
        const frp = f.properties.frp ?? 'N/A';
        const color = conf >= 80 ? 'red' : conf >= 50 ? 'orange' : 'yellow';
        
        const marker = L.circleMarker([lat, lon], {
          radius: 5,
          color: color,
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
          <hr style="margin:6px 0; border-color:#ddd"/>
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
          html: '<svg width="14" height="14" viewBox="0 0 14 14"><polygon points="7,1 13,5 11,13 3,13 1,5" fill="#e53935" stroke="#fff" stroke-width="1.5"/></svg>',
          className: '', iconSize: [14, 14], iconAnchor: [7, 7],
        }),
      });
      marker.bindPopup(`<strong>Sentinel-3</strong><br/>Tile: ${p.tile_id || 'N/A'}<br/>Scene: ${p.scene_id || 'N/A'}<br/>Lat: ${lat.toFixed(4)}, Lon: ${lon.toFixed(4)}`);
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
          html: `<div style="background-color: rgba(255, 87, 34, 0.8); width: ${size}px; height: ${size}px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; border: 3px solid rgba(255, 255, 255, 0.8);">${count}</div>`,
          className: 'fire-cluster-marker',
          iconSize: [size, size]
        })
      });
      
      // Zoom in on cluster click
      marker.on('click', () => {
        const expansionZoom = fireClusterIndex.getClusterExpansionZoom(cluster.id);
        map.setView([lat, lon], expansionZoom);
      });
      
      marker.addTo(heatmapLayer);
    } else {
      // Individual fire point
      const conf = props.confidence ?? 50;
      const frp = props.frp ?? 'N/A';
      const color = conf >= 80 ? '#d32f2f' : conf >= 50 ? '#ff6f00' : '#ffa000';
      
      const marker = L.circleMarker([lat, lon], {
        radius: 8,
        color: color,
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

// Map move/zoom handlers to update clusters dynamically
map.on('moveend', () => {
  if (fireClusterIndex && document.querySelector('#heatmapToggle').checked) {
    renderClusters();
  }
});

map.on('zoomend', () => {
  if (fireClusterIndex && document.querySelector('#heatmapToggle').checked) {
    renderClusters();
  }
});

// Sentinel-2 search
document.querySelector('#s2SearchBtn').addEventListener('click', async () => {
  const days = document.querySelector('#s2Days').value;
  const cloud = document.querySelector('#s2Cloud').value;
  const box = document.querySelector('#s2Results');
  
  box.innerHTML = '<p>Searching...</p>';
  
  try {
    const res = await fetch(`/api/sentinel2/search?days=${days}&cloud_pct=${cloud}&limit=8`);
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
        box.innerHTML = '<p>No thumbnails available</p>';
      }
    } else {
      box.innerHTML = '<p>No results found</p>';
    }
  } catch (error) {
    console.error('Error searching Sentinel-2:', error);
    box.innerHTML = '<p>Error searching</p>';
  }
});

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

// Load ML model status into sidebar
async function loadMLStatus() {
  try {
    const res = await fetch('/api/inference/status');
    if (!res.ok) return;
    const data = await res.json();
    const card = document.getElementById('mlStatusCard');
    const text = document.getElementById('mlStatusText');
    if (!card || !text) return;
    if (data.loaded) {
      card.style.display = '';
      // Also fetch test metrics if available
      let metricsNote = '';
      try {
        const mr = await fetch('/api/inference/metrics');
        if (mr.ok) {
          const m = await mr.json();
          metricsNote = `<br/>Test F1: <strong>${(m.f1 * 100).toFixed(1)}%</strong> &middot; AUC: ${(m.auc * 100).toFixed(1)}%`;
        }
      } catch (_) {}
      text.innerHTML = `
        <span class="ml-status-ok">&#10003; Model loaded</span><br/>
        ${data.tiles_indexed} tiles indexed &middot; threshold ${(data.threshold * 100).toFixed(0)}%${metricsNote}<br/>
        <span class="ml-status-hint">Click any fire marker &rarr; ML / Live ML Check</span>
      `;
    }
  } catch (e) { /* model not available */ }
}

// ML fire inference check
async function checkMLFire(lat, lon, popupId, live = false) {
  const el = document.getElementById(popupId);
  if (!el) return;
  el.innerHTML = live
    ? '<span class="ml-loading">Fetching live S2 tile… (~5–10s)</span>'
    : '<span class="ml-loading">Running model...</span>';
  try {
    const res = await fetch('/api/inference/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ lat, lon, live }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      el.innerHTML = `<span class="ml-error">ML error: ${err.detail || res.status}</span>`;
      return;
    }
    const data = await res.json();
    const source = data.source || 'training_tile';
    const pct = (data.fire_probability * 100).toFixed(1);
    const severityClass = data.fire_probability >= 0.8 ? 'ml-fire-high'
      : data.fire_probability >= 0.5 ? 'ml-fire-medium' : 'ml-fire-none';
    const label = data.is_fire ? 'FIRE CONFIRMED' : 'No fire detected';
    // Text label for colorblind users (shape/text, not color alone)
    const severityText = data.fire_probability >= 0.8 ? '▲ High'
      : data.fire_probability >= 0.5 ? '◆ Medium' : '● Low';

    // Correct haversine-approximated distance (longitude degree varies with latitude)
    let distKm = null;
    if (data.tile_lat != null && data.tile_lon != null) {
      const latRad = lat * Math.PI / 180;
      const dlat = (data.tile_lat - lat) * 111;
      const dlon = (data.tile_lon - lon) * 111 * Math.cos(latRad);
      distKm = Math.sqrt(dlat * dlat + dlon * dlon);
    }

    let distNote = '';
    if (source === 'live') {
      const sceneDate = data.scene_date ? ` · scene ${data.scene_date}` : '';
      distNote = `<br/><span class="ml-source-live">Live S2 tile${sceneDate}</span>`;
    } else if (distKm !== null) {
      const warnClass = distKm > 50 ? 'ml-dist-warn' : 'ml-dist-ok';
      const warnIcon = distKm > 50 ? ' ⚠' : '';
      distNote = `<br/><span class="${warnClass}">Nearest tile: ${distKm.toFixed(1)} km away${warnIcon}</span>`;
      if (distKm > 50) {
        distNote += `<br/><span class="ml-dist-warn">Result may be unreliable at this distance</span>`;
      }
    }

    el.innerHTML = `
      <div class="ml-result ${severityClass}">
        <strong class="ml-label">ML: ${label}</strong>
        <span class="ml-severity-text">${severityText}</span><br/>
        Probability: <strong>${pct}%</strong>
        <div class="ml-prob-bar" role="progressbar" aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100" aria-label="Fire probability ${pct}%">
          <div class="ml-prob-bar-fill"></div>
        </div>
        ${distNote}
      </div>`;
    // Trigger CSS transition on the bar after DOM insertion
    requestAnimationFrame(() => {
      const fill = el.querySelector('.ml-prob-bar-fill');
      if (fill) fill.style.width = `${pct}%`;
    });
  } catch (e) {
    el.innerHTML = `<span class="ml-error">ML error: ${e.message}</span>`;
  }
}

// Start the application
initializeApp();
