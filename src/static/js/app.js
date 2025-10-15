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
let allFireData = []; // Store all fire data for heatmap and CSV

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
        
        marker.bindPopup(`
          <strong>${productName}</strong><br/>
          Time: ${f.properties.acq_time}<br/>
          Confidence: ${conf.toFixed(0)}%<br/>
          FRP: ${frp} MW
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

// Update heatmap layer
function updateHeatmap() {
  if (!allFireData || allFireData.length === 0) {
    updateBadge('heatmapCount', 0);
    if (heatmapLayer && map.hasLayer(heatmapLayer)) {
      map.removeLayer(heatmapLayer);
    }
    return;
  }
  
  // Convert fire data to heatmap format [lat, lng, intensity]
  const heatData = allFireData.map(f => {
    const [lon, lat] = f.geometry.coordinates;
    const intensity = (f.properties.confidence || 50) / 100; // Normalize confidence to 0-1
    return [lat, lon, intensity];
  });
  
  // Remove old heatmap
  if (heatmapLayer && map.hasLayer(heatmapLayer)) {
    map.removeLayer(heatmapLayer);
  }
  
  // Create new heatmap using canvas overlay
  heatmapLayer = L.layerGroup();
  
  // Simple circle-based heatmap (fallback if Leaflet.heat not available)
  heatData.forEach(([lat, lon, intensity]) => {
    L.circle([lat, lon], {
      radius: 5000,
      fillColor: intensity > 0.7 ? '#ff0000' : intensity > 0.4 ? '#ff6600' : '#ffaa00',
      fillOpacity: 0.3,
      stroke: false
    }).addTo(heatmapLayer);
  });
  
  updateBadge('heatmapCount', allFireData.length);
  
  if (document.querySelector('#heatmapToggle').checked) {
    heatmapLayer.addTo(map);
  }
}

// CSV export function
function exportToCSV() {
  if (!allFireData || allFireData.length === 0) {
    updateStatus('No fire data to export', 'warning');
    return;
  }
  
  // Create CSV content
  const headers = ['Date/Time', 'Latitude', 'Longitude', 'Confidence (%)', 'FRP (MW)', 'Source'];
  const rows = allFireData.map(f => {
    const [lon, lat] = f.geometry.coordinates;
    const props = f.properties;
    return [
      props.acq_time || 'N/A',
      lat.toFixed(6),
      lon.toFixed(6),
      (props.confidence || 0).toFixed(1),
      (props.frp || 'N/A'),
      props.source || 'VIIRS'
    ].join(',');
  });
  
  const csv = [headers.join(','), ...rows].join('\n');
  
  // Download CSV
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `portugal-fires-${new Date().toISOString().split('T')[0]}.csv`;
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
    if (heatmapLayer) {
      heatmapLayer.addTo(map);
    } else {
      updateHeatmap();
    }
  } else {
    if (heatmapLayer) {
      map.removeLayer(heatmapLayer);
    }
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
    
    // Try to detect which region matches the configured bbox
    let detectedRegion = 'mainland'; // default
    if (config.bbox) {
      const [cfg_lon_min, cfg_lat_min, cfg_lon_max, cfg_lat_max] = config.bbox;
      // Check if bbox matches California
      if (cfg_lon_min < -100) { // Simple heuristic: California is west of -100
        detectedRegion = 'california';
      }
    }
    
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
    
    // Add helpful info
    setTimeout(() => {
      updateStatus('Tip: Use the region selector to switch between Portugal and California', 'info');
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

// Start the application
initializeApp();
