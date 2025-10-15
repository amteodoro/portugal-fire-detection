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
async function loadFIRMS(product, layer, days = 1) {
  const productName = product.toUpperCase();
  const badgeId = product === 'viirs' ? 'viirsCount' : 'modisCount';
  
  try {
    updateStatus(`Loading ${productName} data (${days} days)...`);
    updateBadge(badgeId, 0, '...');
    
    const res = await fetch(`/api/firms/${product}?days=${days}`);
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

// Helper function to refresh fire data
function refreshFireData() {
  const days = parseInt(document.querySelector('#fireDays').value);
  updateStatus(`Refreshing fire data (${days} days)...`);
  
  if (document.querySelector('#viirsToggle').checked) {
    loadFIRMS('viirs', viirsLayer, days);
  }
  if (document.querySelector('#modisToggle').checked && map.hasLayer(modisLayer)) {
    loadFIRMS('modis', modisLayer, days);
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
      loadFIRMS('modis', modisLayer, days);
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

// Initial data load
updateStatus('Application started. Loading initial data...');
const initialDays = parseInt(document.querySelector('#fireDays').value);
loadFIRMS('viirs', viirsLayer, initialDays);
loadFRP();

// Add helpful info
setTimeout(() => {
  updateStatus('Tip: Use the time period selector to view historical fire data', 'info');
}, 3000);
