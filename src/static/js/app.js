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

// Load FIRMS fire data (VIIRS or MODIS)
async function loadFIRMS(product, layer) {
  try {
    const res = await fetch(`/api/firms/${product}`);
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
          <strong>${product.toUpperCase()}</strong><br/>
          Time: ${f.properties.acq_time}<br/>
          Confidence: ${conf.toFixed(0)}%<br/>
          FRP: ${frp}
        `);
        
        marker.addTo(layer);
      });
      console.log(`Loaded ${data.features.length} ${product} fire points`);
    } else {
      console.log(`No ${product} fires detected`);
    }
  } catch (error) {
    console.error(`Error loading ${product}:`, error);
  }
}

// Load Meteosat FRP overlay
async function loadFRP() {
  try {
    const res = await fetch('/api/eumetsat/frp/latest');
    const meta = await res.json();
    
    if (frpOverlay) {
      map.removeLayer(frpOverlay);
      frpOverlay = null;
    }
    
    if (meta && meta.path && meta.bbox) {
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
      console.log('Loaded Meteosat FRP overlay');
    } else {
      console.log('No FRP overlay available');
    }
  } catch (error) {
    console.error('Error loading FRP:', error);
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
      loadFIRMS('modis', modisLayer);
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
console.log('Loading initial data...');
loadFIRMS('viirs', viirsLayer);
loadFRP();
