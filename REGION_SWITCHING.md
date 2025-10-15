# Region Switching Guide

## Dynamic Region Selection via UI

The application now supports **dynamic region switching** directly from the web interface!

### Available Regions

#### Portugal Regions:
- **Mainland Portugal** - Main Portuguese territory
- **Madeira** - Madeira archipelago
- **Açores** - Azores archipelago  
- **All Portugal Regions** - Combined view

#### Test Regions (for verification):
- **California, USA** - High fire activity region for testing
- **Western USA** - Broader western states view

### How to Use

1. **Open the web interface**: http://localhost:8000

2. **Select a region** from the dropdown at the top of the sidebar

3. **Watch as the app automatically:**
   - Centers the map on the selected region
   - Clears old fire data
   - Loads fresh fire data for the new region
   - Updates all badges and status messages

### Features

✅ **No restart required** - Switch regions instantly  
✅ **Automatic data refresh** - Fire data reloads for new region  
✅ **Preserved settings** - Time period and layer toggles are maintained  
✅ **Smart detection** - Auto-detects current region from .env on startup  

### Technical Details

- The app detects the configured bbox from `.env` on startup
- If longitude is west of -100°, it auto-selects California
- Otherwise defaults to Mainland Portugal
- Each region has pre-configured bbox, center point, and zoom level
- Fire data API calls include the bbox parameter dynamically

### Example Usage

**Scenario 1**: Testing the system with active fires
1. Select "California, USA" from dropdown
2. You should immediately see ~48 fire markers
3. Click markers to see fire details

**Scenario 2**: Monitoring Portugal during fire season
1. Select "Mainland Portugal" from dropdown  
2. Map centers on Portugal
3. Fire data loads for Portuguese territory

**Scenario 3**: Monitoring island territories
1. Select "Madeira" or "Açores"
2. Map zooms to island region
3. Fire data filtered to that specific area

### Tips

- The `.env` file now only sets the **default** region on startup
- You can switch regions as many times as you want in the UI
- All regions use the same time period selector (1 day to 1 year)
- Export CSV works for whichever region is currently active
- Heatmap displays fires from the current region

### No More Manual Configuration!

**Before**: Had to edit `.env`, restart container, refresh browser  
**Now**: Just use the dropdown selector in the UI! 🎉
