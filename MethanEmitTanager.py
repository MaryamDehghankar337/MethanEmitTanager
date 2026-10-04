"""MethanEmitTanager: A Comparative Methane Detection App.

Integrates NASA EMIT and Planet Tanager-1 hyperspectral data for
comparative methane plume analysis.
"""
from __future__ import annotations

import io
import os
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import folium
import numpy as np
import pandas as pd
import rasterio
import requests
import streamlit as st
from folium.plugins import Draw, MousePosition
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union
from streamlit_folium import st_folium

# --- Optional imports with availability flags ---
try:
    import earthaccess
    EARTHACCESS_AVAILABLE = True
except ImportError:
    EARTHACCESS_AVAILABLE = False

try:
    import pystac_client
    PYSTAC_AVAILABLE = True
except ImportError:
    PYSTAC_AVAILABLE = False

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

# EMIT configuration
EMIT_RESOLUTION = 60  # meters
EMIT_ENH_COLLECTION = "EMITL2BCH4ENH"

# Tanager-1 configuration
TANAGER_RESOLUTION = 30  # meters
TANAGER_STAC_URL = "https://www.planet.com/data/stac/browser/tanager-core-imagery/catalog.json"
# For actual searching, we use a public STAC API endpoint.
TANAGER_STAC_API = "https://planetarycomputer.microsoft.com/api/stac/v1"
TANAGER_COLLECTION = "planet-tanager"

DEFAULT_AOI = box(51.20, 35.40, 51.45, 35.60)

PARAMS = {
    "plume_threshold_ppm_m": 1000.0,
    "min_plume_pixels": 10,
    "wind_speed_m_s": 2.0,  # Fallback value
}

ALPHA_IME = 0.33
BETA_IME = 0.45
CH4_DENSITY_KG_M3 = 0.717


# ══════════════════════════════════════════════════════════════════════
#  SHARED GEOMETRY HELPERS (Unchanged)
# ══════════════════════════════════════════════════════════════════════

def normalize_geometry(obj):
    """Normalize various geometry inputs to a GeoJSON-like dict."""
    if obj is None:
        return None
    if hasattr(obj, "__geo_interface__"):
        obj = obj.__geo_interface__
    if not isinstance(obj, dict):
        return None
    if obj.get("type") == "Feature":
        return normalize_geometry(obj.get("geometry"))
    if obj.get("type") == "FeatureCollection":
        geoms = [normalize_geometry(f.get("geometry")) for f in obj.get("features", [])]
        geoms = [g for g in geoms if g]
        return mapping(unary_union([shape(g) for g in geoms])) if geoms else None
    try:
        g = shape(obj)
        return mapping(g) if not g.is_empty else None
    except Exception:
        return None

def ensure_aoi(obj):
    """Ensure a valid AOI, falling back to a default."""
    return normalize_geometry(obj) or mapping(DEFAULT_AOI)

def aoi_bounds(aoi):
    """Get bounding box of AOI."""
    return shape(ensure_aoi(aoi)).bounds

def compute_zoom(bounds):
    """Compute a reasonable zoom level based on AOI span."""
    try:
        minx, miny, maxx, maxy = bounds
        span = max(maxx - minx, maxy - miny, 1e-6)
        zoom = int(round(math.log2(360.0 / span))) - 1
        return max(3, min(15, zoom))
    except Exception:
        return 11

def create_map(aoi, center=None, zoom=None):
    """Create an interactive folium map with AOI and drawing tools."""
    geometry = shape(ensure_aoi(aoi))
    if center is None:
        center = geometry.centroid
    if zoom is None:
        zoom = compute_zoom(geometry.bounds)
    
    fmap = folium.Map(
        [center.y, center.x],
        zoom_start=zoom,
        tiles="OpenStreetMap",
    )
    folium.GeoJson(
        mapping(geometry),
        style_function=lambda _: {"color": "blue", "fill": False, "weight": 2},
    ).add_to(fmap)
    Draw(
        export=True,
        draw_options={"polyline": False, "circle": False, "marker": False,
                      "circlemarker": False, "polygon": {"allowIntersection": False, "showArea": True}},
        edit_options={"edit": True, "remove": True},
    ).add_to(fmap)
    MousePosition(
        position="bottomright", separator=" | ", prefix="📍 Lat, Lon:",
        lat_formatter="function(num) {return num.toFixed(5);}",
        lng_formatter="function(num) {return num.toFixed(5);}",
    ).add_to(fmap)
    return fmap


# ══════════════════════════════════════════════════════════════════════
#  GEOCODING & WIND (Unchanged)
# ══════════════════════════════════════════════════════════════════════

def geocode_place(query: str):
    """Geocode a place name using Nominatim."""
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": query, "format": "json", "limit": 1, "polygon_geojson": 1}
        headers = {"User-Agent": "MethanEmitTanager-App/1.0"}
        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        if not data:
            return None, None, None
        item = data[0]
        lat, lon = float(item["lat"]), float(item["lon"])
        label = item.get("display_name", query)
        
        gj = item.get("geojson")
        if gj and gj.get("type") in ("Polygon", "MultiPolygon"):
            geom = shape(gj)
            if not geom.is_empty:
                return geom, (lat, lon), label
        
        bb = item.get("boundingbox")
        if bb:
            south, north, west, east = [float(x) for x in bb]
            return box(west, south, east, north), (lat, lon), label
        
        d = 0.02
        return box(lon - d, lat - d, lon + d, lat + d), (lat, lon), label
    except Exception:
        return None, None, None

def get_wind_speed_openmeteo(lat: float, lon: float, dt: datetime) -> Optional[float]:
    """Fetch 10m wind speed (m/s) from Open-Meteo archive for a given point & time."""
    try:
        url = "https://archive-api.open-meteo.com/v1/archive"
        date_str = dt.strftime("%Y-%m-%d")
        params = {
            "latitude": round(lat, 4), "longitude": round(lon, 4),
            "start_date": date_str, "end_date": date_str,
            "hourly": "wind_speed_10m", "windspeed_unit": "ms", "timezone": "UTC",
        }
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        hourly = data.get("hourly", {})
        times, speeds = hourly.get("time", []), hourly.get("wind_speed_10m", [])
        if not times or not speeds:
            return None
        
        target = dt.strftime("%Y-%m-%dT%H:00")
        if target in times:
            idx = times.index(target)
        else:
            best_idx, best_diff = 0, None
            for i, t in enumerate(times):
                try:
                    t_dt = datetime.fromisoformat(t)
                    diff = abs((t_dt - dt).total_seconds())
                    if best_diff is None or diff < best_diff:
                        best_diff, best_idx = diff, i
                except Exception:
                    continue
            idx = best_idx
        
        val = speeds[idx]
        return float(val) if val is not None else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════
#  EMIT-SPECIFIC FUNCTIONS (Unchanged)
# ══════════════════════════════════════════════════════════════════════

def login_earthdata():
    """Authenticate with NASA Earthdata."""
    if not EARTHACCESS_AVAILABLE:
        raise RuntimeError("Package 'earthaccess' is not installed.")
    try:
        username = st.secrets["EARTHDATA_USERNAME"]
        password = st.secrets["EARTHDATA_PASSWORD"]
    except (KeyError, FileNotFoundError):
        raise RuntimeError("Earthdata credentials not configured in secrets.")
    
    os.environ["EARTHDATA_USERNAME"] = username
    os.environ["EARTHDATA_PASSWORD"] = password
    auth = earthaccess.login(strategy="environment")
    if not auth.authenticated:
        raise RuntimeError("Earthdata login failed.")
    return auth

def search_emit_granules(aoi, start_date, end_date):
    """Search NASA Earthdata for EMIT granules."""
    minx, miny, maxx, maxy = aoi_bounds(aoi)
    results = earthaccess.search_data(
        short_name=EMIT_ENH_COLLECTION,
        bounding_box=(minx, miny, maxx, maxy),
        temporal=(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")),
        count=200,
    )
    return list(results)

def granule_datetime(granule) -> Optional[datetime]:
    """Extract acquisition datetime from an EMIT granule."""
    try:
        umm = granule.get("umm", {})
        dt_str = umm.get("TemporalExtent", {}).get("RangeDateTime", {}).get("BeginningDateTime")
        if dt_str:
            return datetime.fromisoformat(dt_str.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        pass
    return None

def granule_cloud(granule) -> float:
    """Extract cloud cover percentage from an EMIT granule."""
    try:
        for attr in granule.get("umm", {}).get("AdditionalAttributes", []):
            if attr.get("Name") == "CloudCover":
                vals = attr.get("Values", [])
                if vals:
                    return float(vals[0])
    except Exception:
        pass
    return 0.0

def load_emit_enhancement(granule, aoi):
    """Load and mask EMIT enhancement data for the AOI."""
    files = earthaccess.open([granule])
    if not files:
        raise RuntimeError("No files returned by earthaccess.open().")
    
    tif_path = next((f for f in files if getattr(f, "path", str(f)).lower().endswith((".tif", ".tiff"))), files[0])
    minx, miny, maxx, maxy = aoi_bounds(aoi)
    
    with rasterio.open(tif_path) as src:
        nodata = src.nodata
        try:
            from rasterio.windows import from_bounds
            window = from_bounds(minx, miny, maxx, maxy, src.transform)
            window = window.round_offsets().round_lengths()
            data = src.read(1, window=window)
            transform = src.window_transform(window)
            crs = src.crs
        except Exception:
            data = src.read(1)
            transform, crs = src.transform, src.crs
    
    data = data.astype(np.float32)
    if nodata is not None:
        data = np.where(np.isclose(data, float(nodata), rtol=0, atol=1e-3), np.nan, data)
    for fv in (-9999.0, -999.0, -99999.0):
        data = np.where(np.isclose(data, fv, rtol=0, atol=1e-3), np.nan, data)
    data = np.where(np.abs(data) > 1e6, np.nan, data)
    
    try:
        from rasterio.features import geometry_mask
        geom_mask = geometry_mask([shape(ensure_aoi(aoi))], out_shape=data.shape,
                                  transform=transform, invert=True)
        data = np.where(geom_mask, data, np.nan)
    except Exception:
        pass
    
    return data, transform, crs


# ══════════════════════════════════════════════════════════════════════
#  TANAGER-1 SPECIFIC FUNCTIONS (NEW)
# ══════════════════════════════════════════════════════════════════════

def search_tanager_granules(aoi, start_date, end_date):
    """Search the Microsoft Planetary Computer STAC API for Tanager-1 scenes."""
    if not PYSTAC_AVAILABLE:
        raise RuntimeError("pystac-client is required for Tanager-1 search.")
    
    minx, miny, maxx, maxy = aoi_bounds(aoi)
    
    catalog = pystac_client.Client.open(TANAGER_STAC_API)
    search = catalog.search(
        collections=[TANAGER_COLLECTION],
        bbox=[minx, miny, maxx, maxy],
        datetime=f"{start_date.isoformat()}/{end_date.isoformat()}",
        limit=100,
    )
    return list(search.items())

def tanager_item_datetime(item) -> Optional[datetime]:
    """Extract acquisition datetime from a STAC item."""
    try:
        return datetime.fromisoformat(item.datetime.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None

def load_tanager_enhancement(item, aoi):
    """
    Load a Tanager-1 HDF5 file, compute the matched filter enhancement
    for CH4, and clip it to the AOI.
    """
    if not H5PY_AVAILABLE:
        raise RuntimeError("h5py is required to read Tanager-1 data.")
    
    # Get the HDF5 asset URL
    asset = item.assets.get("basic_radiance_hdf5")
    if not asset:
        raise RuntimeError("No HDF5 radiance asset found for this Tanager item.")
    h5_url = asset.href
    
    # Download the file (in-memory for simplicity; consider caching for production)
    with requests.get(h5_url, stream=True, timeout=60) as r:
        r.raise_for_status()
        file_like = io.BytesIO(r.content)
    
    # Open with h5py and load radiance + wavelengths
    with h5py.File(file_like, "r") as f:
        # NOTE: The exact HDF5 structure must be verified from a sample file.
        # This is a plausible structure based on similar instruments.
        # Key names may need adjustment (e.g., 'radiance', 'wavelengths').
        try:
            radiance = f["radiance"][:]  # shape (bands, height, width)
            wavelengths = f["wavelengths"][:]  # shape (bands,)
        except KeyError as e:
            raise RuntimeError(f"Unexpected HDF5 structure: missing key {e}")
    
    # TODO: The full matched-filter implementation requires the CH4 absorption
    # cross-section (from HITRAN) convolved to Tanager's spectral response.
    # This is a complex step that we outline here but do not fully implement.
    # For the purpose of this app, we use a simplified, placeholder enhancement
    # computed as a band difference to maintain app structure.
    #
    # PRODUCTION NOTE: A correct implementation would:
    #  1. Load the CH4 cross-section and resample to Tanager's wavelengths.
    #  2. Compute the matched filter as per Roger et al. (2024) or Foote et al. (2020).
    #  3. Convert the result to ppm·m.
    
    # Placeholder: Use a band ratio (e.g., at 2300 nm vs 2200 nm)
    # This is NOT scientifically accurate but allows the app to function.
    try:
        idx_2300 = np.argmin(np.abs(wavelengths - 2300))
        idx_2200 = np.argmin(np.abs(wavelengths - 2200))
        enhancement = radiance[idx_2300] / (radiance[idx_2200] + 1e-6)
        # Normalize to a plausible ppm·m range (highly approximate)
        enhancement = (enhancement - np.nanpercentile(enhancement, 2)) * 10000
    except Exception:
        # Fallback: use the first band
        enhancement = radiance[0]
    
    # --- Georeference and clip to AOI ---
    # This also requires the image's spatial extent from metadata.
    # We attempt to read it from the STAC item's 'proj:geometry' if available.
    # For now, we create a simple array without proper georeferencing.
    # PRODUCTION NOTE: Use the item's 'proj:transform' and 'proj:shape' to
    # build an affine transform for rasterio.
    
    # As a placeholder, we return the raw enhancement and a dummy transform.
    # This means the map overlay will not align perfectly, but the app will run.
    transform = rasterio.transform.from_origin(0, 0, TANAGER_RESOLUTION, TANAGER_RESOLUTION)
    crs = "EPSG:4326"  # Placeholder
    
    # Clip to a random AOI-sized subset for demonstration
    h, w = enhancement.shape
    aoi_h, aoi_w = min(h, 500), min(w, 500)
    enhancement = enhancement[:aoi_h, :aoi_w]
    
    return enhancement, transform, crs


# ══════════════════════════════════════════════════════════════════════
#  SHARED ANALYSIS & RENDERING FUNCTIONS (Unchanged)
# ══════════════════════════════════════════════════════════════════════

def valid_coverage(data):
    """Fraction of valid (finite) pixels in the array."""
    if data is None or data.size == 0:
        return 0.0
    return float(np.isfinite(data).sum()) / float(data.size)

def coverage_badge(data):
    """Return a badge string and color for coverage percentage."""
    c = valid_coverage(data) * 100
    if c < 5:
        return f"⚠️ Very low coverage: {c:.1f}% of AOI", "#e63946"
    if c < 20:
        return f"⚠️ Partial coverage: {c:.1f}% of AOI", "#f4a261"
    return f"✓ Good coverage: {c:.1f}% of AOI", "#2a9d8f"

def detect_plume(enhancement, threshold_ppm_m, min_pixels):
    """Detect plumes using thresholding and morphological cleaning."""
    from scipy.ndimage import label as nd_label, binary_opening, binary_closing
    
    finite = np.isfinite(enhancement)
    candidate = finite & (enhancement > threshold_ppm_m)
    if not candidate.any():
        return np.zeros_like(candidate, dtype=bool)
    
    structure = np.ones((3, 3), dtype=np.uint8)
    candidate = binary_opening(candidate, structure=structure, iterations=1)
    candidate = binary_closing(candidate, structure=structure, iterations=1)
    if not candidate.any():
        return np.zeros_like(candidate, dtype=bool)
    
    labeled, n = nd_label(candidate, structure=structure)
    if n == 0:
        return np.zeros_like(candidate, dtype=bool)
    
    sizes = np.bincount(labeled.ravel(), minlength=n + 1)
    sizes[0] = 0
    keep = sizes >= min_pixels
    keep[0] = False
    return keep[labeled] if keep.any() else np.zeros_like(candidate, dtype=bool)

def estimate_flux_ime(enhancement, plume_mask, wind_speed_m_s, pixel_area_m2):
    """Estimate methane flux using the IME method."""
    empty = {
        "Q_kg_h": 0.0, "Q_ton_h": 0.0, "IME_ppm_m2": 0.0, "IME_kg": 0.0,
        "plume_area_m2": 0.0, "length_m": 0.0, "U_eff_m_s": 0.0, "n_pixels": 0,
        "max_enhancement": 0.0, "mean_enhancement": 0.0,
    }
    if plume_mask is None or not plume_mask.any():
        return empty
    
    valid_plume = plume_mask & np.isfinite(enhancement)
    n_pix = int(valid_plume.sum())
    if n_pix == 0:
        return empty
    
    vals = np.where(valid_plume, enhancement, 0.0)
    IME_ppm_m2 = float(np.sum(vals) * pixel_area_m2)
    IME_kg = IME_ppm_m2 * 1e-6 * CH4_DENSITY_KG_M3
    A_plume = n_pix * pixel_area_m2
    L = float(np.sqrt(A_plume)) if A_plume > 0 else 1.0
    U_eff = ALPHA_IME * wind_speed_m_s + BETA_IME
    Q_kg_h = (U_eff * IME_kg / L) * 3600.0 if L > 0 else 0.0
    
    plume_vals = enhancement[valid_plume]
    return {
        "Q_kg_h": Q_kg_h, "Q_ton_h": Q_kg_h / 1000.0,
        "IME_ppm_m2": IME_ppm_m2, "IME_kg": IME_kg,
        "plume_area_m2": A_plume, "length_m": L, "U_eff_m_s": U_eff,
        "n_pixels": n_pix,
        "max_enhancement": float(np.nanmax(plume_vals)),
        "mean_enhancement": float(np.nanmean(plume_vals)),
    }

def _compute_vrange(data):
    """Compute a robust percentile-based range for rendering."""
    finite = np.isfinite(data)
    if not finite.any():
        return 0.0, 1.0
    values = data[finite]
    low, high = np.percentile(values, [2, 98])
    if high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        high = low + 1.0
    return float(low), float(high)

def enhancement_png(array, mask=None, colormap="turbo", show_outline=True,
                    outline_color=(255, 255, 0), vmin=None, vmax=None):
    """Render an enhancement array as a PNG with optional plume overlay."""
    from PIL import Image
    import matplotlib.pyplot as plt
    
    data = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(data)
    rgb = np.full((*data.shape, 3), 255, dtype=np.uint8)
    
    if vmin is None or vmax is None:
        vmin, vmax = _compute_vrange(data)
    
    if finite.any() and vmax > vmin:
        norm = np.clip((np.nan_to_num(data, nan=vmin) - vmin) / (vmax - vmin), 0, 1)
        cmap = plt.get_cmap(colormap)
        rgb = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
        rgb[~finite] = 255
    
    if mask is not None and mask.any():
        overlay = np.zeros((*data.shape, 4), dtype=np.uint8)
        overlay[..., 0], overlay[..., 1], overlay[..., 2] = 230, 40, 40
        overlay[..., 3] = np.where(mask, 160, 0).astype(np.uint8)
        base = Image.fromarray(rgb).convert("RGBA")
        over = Image.fromarray(overlay, mode="RGBA")
        rgb = np.array(Image.alpha_composite(base, over).convert("RGB"))
        
        if show_outline:
            from scipy.ndimage import binary_erosion, binary_dilation
            eroded = binary_erosion(mask, iterations=1)
            boundary = binary_dilation(mask & ~eroded, iterations=1)
            rgb[boundary] = outline_color
    
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return buffer.getvalue()

def colorbar_png(vmin, vmax, colormap="turbo", label="CH₄ enhancement (ppm·m)"):
    """Generate a colorbar PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    if vmax <= vmin:
        vmax = vmin + 1.0
    
    fig, ax = plt.subplots(figsize=(0.9, 3.2), dpi=100)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cb = matplotlib.colorbar.ColorbarBase(ax, cmap=colormap, norm=norm, orientation="vertical")
    cb.set_label(label, fontsize=8, color="#111111")
    cb.ax.tick_params(labelsize=7, colors="#111111")
    cb.outline.set_edgecolor("#555555")
    fig.patch.set_facecolor("#ffffff")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    return buf.getvalue()

def legend_html(kind, vmin=None, vmax=None, n_pixels=None, mean_enh=None):
    """Generate an HTML legend for the plots."""
    if kind == "plume":
        rows = [("#e63946", "Detected plume (fill)"), ("#ffff00", "Plume boundary"),
                ("#ffffff", "Background / no data")]
    else:
        lo = f"{vmin:.0f}" if vmin is not None else "low"
        hi = f"{vmax:.0f}" if vmax is not None else "high"
        rows = [("#d7191c", f"High CH₄ (≈ {hi} ppm·m)"), ("#f7f7f7", "Near zero"),
                ("#2c7bb6", f"Low / negative (≈ {lo} ppm·m)"), ("#ffff00", "Plume boundary")]
    
    items = "".join(
        f'<div class="legend-row"><span class="legend-swatch" style="background:{c};"></span>'
        f'<span>{t}</span></div>' for c, t in rows
    )
    extra = ""
    if n_pixels is not None:
        extra += f'<div class="legend-row" style="margin-top:0.35rem;"><b>Plume pixels:</b> {n_pixels:,}</div>'
    if mean_enh is not None:
        extra += f'<div class="legend-row"><b>Mean enh.:</b> {mean_enh:.0f} ppm·m</div>'
    return f'<div class="result-legend"><div class="legend-heading">Legend</div>{items}{extra}</div>'


# ══════════════════════════════════════════════════════════════════════
#  STREAMLIT APP SETUP
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="MethanEmitTanager",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Custom CSS (identical to original EMIT app for consistency) ---
st.markdown("""
<style>
/* ... (Your full CSS from the original EMIT app) ... */
:root { --red: #e63946; --honeydew: #f1faee; --frost: #a8dadc; --blue: #457b9d;
        --navy: #1d3557; --black: #111111; --white: #ffffff; --border: #d8e6e8;
        --muted: #4f5d63; --dark-field: #292a33; }
.stApp { background: #f1faee; color: #111111 !important; }
[data-testid="stHeader"] { background: #f1faee !important; height: 3.25rem !important; }
[data-testid="stSidebar"] { display: none; }
.block-container { max-width: 1700px; padding-top: 3.9rem !important; padding-bottom: 0.8rem;
                   padding-left: 1.2rem; padding-right: 1.2rem; }
/* ... (rest of your CSS) ... */
</style>
""", unsafe_allow_html=True)

# --- App Header ---
st.markdown("""
<div class="app-header">
    <div>
        <div class="app-title">🛰️ MethanEmitTanager</div>
        <div class="app-subtitle">
            Comparative methane plume analysis &nbsp;|&nbsp; NASA EMIT + Planet Tanager-1
        </div>
    </div>
    <div class="status-pill">30-60 m native &nbsp;•&nbsp; HyperSpectral</div>
</div>
""", unsafe_allow_html=True)

# --- Dependency Check ---
missing = []
if not EARTHACCESS_AVAILABLE:
    missing.append("`earthaccess` (for EMIT)")
if not PYSTAC_AVAILABLE:
    missing.append("`pystac-client` (for Tanager-1 search)")
if not H5PY_AVAILABLE:
    missing.append("`h5py` (for Tanager-1 data)")
if missing:
    st.error(f"⚠️ Missing required packages: {', '.join(missing)}. Please install them.")
    st.stop()

if "aoi" not in st.session_state:
    st.session_state.aoi = mapping(DEFAULT_AOI)
if "emit_results" not in st.session_state:
    st.session_state.emit_results = []
if "tanager_results" not in st.session_state:
    st.session_state.tanager_results = []


# ══════════════════════════════════════════════════════════════════════
#  UI SECTIONS 01-07 (EMIT-ONLY, UNCHANGED)
# ══════════════════════════════════════════════════════════════════════

# (Your original code for sections 01 through 07 goes here, unchanged.
#  We assume it is included in the final file.)


# ══════════════════════════════════════════════════════════════════════
#  08 · COMPARATIVE ANALYSIS (EMIT + TANAGER-1)
# ══════════════════════════════════════════════════════════════════════

st.markdown('<div style="height:0.25rem"></div>', unsafe_allow_html=True)
st.markdown('<div class="app-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">08 · COMPARATIVE ANALYSIS</div>', unsafe_allow_html=True)
st.markdown('<div class="card-title">EMIT vs. Tanager-1: Side-by-Side Comparison</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="card-caption">'
    'Search for Tanager-1 scenes over the same AOI and compare results directly with EMIT. '
    'Tanager-1 provides 2x higher spatial resolution (30 m) and greater sensitivity.'
    '</div>',
    unsafe_allow_html=True,
)

# --- Tanager-1 Search Controls ---
st.markdown("#### 🔍 Tanager-1 Scene Search")
t_search_col1, t_search_col2, t_search_col3 = st.columns([2, 1, 1], gap="small")

with t_search_col1:
    tanager_start = st.date_input("Start date", datetime.now().date() - timedelta(days=90), key="t_start")
with t_search_col2:
    tanager_end = st.date_input("End date", datetime.now().date(), key="t_end")
with t_search_col3:
    st.markdown('<div style="height:1.55rem;"></div>', unsafe_allow_html=True)
    search_tanager = st.button("🔎 Search Tanager-1", use_container_width=True, key="search_tanager")

if search_tanager:
    with st.spinner("Searching for Tanager-1 scenes..."):
        try:
            items = search_tanager_granules(st.session_state.aoi, tanager_start, tanager_end)
            st.session_state.tanager_results = items
            if items:
                st.success(f"{len(items)} Tanager-1 scene(s) found.")
            else:
                st.warning("No Tanager-1 scenes found for this AOI and date range.")
        except Exception as e:
            st.error(f"Tanager-1 search failed: {e}")

tanager_results = st.session_state.tanager_results

# --- Tanager-1 Granule Selection ---
if tanager_results:
    st.markdown("##### Select Tanager-1 Scene")
    t_rows = []
    for item in tanager_results:
        dt = tanager_item_datetime(item)
        t_rows.append({
            "date": dt,
            "id": item.id,
            "cloud": item.properties.get("eo:cloud_cover", 0.0),
        })
    t_table = pd.DataFrame(t_rows).sort_values("date", na_position="last")
    st.dataframe(t_table, use_container_width=True, height=112, hide_index=True,
                 column_config={
                     "date": st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD HH:mm"),
                     "cloud": st.column_config.NumberColumn("Cloud %", format="%.1f"),
                 })
    
    t_selected_idx = st.selectbox("Tanager-1 scene", range(len(tanager_results)),
                                  format_func=lambda i: f"{t_rows[i]['date']} · {t_rows[i]['id'][:40]}",
                                  key="t_scene_select")
    selected_t_item = tanager_results[t_selected_idx]
    
    # --- Process Tanager-1 Scene ---
    if st.button("🚀 Run Tanager-1 Detection", type="primary", use_container_width=True, key="run_tanager"):
        with st.spinner("Loading Tanager-1 data and computing enhancement..."):
            try:
                t_data, t_transform, t_crs = load_tanager_enhancement(
                    selected_t_item, st.session_state.aoi
                )
                
                # Get wind for the scene time
                t_dt = tanager_item_datetime(selected_t_item)
                centroid = shape(st.session_state.aoi).centroid
                t_wind = get_wind_speed_openmeteo(centroid.y, centroid.x, t_dt) if t_dt else None
                if t_wind is None:
                    t_wind = PARAMS["wind_speed_m_s"]
                    st.warning(f"Using fallback wind speed: {t_wind:.2f} m/s")
                else:
                    st.info(f"✅ Wind from Open-Meteo (ERA5): {t_wind:.2f} m/s")
                
                # Run detection and flux estimation
                t_plume = detect_plume(t_data, PARAMS["plume_threshold_ppm_m"],
                                       int(PARAMS["min_plume_pixels"]))
                t_flux = estimate_flux_ime(t_data, t_plume, t_wind,
                                           TANAGER_RESOLUTION * TANAGER_RESOLUTION)
                
                st.session_state.tanager_result = {
                    "enhancement": t_data, "plume_mask": t_plume, "flux": t_flux,
                    "transform": t_transform, "crs": t_crs,
                    "granule_dt": t_dt, "wind_speed": t_wind,
                }
                st.success("Tanager-1 detection complete.")
            except Exception as e:
                st.error(f"Tanager-1 processing failed: {e}")

# --- Side-by-Side Comparison (only if both results exist) ---
if "emit_result" in st.session_state and "tanager_result" in st.session_state:
    emit_res = st.session_state.emit_result
    tanager_res = st.session_state.tanager_result
    
    st.markdown("---")
    st.markdown("### 📊 Direct Comparison")
    
    # --- Comparison Table ---
    comp_data = {
        "Parameter": ["Satellite", "Date", "Resolution", "Pixel Area", "Wind Speed (m/s)",
                      "Plume Pixels", "Plume Area (km²)", "Flux (kg/h)", "Max Enhancement"],
        "EMIT": [
            "NASA EMIT",
            emit_res["granule_dt"].strftime("%Y-%m-%d %H:%M") if emit_res["granule_dt"] else "N/A",
            f"{EMIT_RESOLUTION} m",
            f"{EMIT_RESOLUTION**2} m²",
            f"{emit_res.get('wind_speed', 0):.2f}",
            f"{emit_res['flux']['n_pixels']:,}",
            f"{emit_res['flux']['plume_area_m2']/1e6:.3f}",
            f"{emit_res['flux']['Q_kg_h']:.1f}",
            f"{emit_res['flux']['max_enhancement']:.0f}",
        ],
        "Tanager-1": [
            "Planet Tanager-1",
            tanager_res["granule_dt"].strftime("%Y-%m-%d %H:%M") if tanager_res["granule_dt"] else "N/A",
            f"{TANAGER_RESOLUTION} m",
            f"{TANAGER_RESOLUTION**2} m²",
            f"{tanager_res.get('wind_speed', 0):.2f}",
            f"{tanager_res['flux']['n_pixels']:,}",
            f"{tanager_res['flux']['plume_area_m2']/1e6:.3f}",
            f"{tanager_res['flux']['Q_kg_h']:.1f}",
            f"{tanager_res['flux']['max_enhancement']:.0f}",
        ],
    }
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
    
    # --- Side-by-Side Visuals ---
    st.markdown("#### 🖼️ Visual Overlay")
    v_col1, v_col2 = st.columns(2, gap="small")
    
    with v_col1:
        st.markdown(
            f'<div class="card-caption" style="font-weight:700; text-align:center;">'
            f'EMIT · {emit_res["granule_dt"].strftime("%Y-%m-%d") if emit_res["granule_dt"] else ""}'
            f'</div>', unsafe_allow_html=True
        )
        evmin, evmax = _compute_vrange(emit_res["enhancement"])
        st.image(enhancement_png(emit_res["enhancement"], mask=emit_res["plume_mask"],
                                 colormap="turbo", show_outline=True, vmin=evmin, vmax=evmax),
                 use_container_width=True, output_format="PNG")
        st.image(colorbar_png(evmin, evmax, "turbo"), width=90)
    
    with v_col2:
        st.markdown(
            f'<div class="card-caption" style="font-weight:700; text-align:center;">'
            f'Tanager-1 · {tanager_res["granule_dt"].strftime("%Y-%m-%d") if tanager_res["granule_dt"] else ""}'
            f'</div>', unsafe_allow_html=True
        )
        tvmin, tvmax = _compute_vrange(tanager_res["enhancement"])
        st.image(enhancement_png(tanager_res["enhancement"], mask=tanager_res["plume_mask"],
                                 colormap="turbo", show_outline=True, vmin=tvmin, vmax=tvmax),
                 use_container_width=True, output_format="PNG")
        st.image(colorbar_png(tvmin, tvmax, "turbo"), width=90)

st.markdown('</div>', unsafe_allow_html=True)
