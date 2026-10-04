"""EMIT + Tanager-1 Methane Plume Detection App.

Carbon Mapper-style methane detection on NASA EMIT and Planet Tanager-1
hyperspectral data with side-by-side comparison mode.
UI/design preserved from the original Sentinel-2/EMIT app.
"""
from __future__ import annotations

import io
import os
import json
import math
import time
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
from shapely.geometry import box, mapping, shape, Point
from shapely.ops import unary_union
from streamlit_folium import st_folium

try:
    import earthaccess
    EARTHACCESS_AVAILABLE = True
except ImportError:
    EARTHACCESS_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════

# ── Satellite definitions ──
SATELLITES = {
    "EMIT": {
        "label": "EMIT (NASA)",
        "resolution": 60,          # native pixel size (m)
        "collection": "EMITL2BCH4ENH",
        "source": "earthaccess",
        "color": "#457b9d",        # blue (existing theme)
        "icon": "🛰️",
    },
    "Tanager-1": {
        "label": "Tanager-1 (Planet/Carbon Mapper)",
        "resolution": 30,          # native pixel size (m)
        "collection": "l2b-ch4-mfa-v3a",
        "source": "carbonmapper",
        "color": "#e63946",        # red (existing theme)
        "icon": "📡",
    },
}

# ── Carbon Mapper API ──
CM_API_BASE = "https://api.carbonmapper.org/api/v1"
CM_PLUME_ENDPOINT = f"{CM_API_BASE}/catalog/plumes/annotated"
CM_STAC_BASE = f"{CM_API_BASE}/stac"

DEFAULT_AOI = box(51.20, 35.40, 51.45, 35.60)

EMIT_ENH_COLLECTION = "EMITL2BCH4ENH"   # Methane Enhancement (ppm·m)
EMIT_PLM_COLLECTION = "EMITL2BCH4PLM"   # Plume Complexes

PARAMS = {
    "plume_threshold_ppm_m": 1000.0,
    "min_plume_pixels": 10,
    "wind_speed_m_s": 2.0,  # Default fallback value
    "max_plume_area_km2": 100.0,
}

PPB_TO_KG_M2 = 5.72e-6
ALPHA_IME = 0.33
BETA_IME = 0.45
CH4_DENSITY_KG_M3 = 0.717

# Carbon Mapper plume platform prefixes
CM_PLATFORM_MAP = {
    "tan": "Tanager-1",
    "emi": "EMIT",
    "ang": "AVIRIS-NG",
    "av3": "AVIRIS-3",
    "gao": "GAO",
}


# ══════════════════════════════════════════════════════════════════════
#  GEOMETRY HELPERS
# ══════════════════════════════════════════════════════════════════════

def normalize_geometry(obj):
    if obj is None:
        return None
    if hasattr(obj, "__geo_interface__"):
        obj = obj.__geo_interface__
    if not isinstance(obj, dict):
        return None
    if obj.get("type") == "Feature":
        return normalize_geometry(obj.get("geometry"))
    if obj.get("type") == "FeatureCollection":
        geoms = []
        for feature in obj.get("features", []):
            g = normalize_geometry(feature.get("geometry"))
            if g:
                geoms.append(shape(g))
        return mapping(unary_union(geoms)) if geoms else None
    try:
        g = shape(obj)
        return mapping(g) if not g.is_empty else None
    except Exception:
        return None


def ensure_aoi(obj):
    return normalize_geometry(obj) or mapping(DEFAULT_AOI)


def aoi_bounds(aoi):
    return shape(ensure_aoi(aoi)).bounds


def compute_zoom(bounds):
    try:
        minx, miny, maxx, maxy = bounds
        span = max(maxx - minx, maxy - miny, 1e-6)
        zoom = int(round(math.log2(360.0 / span))) - 1
        return max(3, min(15, zoom))
    except Exception:
        return 11


def create_map(aoi, extra_layers=None):
    """Build Folium map. ``extra_layers`` can hold GeoJSON overlays."""
    geometry = shape(ensure_aoi(aoi))
    centroid = geometry.centroid
    zoom = compute_zoom(geometry.bounds)
    fmap = folium.Map(
        [centroid.y, centroid.x],
        zoom_start=zoom,
        tiles="OpenStreetMap",
    )
    folium.GeoJson(
        mapping(geometry),
        style_function=lambda _: {"color": "blue", "fill": False, "weight": 2},
        name="AOI",
    ).add_to(fmap)
    if extra_layers:
        for name, geojson, color in extra_layers:
            folium.GeoJson(
                geojson,
                name=name,
                style_function=lambda _, c=color: {
                    "color": c, "fill": True, "fillOpacity": 0.25, "weight": 2
                },
                tooltip=name,
            ).add_to(fmap)
        folium.LayerControl().add_to(fmap)
    Draw(
        export=True,
        draw_options={
            "polyline": False,
            "circle": False,
            "marker": False,
            "circlemarker": False,
            "polygon": {
                "allowIntersection": False,
                "showArea": True,
            },
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(fmap)
    MousePosition(
        position="bottomright",
        separator=" | ",
        prefix="📍 Lat, Lon:",
        lat_formatter="function(num) {return num.toFixed(5);}",
        lng_formatter="function(num) {return num.toFixed(5);}",
    ).add_to(fmap)
    return fmap


# ══════════════════════════════════════════════════════════════════════
#  GEOCODING
# ══════════════════════════════════════════════════════════════════════

def geocode_place(query: str):
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": query,
            "format": "json",
            "limit": 1,
            "polygon_geojson": 1,
        }
        headers = {"User-Agent": "EMIT-Tanager-Methane-App/1.0 (streamlit)"}
        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        if not data:
            return None, None, None
        item = data[0]
        lat = float(item["lat"])
        lon = float(item["lon"])
        label = item.get("display_name", query)

        gj = item.get("geojson")
        if gj and gj.get("type") in ("Polygon", "MultiPolygon"):
            try:
                geom = shape(gj)
                if not geom.is_empty:
                    return geom, (lat, lon), label
            except Exception:
                pass

        bb = item.get("boundingbox")
        if bb:
            south, north, west, east = [float(x) for x in bb]
            return box(west, south, east, north), (lat, lon), label

        d = 0.02
        return box(lon - d, lat - d, lon + d, lat + d), (lat, lon), label
    except Exception:
        return None, None, None


# ══════════════════════════════════════════════════════════════════════
#  OPEN-METEO WIND
# ══════════════════════════════════════════════════════════════════════

def get_wind_speed_openmeteo(lat: float, lon: float, dt: datetime) -> Optional[float]:
    """Fetch 10m wind speed (m/s) from Open-Meteo archive for a given point & time.

    Uses ERA5 reanalysis (free, no API key). Returns the wind speed
    at the closest hour to ``dt``, or ``None`` on failure.
    """
    try:
        url = "https://archive-api.open-meteo.com/v1/archive"
        date_str = dt.strftime("%Y-%m-%d")
        params = {
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "start_date": date_str,
            "end_date": date_str,
            "hourly": "wind_speed_10m",
            "windspeed_unit": "ms",
            "timezone": "UTC",
        }
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        speeds = hourly.get("wind_speed_10m", [])
        if not times or not speeds:
            return None

        target = dt.strftime("%Y-%m-%dT%H:00")
        if target in times:
            idx = times.index(target)
        else:
            best_idx = 0
            best_diff = None
            for i, t in enumerate(times):
                try:
                    t_dt = datetime.fromisoformat(t)
                    diff = abs((t_dt - dt).total_seconds())
                except Exception:
                    continue
                if best_diff is None or diff < best_diff:
                    best_diff = diff
                    best_idx = i
            idx = best_idx

        val = speeds[idx]
        if val is None:
            return None
        return float(val)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════
#  EARTHDATA AUTH
# ══════════════════════════════════════════════════════════════════════

def login_earthdata():
    if not EARTHACCESS_AVAILABLE:
        raise RuntimeError(
            "Package 'earthaccess' is not installed. "
            "Please check requirements.txt."
        )
    try:
        username = st.secrets["EARTHDATA_USERNAME"]
        password = st.secrets["EARTHDATA_PASSWORD"]
    except (KeyError, FileNotFoundError):
        raise RuntimeError(
            "Earthdata credentials are not configured. "
            "Add EARTHDATA_USERNAME and EARTHDATA_PASSWORD to Streamlit secrets."
        )

    os.environ["EARTHDATA_USERNAME"] = username
    os.environ["EARTHDATA_PASSWORD"] = password

    try:
        auth = earthaccess.login(strategy="environment")
    except Exception as e:
        raise RuntimeError(f"Earthdata login failed: {e}")

    if not auth.authenticated:
        raise RuntimeError(
            "Earthdata did not accept the credentials. "
            "Check your username/password or register at urs.earthdata.nasa.gov."
        )
    return auth


# ══════════════════════════════════════════════════════════════════════
#  CARBON MAPPER AUTH  (Tanager-1 / EMIT plume catalog)
# ══════════════════════════════════════════════════════════════════════

def get_carbonmapper_token() -> str:
    """Retrieve Carbon Mapper Bearer token from Streamlit secrets."""
    try:
        token = st.secrets["CARBONMAPPER_TOKEN"]
    except (KeyError, FileNotFoundError):
        raise RuntimeError(
            "Carbon Mapper credentials are not configured. "
            "Add CARBONMAPPER_TOKEN to Streamlit secrets. "
            "Register free at https://api.carbonmapper.org"
        )
    return token


def _cm_headers() -> dict:
    return {
        "Authorization": f"Bearer {get_carbonmapper_token()}",
        "Accept": "application/json",
    }


def _cm_bbox_rest(bounds):
    """Carbon Mapper REST catalog expects repeated bbox keys."""
    minx, miny, maxx, maxy = bounds
    return {"bbox": [minx, miny, maxx, maxy]}


def _cm_bbox_stac(bounds):
    """Carbon Mapper STAC expects comma-joined bbox."""
    minx, miny, maxx, maxy = bounds
    return f"{minx},{miny},{maxx},{maxy}"


# ══════════════════════════════════════════════════════════════════════
#  EMIT SEARCH & LOADING
# ══════════════════════════════════════════════════════════════════════

def search_emit_granules(aoi, start_date, end_date):
    minx, miny, maxx, maxy = aoi_bounds(aoi)
    results = earthaccess.search_data(
        short_name=EMIT_ENH_COLLECTION,
        bounding_box=(minx, miny, maxx, maxy),
        temporal=(start_date.strftime("%Y-%m-%d"),
                  end_date.strftime("%Y-%m-%d")),
        count=200,
    )
    return list(results)


def granule_datetime(granule) -> Optional[datetime]:
    try:
        umm = granule.get("umm", {}) if hasattr(granule, "get") else {}
    except Exception:
        umm = {}
    temporal = umm.get("TemporalExtent", {}).get("RangeDateTime", {})
    dt_str = temporal.get("BeginningDateTime")
    if dt_str:
        try:
            return datetime.fromisoformat(dt_str.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            pass
    try:
        gid = granule.get("meta", {}).get("native-id", "")
        for part in gid.split("_"):
            if len(part) >= 15 and part[:8].isdigit():
                return datetime.strptime(part[:15], "%Y%m%dT%H%M%S")
    except Exception:
        pass
    return None


def granule_cloud(granule) -> float:
    try:
        umm = granule.get("umm", {})
        for attr in umm.get("AdditionalAttributes", []):
            if attr.get("Name") == "CloudCover":
                vals = attr.get("Values", [])
                if vals:
                    return float(vals[0])
    except Exception:
        pass
    return 0.0


def load_emit_enhancement(granule, aoi, resolution=60):
    """Load EMIT enhancement clipped to AOI, with nodata masked to NaN."""
    try:
        files = earthaccess.open([granule])
    except Exception as e:
        raise RuntimeError(f"Failed to open granule stream: {e}")

    if not files:
        raise RuntimeError("No files returned by earthaccess.open().")

    tif_path = None
    for f in files:
        name = getattr(f, "path", str(f))
        if name.lower().endswith((".tif", ".tiff")):
            tif_path = f
            break
    if tif_path is None:
        tif_path = files[0]

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
            transform = src.transform
            crs = src.crs

    data = data.astype(np.float32)

    if nodata is not None:
        try:
            nd = float(nodata)
            data = np.where(np.isclose(data, nd, rtol=0, atol=1e-3),
                            np.nan, data)
        except Exception:
            pass

    for fv in (-9999.0, -999.0, -99999.0):
        data = np.where(np.isclose(data, fv, rtol=0, atol=1e-3),
                        np.nan, data)

    data = np.where(np.abs(data) > 1e6, np.nan, data)

    try:
        from rasterio.features import geometry_mask
        geom_mask = geometry_mask(
            [shape(ensure_aoi(aoi))],
            out_shape=data.shape,
            transform=transform,
            invert=True,
        )
        data = np.where(geom_mask, data, np.nan)
    except Exception:
        pass

    return data, transform, crs


# ══════════════════════════════════════════════════════════════════════
#  TANAGER-1  (Carbon Mapper API)
# ══════════════════════════════════════════════════════════════════════

def parse_cm_plume_datetime(plume_id: str) -> Optional[datetime]:
    """Parse datetime from Carbon Mapper plume ID.

    Format: ``tan20251212t185057c20s4001-E``
    Prefix ``tan`` / ``emi`` / ``ang`` / ``av3`` / ``gao`` then ``YYYYMMDDThhmmss``.
    """
    try:
        for prefix in CM_PLATFORM_MAP:
            if plume_id.startswith(prefix):
                rest = plume_id[len(prefix):]
                ts = rest[:15]  # YYYYMMDDThhmmss
                return datetime.strptime(ts, "%Y%m%dT%H%M%S")
    except Exception:
        pass
    return None


def cm_plume_platform(plume_id: str) -> str:
    for prefix, name in CM_PLATFORM_MAP.items():
        if plume_id.startswith(prefix):
            return name
    return "unknown"


def search_carbonmapper_plumes(aoi, start_date, end_date,
                               gas="CH4", instrument="tan"):
    """Search Carbon Mapper plume catalog (Tanager-1 by default).

    Correct API contract (from https://api.carbonmapper.org/api/v1/docs):
      - bbox      : repeated keys  -> ?bbox=W&bbox=S&bbox=E&bbox=N
      - datetime  : "START/END" ISO-8601 interval (not start_time/end_time)
      - plume_gas : "CH4" (not "gas")
      - instrument: "tan" / "emi" / "ang" / "av3" / "GAO" (case-sensitive)
    """
    minx, miny, maxx, maxy = aoi_bounds(aoi)

    dt_start = start_date.strftime("%Y-%m-%dT00:00:00.000Z")
    dt_end   = end_date.strftime("%Y-%m-%dT23:59:59.999Z")
    datetime_range = f"{dt_start}/{dt_end}"

    all_features = []
    offset = 0
    limit = 100
    max_pages = 20

    for _ in range(max_pages):
        # ✅ Use list of tuples to send repeated bbox keys
        params = [
            ("bbox", minx),
            ("bbox", miny),
            ("bbox", maxx),
            ("bbox", maxy),
            ("datetime", datetime_range),
            ("plume_gas", gas),
            ("instrument", instrument),
            ("limit", limit),
            ("offset", offset),
        ]
        try:
            r = requests.get(
                CM_PLUME_ENDPOINT,
                params=params,
                headers=_cm_headers(),
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            raise RuntimeError(f"Carbon Mapper plume search failed: {e}")

        features = data.get("features", [])
        if not features:
            break
        all_features.extend(features)

        if len(features) < limit:
            break
        offset += limit

    return all_features
                                   

def tanager_plumes_only(features):
    """Keep only Tanager-1 plumes (safety filter)."""
    out = []
    for f in features:
        pid = f.get("properties", {}).get("plume_id", "")
        if pid.startswith("tan"):
            out.append(f)
    return out

def cm_plume_datetime(feature) -> Optional[datetime]:
    """Extract datetime from a Carbon Mapper plume feature."""
    props = feature.get("properties", {})
    pid = props.get("plume_id", "")
    dt = parse_cm_plume_datetime(pid)
    if dt:
        return dt
    # fallback: try datetime field
    dt_str = props.get("datetime") or props.get("acquisition_date")
    if dt_str:
        try:
            return datetime.fromisoformat(dt_str.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            pass
    return None


def cm_plume_emission(feature) -> float:
    """Reported emission rate (kg/h) from Carbon Mapper."""
    props = feature.get("properties", {})
    for key in ("emission_auto", "emission_rate_kg_hr", "emission_rate"):
        if key in props and props[key] is not None:
            try:
                return float(props[key])
            except Exception:
                pass
    return 0.0


def cm_plume_wind(feature) -> Optional[float]:
    """Wind speed (m/s) from Carbon Mapper plume properties."""
    props = feature.get("properties", {})
    for key in ("wind_speed", "wind_speed_m_s"):
        if key in props and props[key] is not None:
            try:
                return float(props[key])
            except Exception:
                pass
    return None


def find_overlap_dates(emit_results, tanager_results):
    """Find calendar dates that have data from BOTH satellites."""
    emit_dates = set()
    for g in emit_results:
        dt = granule_datetime(g)
        if dt:
            emit_dates.add(dt.date())

    tan_dates = set()
    for f in tanager_results:
        dt = cm_plume_datetime(f)
        if dt:
            tan_dates.add(dt.date())

    return sorted(emit_dates & tan_dates)


def tanager_scene_id(plume_id: str) -> str:
    """Scene ID from plume ID: ``tan20251212t185057c20s4001-E`` → ``tan20251212t185057c20s4001``."""
    return plume_id.rsplit("-", 1)[0]


def load_tanager_enhancement(plume_feature, aoi):
    """Load Tanager-1 enhancement raster from Carbon Mapper STAC.

    Returns ``(data, transform, crs)`` or ``(None, None, None)`` if the
    L2B scene is not yet published (publication lag: weeks to months).
    """
    props = plume_feature.get("properties", {})
    plume_id = props.get("plume_id", "")
    scene_id = tanager_scene_id(plume_id)

    # Try to fetch STAC item for the L2B scene
    try:
        url = f"{CM_STAC_BASE}/collections/{SATELLITES['Tanager-1']['collection']}/items/{scene_id}"
        r = requests.get(url, headers=_cm_headers(), timeout=30)
        r.raise_for_status()
        item = r.json()
    except Exception:
        return None, None, None

    # Prefer cmf.tif (concentration methane file, orthorectified)
    assets = item.get("assets", {})
    cmf_url = None
    for key in ("cmf.tif", "cmf", "concentration"):
        if key in assets:
            cmf_url = assets[key].get("href")
            break
    if cmf_url is None:
        return None, None, None

    minx, miny, maxx, maxy = aoi_bounds(aoi)

    try:
        # Carbon Mapper STAC asset URLs are token-gated; pass header via rasterio
        import rasterio
        from rasterio.session import DummySession

        # Download to memory using requests (handles auth), then open
        resp = requests.get(cmf_url, headers=_cm_headers(), timeout=60, stream=True)
        resp.raise_for_status()
        raw = io.BytesIO(resp.content)

        with rasterio.open(raw) as src:
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
                transform = src.transform
                crs = src.crs
    except Exception:
        return None, None, None

    data = data.astype(np.float32)

    if nodata is not None:
        try:
            nd = float(nodata)
            data = np.where(np.isclose(data, nd, rtol=0, atol=1e-3), np.nan, data)
        except Exception:
            pass

    for fv in (-9999.0, -999.0, -99999.0):
        data = np.where(np.isclose(data, fv, rtol=0, atol=1e-3), np.nan, data)

    data = np.where(np.abs(data) > 1e6, np.nan, data)

    try:
        from rasterio.features import geometry_mask
        geom_mask = geometry_mask(
            [shape(ensure_aoi(aoi))],
            out_shape=data.shape,
            transform=transform,
            invert=True,
        )
        data = np.where(geom_mask, data, np.nan)
    except Exception:
        pass

    return data, transform, crs


def cm_plume_geojson(features):
    """Build a FeatureCollection of plume geometries for map overlay."""
    feats = []
    for f in features:
        geom = f.get("geometry")
        if geom:
            feats.append({
                "type": "Feature",
                "geometry": geom,
                "properties": f.get("properties", {}),
            })
    return {"type": "FeatureCollection", "features": feats}


# ══════════════════════════════════════════════════════════════════════
#  UNIFIED SEARCH
# ══════════════════════════════════════════════════════════════════════

def search_all_satellites(aoi, start_date, end_date, satellites=None):
    """Search both EMIT and Tanager-1; return dict of results."""
    if satellites is None:
        satellites = ["EMIT", "Tanager-1"]

    results = {"EMIT": [], "Tanager-1": [], "errors": []}

    if "EMIT" in satellites:
        try:
            results["EMIT"] = search_emit_granules(aoi, start_date, end_date)
        except Exception as e:
            results["errors"].append(f"EMIT: {e}")

    if "Tanager-1" in satellites:
        try:
            cm_feats = search_carbonmapper_plumes(aoi, start_date, end_date)
            results["Tanager-1"] = tanager_plumes_only(cm_feats)
        except Exception as e:
            results["errors"].append(f"Tanager-1: {e}")

    return results


# ══════════════════════════════════════════════════════════════════════
#  COVERAGE HELPERS
# ══════════════════════════════════════════════════════════════════════

def valid_coverage(data):
    if data is None or data.size == 0:
        return 0.0
    return float(np.isfinite(data).sum()) / float(data.size)


def coverage_badge(data):
    c = valid_coverage(data) * 100
    if c < 5:
        return f"⚠️ Very low coverage: {c:.1f}% of AOI", "#e63946"
    if c < 20:
        return f"⚠️ Partial coverage: {c:.1f}% of AOI", "#f4a261"
    return f"✓ Good coverage: {c:.1f}% of AOI", "#2a9d8f"


# ══════════════════════════════════════════════════════════════════════
#  ALGORITHM
# ══════════════════════════════════════════════════════════════════════

def detect_plume(enhancement, threshold_ppm_m, min_pixels):
    from scipy.ndimage import (
        label as nd_label,
        binary_opening,
        binary_closing,
    )

    finite = np.isfinite(enhancement)
    candidate = finite & (enhancement > threshold_ppm_m)
    plume = np.zeros_like(candidate, dtype=bool)

    if not candidate.any():
        return plume

    structure = np.ones((3, 3), dtype=np.uint8)
    candidate = binary_opening(candidate, structure=structure, iterations=1)
    candidate = binary_closing(candidate, structure=structure, iterations=1)

    if not candidate.any():
        return plume

    labeled, n = nd_label(candidate, structure=structure)
    if n == 0:
        return plume

    sizes = np.bincount(labeled.ravel(), minlength=n + 1)
    sizes[0] = 0
    keep = sizes >= min_pixels
    keep[0] = False
    if keep.any():
        plume = keep[labeled]
    return plume


def estimate_flux_ime(enhancement, plume_mask, wind_speed_m_s, resolution=60):
    """IME-based flux estimation. ``resolution`` is the native pixel size (m)."""
    empty = {
        "Q_kg_h": 0.0, "Q_ton_h": 0.0,
        "IME_ppm_m2": 0.0, "IME_kg": 0.0,
        "plume_area_m2": 0.0, "length_m": 0.0,
        "U_eff_m_s": 0.0, "n_pixels": 0,
        "max_enhancement": 0.0, "mean_enhancement": 0.0,
    }
    if plume_mask is None or not plume_mask.any():
        return empty

    valid_plume = plume_mask & np.isfinite(enhancement)
    n_pix = int(valid_plume.sum())
    if n_pix == 0:
        return empty

    pixel_area = resolution * resolution
    vals = np.where(valid_plume, enhancement, 0.0)
    IME_ppm_m2 = float(np.sum(vals) * pixel_area)
    IME_kg = IME_ppm_m2 * 1e-6 * CH4_DENSITY_KG_M3

    A_plume = n_pix * pixel_area
    L = float(np.sqrt(A_plume)) if A_plume > 0 else 1.0
    U_eff = ALPHA_IME * wind_speed_m_s + BETA_IME

    Q_kg_s = U_eff * IME_kg / L if L > 0 else 0.0
    Q_kg_h = Q_kg_s * 3600.0

    plume_vals = enhancement[valid_plume]

    return {
        "Q_kg_h": Q_kg_h,
        "Q_ton_h": Q_kg_h / 1000.0,
        "IME_ppm_m2": IME_ppm_m2,
        "IME_kg": IME_kg,
        "plume_area_m2": A_plume,
        "length_m": L,
        "U_eff_m_s": U_eff,
        "n_pixels": n_pix,
        "max_enhancement": float(np.nanmax(plume_vals)),
        "mean_enhancement": float(np.nanmean(plume_vals)),
    }


# ══════════════════════════════════════════════════════════════════════
#  IMAGE RENDERING
# ══════════════════════════════════════════════════════════════════════

def _compute_vrange(data):
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


def enhancement_png(
    array,
    mask=None,
    colormap="turbo",
    show_outline=True,
    outline_color=(255, 255, 0),
    vmin=None,
    vmax=None,
):
    """Render enhancement with optional plume overlay + outline."""
    from PIL import Image
    import matplotlib.pyplot as plt

    data = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(data)
    rgb = np.full((*data.shape, 3), 255, dtype=np.uint8)

    if vmin is None or vmax is None:
        vmin, vmax = _compute_vrange(data)

    if finite.any() and vmax > vmin:
        norm = np.clip(
            (np.nan_to_num(data, nan=vmin) - vmin) / (vmax - vmin), 0, 1
        )
        cmap = plt.get_cmap(colormap)
        rgb = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
        rgb[~finite] = 255

    if mask is not None and mask.any():
        overlay = np.zeros((*data.shape, 4), dtype=np.uint8)
        overlay[..., 0] = 230
        overlay[..., 1] = 40
        overlay[..., 2] = 40
        overlay[..., 3] = np.where(mask, 160, 0).astype(np.uint8)
        base = Image.fromarray(rgb).convert("RGBA")
        over = Image.fromarray(overlay, mode="RGBA")
        rgb = np.array(Image.alpha_composite(base, over).convert("RGB"))

        if show_outline:
            try:
                from scipy.ndimage import binary_erosion, binary_dilation
                eroded = binary_erosion(mask, iterations=1)
                boundary = mask & ~eroded
                boundary = binary_dilation(boundary, iterations=1)
                rgb[boundary] = outline_color
            except Exception:
                pass

    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return buffer.getvalue()


def placeholder_png(text="No raster available"):
    """Render a simple placeholder PNG."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (600, 300), "#f8fbfb")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    d.text((300, 140), text, fill="#4f5d63", anchor="mm", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def colorbar_png(vmin, vmax, colormap="turbo", label="CH₄ enhancement (ppm·m)"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if vmax <= vmin:
        vmax = vmin + 1.0

    fig, ax = plt.subplots(figsize=(0.9, 3.2), dpi=100)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cb = matplotlib.colorbar.ColorbarBase(
        ax, cmap=colormap, norm=norm, orientation="vertical"
    )
    cb.set_label(label, fontsize=8, color="#111111")
    cb.ax.tick_params(labelsize=7, colors="#111111")
    cb.outline.set_edgecolor("#555555")
    fig.patch.set_facecolor("#ffffff")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    return buf.getvalue()


def legend_html(kind, vmin=None, vmax=None, n_pixels=None, mean_enh=None, satellite=None):
    if kind == "plume":
        rows = [
            ("#e63946", "Detected plume (fill)"),
            ("#ffff00", "Plume boundary"),
            ("#ffffff", "Background / no data"),
        ]
    elif kind == "enhancement":
        lo = f"{vmin:.0f}" if vmin is not None else "low"
        hi = f"{vmax:.0f}" if vmax is not None else "high"
        rows = [
            ("#d7191c", f"High CH₄ (≈ {hi} ppm·m)"),
            ("#f7f7f7", "Near zero"),
            ("#2c7bb6", f"Low / negative (≈ {lo} ppm·m)"),
            ("#ffff00", "Plume boundary"),
        ]
    else:
        rows = [
            ("#d7191c", "High"),
            ("#ffffff", "No data"),
        ]

    items = "".join(
        f'<div class="legend-row">'
        f'<span class="legend-swatch" style="background:{c};"></span>'
        f'<span>{t}</span></div>'
        for c, t in rows
    )
    extra = ""
    if satellite:
        extra += (
            f'<div class="legend-row" style="margin-top:0.35rem;">'
            f'<b>Satellite:</b> {satellite}</div>'
        )
    if n_pixels is not None:
        extra += (
            f'<div class="legend-row">'
            f'<b>Plume pixels:</b> {n_pixels:,}</div>'
        )
    if mean_enh is not None:
        extra += (
            f'<div class="legend-row">'
            f'<b>Mean enh.:</b> {mean_enh:.0f} ppm·m</div>'
        )
    return (
        f'<div class="result-legend">'
        f'<div class="legend-heading">Legend</div>{items}{extra}</div>'
    )


# ══════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="EMIT + Tanager-1 Methane Detection",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
:root {
    --red: #e63946;
    --honeydew: #f1faee;
    --frost: #a8dadc;
    --blue: #457b9d;
    --navy: #1d3557;
    --black: #111111;
    --white: #ffffff;
    --border: #d8e6e8;
    --muted: #4f5d63;
    --dark-field: #292a33;
}
.stApp { background: #f1faee; color: #111111 !important; }
[data-testid="stHeader"] { background: #f1faee !important; height: 3.25rem !important; }
[data-testid="stSidebar"] { display: none; }
.block-container { max-width: 1700px; padding-top: 3.9rem !important; padding-bottom: 0.8rem; padding-left: 1.2rem; padding-right: 1.2rem; }
.app-header { position: relative; z-index: 10; display: flex; align-items: center; justify-content: space-between; background: #ffffff; border: 1px solid var(--border); border-radius: 16px; padding: 0.75rem 1rem; margin-top: 0.15rem; margin-bottom: 0.9rem; box-shadow: 0 2px 10px rgba(29,53,87,0.05); }
.app-title { color: #111111 !important; font-size: 1.45rem; font-weight: 850; line-height: 1.1; }
.app-subtitle { color: #111111 !important; font-size: 0.78rem; margin-top: 0.15rem; }
.status-pill { background: #f1faee; color: #111111 !important; border: 1px solid #a8dadc; border-radius: 999px; padding: 0.35rem 0.7rem; font-size: 0.72rem; font-weight: 750; white-space: nowrap; }
.app-card { background: #ffffff; border: 1px solid var(--border); border-radius: 15px; padding: 0.75rem; box-shadow: 0 2px 10px rgba(29,53,87,0.04); height: 100%; color: #111111 !important; }
.card-title { color: #111111 !important; font-size: 1rem; font-weight: 800; margin-bottom: 0.1rem; }
.card-caption { color: #111111 !important; font-size: 0.73rem; margin-bottom: 0.45rem; }
.section-label { display: inline-block; background: #a8dadc; color: #111111 !important; border-radius: 999px; padding: 0.2rem 0.55rem; font-size: 0.65rem; font-weight: 800; letter-spacing: 0.03em; margin-bottom: 0.35rem; }
.stApp p, .stApp label, .stApp small, .stApp strong, .stApp em, .stApp li, .stApp td, .stApp th, .stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stMarkdownContainer"] p, .stApp [data-testid="stMarkdownContainer"] span, .stApp [data-testid="stMarkdownContainer"] li { color: #111111 !important; }
div[data-testid="stDateInput"] div[data-baseweb="input"], div[data-testid="stDateInput"] div[data-baseweb="input"] > div, div[data-testid="stDateInput"] input, div[data-testid="stDateInput"] input[type="text"], .stDateInput div[data-baseweb="input"], .stDateInput div[data-baseweb="input"] > div, .stDateInput input, .stDateInput input[type="text"] { background-color: var(--dark-field) !important; color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; caret-color: #ffffff !important; opacity: 1 !important; }
div[data-testid="stDateInput"] input::-webkit-datetime-edit, div[data-testid="stDateInput"] input::-webkit-datetime-edit-text, div[data-testid="stDateInput"] input::-webkit-datetime-edit-month-field, div[data-testid="stDateInput"] input::-webkit-datetime-edit-day-field, div[data-testid="stDateInput"] input::-webkit-datetime-edit-year-field, div[data-testid="stDateInput"] input::-webkit-datetime-edit-fields-wrapper, .stDateInput input::-webkit-datetime-edit, .stDateInput input::-webkit-datetime-edit-text, .stDateInput input::-webkit-datetime-edit-month-field, .stDateInput input::-webkit-datetime-edit-day-field, .stDateInput input::-webkit-datetime-edit-year-field, .stDateInput input::-webkit-datetime-edit-fields-wrapper { color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; opacity: 1 !important; }
div[data-testid="stNumberInput"] input, div[data-testid="stTextInput"] input, .stNumberInput input, .stTextInput input { background-color: var(--dark-field) !important; color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; caret-color: #ffffff !important; }
input::placeholder, textarea::placeholder { color: #bfc3cc !important; opacity: 1 !important; }
div[data-baseweb="select"] input, div[data-baseweb="select"] [role="combobox"], div[data-baseweb="select"] * { color: #111111 !important; }
div[data-baseweb="popover"] [role="listbox"], div[data-baseweb="popover"] ul[role="listbox"], div[data-baseweb="popover"] [role="option"], div[data-baseweb="popover"] li[role="option"] { background: #111318 !important; }
div[data-baseweb="popover"] [role="listbox"] *, div[data-baseweb="popover"] [role="option"] *, ul[role="listbox"] *, li[role="option"] * { color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; }
div[data-baseweb="popover"] [role="option"]:hover, div[data-baseweb="popover"] li[role="option"]:hover { background: #2b2e38 !important; }
.stDateInput, .stSlider, .stNumberInput, .stSelectbox { margin-bottom: 0.15rem; }
.stSlider > div { padding-top: 0.05rem; padding-bottom: 0.05rem; }
.stSlider label, .stSlider [data-testid="stTickBar"] * { color: #111111 !important; }
.stSlider [data-testid="stThumbValue"], .stSlider [data-testid="stThumbValue"] * { color: #ffffff !important; }
[data-baseweb="calendar"] *, [data-baseweb="popover"] [data-baseweb="calendar"] *, [data-baseweb="calendar"] button { color: #ffffff !important; }
input:-webkit-autofill, input:-webkit-autofill:hover, input:-webkit-autofill:focus { -webkit-text-fill-color: #ffffff !important; caret-color: #ffffff !important; }

/* BUTTONS */
.stButton > button, .stDownloadButton > button {
    border-radius: 9px;
    min-height: 2.15rem;
    font-weight: 750;
    font-size: 0.78rem;
}
.stButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"] {
    background: #e63946 !important;
    border: 1px solid #e63946 !important;
    color: #ffffff !important;
}
.stButton > button[kind="primary"] *,
.stDownloadButton > button[kind="primary"] * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}
.stButton > button[kind="primary"]:hover,
.stDownloadButton > button[kind="primary"]:hover {
    background: #c92f3b !important;
    border-color: #c92f3b !important;
}
.stButton > button[kind="secondary"],
.stDownloadButton > button {
    background: #1d3557 !important;
    color: #ffffff !important;
    border: 1px solid #1d3557 !important;
}
.stButton > button[kind="secondary"] *,
.stDownloadButton > button * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}
.stButton > button[kind="secondary"]:hover,
.stDownloadButton > button:hover {
    background: #ffffff !important;
    color: #111111 !important;
    border: 1px solid #1d3557 !important;
}
.stButton > button[kind="secondary"]:hover *,
.stDownloadButton > button:hover * {
    color: #111111 !important;
    -webkit-text-fill-color: #111111 !important;
}
.stButton > button:disabled,
.stDownloadButton > button:disabled { opacity: 0.55 !important; }

[data-testid="stImage"] {
    max-width: 100% !important;
    overflow: hidden;
    border-radius: 6px;
}
[data-testid="stImage"] > img {
    max-width: 100% !important;
    height: auto !important;
    display: block;
}

.auth-card { background: #f8fbfb; border: 1px solid #d7e4e7; border-radius: 11px; padding: 0.65rem 0.75rem; margin-top: 0.45rem; }
.auth-status { background: #e8f7ea; border: 1px solid #9ed2a4; color: #155724 !important; border-radius: 9px; padding: 0.45rem 0.6rem; font-size: 0.76rem; font-weight: 700; margin-bottom: 0.45rem; }
.auth-help { color: #111111 !important; font-size: 0.72rem; line-height: 1.45; margin: 0.2rem 0 0.45rem 0; }
div[data-testid="stDataFrame"] { border: 1px solid var(--border); }
div[data-testid="stDataFrame"] * { color: #111111 !important; }
.result-legend { background: #ffffff; border: 1px solid #d7e4e7; border-radius: 10px; padding: 0.75rem 0.7rem; min-height: 96px; box-sizing: border-box; display: flex; flex-direction: column; justify-content: center; gap: 0.42rem; }
.result-legend .legend-heading { color: #111111 !important; font-size: 0.88rem; font-weight: 800; }
.result-legend .legend-row { display: flex; align-items: center; gap: 0.45rem; color: #111111 !important; font-size: 0.78rem; line-height: 1.25; }
.legend-swatch { width: 18px; height: 14px; min-width: 18px; border: 1px solid #555; border-radius: 2px; display: inline-block; }
.result-card { background: #ffffff; border: 1px solid #d8e6e8; border-radius: 12px; padding: 0.6rem; }
.result-tag { display: inline-block; background: #a8dadc; color: #111111 !important; border-radius: 999px; padding: 0.12rem 0.45rem; font-size: 0.6rem; font-weight: 800; letter-spacing: 0.03em; margin-bottom: 0.2rem; }
.result-name { color: #111111 !important; font-size: 0.9rem; font-weight: 800; margin-bottom: 0.35rem; }
.result-note { background: #f8fbfb; border: 1px solid #d7e4e7; border-radius: 10px; padding: 0.55rem 0.7rem; font-size: 0.8rem; color: #111111 !important; margin-top: 0.45rem; }
.mouse-readout { background: #f8fbfb; border: 1px dashed #a8dadc; border-radius: 9px; padding: 0.35rem 0.6rem; font-size: 0.74rem; color: #111111 !important; margin-top: 0.35rem; }
.satellite-badge { display: inline-block; border-radius: 999px; padding: 0.1rem 0.45rem; font-size: 0.62rem; font-weight: 800; margin-right: 0.3rem; }
.satellite-badge.emit { background: #dbeafe; color: #1d3557 !important; border: 1px solid #457b9d; }
.satellite-badge.tanager { background: #fde8ea; color: #7a1f27 !important; border: 1px solid #e63946; }
.overlap-banner { background: linear-gradient(90deg, #e8f7ea 0%, #f1faee 100%); border: 1px solid #9ed2a4; border-radius: 11px; padding: 0.55rem 0.75rem; font-size: 0.8rem; font-weight: 700; color: #155724 !important; margin-bottom: 0.45rem; }
.compare-header { display: flex; align-items: center; gap: 0.45rem; font-size: 0.85rem; font-weight: 800; margin-bottom: 0.3rem; }
footer { visibility: hidden; }
.stMarkdown { margin-bottom: 0.1rem; }
.element-container { margin-bottom: 0.15rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
    <div>
        <div class="app-title">🛰️ EMIT + Tanager-1 Methane Detection</div>
        <div class="app-subtitle">NASA EMIT &nbsp;•&nbsp; Planet Tanager-1 &nbsp;|&nbsp; Carbon Mapper-style matched-filter enhancements &nbsp;|&nbsp; 30–60 m native</div>
    </div>
    <div class="status-pill">EMIT 60 m &nbsp;•&nbsp; Tanager 30 m &nbsp;•&nbsp; HyperSpectral</div>
</div>
""", unsafe_allow_html=True)

if not EARTHACCESS_AVAILABLE:
    st.error(
        "⚠️ The `earthaccess` package is not installed. "
        "Add it to your `requirements.txt` and reboot the app."
    )
    st.stop()

if "aoi" not in st.session_state:
    st.session_state.aoi = mapping(DEFAULT_AOI)


# ══════════════════════════════════════════════════════════════════════
#  01 · STUDY AREA  +  02 · SEARCH
# ══════════════════════════════════════════════════════════════════════

map_col, control_col = st.columns([1.65, 1.0], gap="small")

with map_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">01 · STUDY AREA</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Area of Interest</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-caption">Search by place name, enter coordinates manually, '
        'or draw the study area directly on the map with the polygon tool. '
        'Live mouse coordinates appear in the bottom-right corner of the map.</div>',
        unsafe_allow_html=True,
    )

    ps1, ps2 = st.columns([3, 1], gap="small")
    with ps1:
        place_query = st.text_input(
            "Place name",
            placeholder="e.g. Tehran, Paris, Permian Basin, Riyadh…",
            key="place_query",
            label_visibility="collapsed",
        )
    with ps2:
        search_place_clicked = st.button(
            "🔍 Find place",
            use_container_width=True,
            key="search_place_btn",
        )

    if search_place_clicked:
        if not place_query.strip():
            st.warning("Please type a place name first.")
        else:
            with st.spinner("Geocoding place name…"):
                geom, center, label = geocode_place(place_query.strip())
            if geom is not None:
                st.session_state.aoi = mapping(geom)
                st.session_state["aoi_source"] = f"Place: {label[:90]}"
                st.session_state["_ignore_drawings_once"] = True
                st.success(f"Found: {label[:120]}")
            else:
                st.warning(
                    "Place not found. Try a more specific name or use coordinates."
                )

    with st.expander("📍 Or enter coordinates manually"):
        mc1, mc2, mc3 = st.columns(3, gap="small")
        with mc1:
            manual_lat = st.number_input(
                "Latitude",
                value=35.50, min_value=-90.0, max_value=90.0,
                step=0.01, format="%.4f", key="manual_lat",
            )
        with mc2:
            manual_lon = st.number_input(
                "Longitude",
                value=51.30, min_value=-180.0, max_value=180.0,
                step=0.01, format="%.4f", key="manual_lon",
            )
        with mc3:
            manual_size = st.number_input(
                "Half-size (°)",
                value=0.10, min_value=0.005, max_value=5.0,
                step=0.005, format="%.3f", key="manual_size",
            )
        if st.button("Apply coordinates", use_container_width=True, key="apply_coords"):
            st.session_state.aoi = mapping(box(
                manual_lon - manual_size, manual_lat - manual_size,
                manual_lon + manual_size, manual_lat + manual_size,
            ))
            st.session_state["aoi_source"] = (
                f"Manual: ({manual_lat:.4f}, {manual_lon:.4f}) ± {manual_size:.3f}°"
            )
            st.session_state["_ignore_drawings_once"] = True
            st.success("AOI set from coordinates.")

    if st.session_state.get("aoi_source"):
        st.markdown(
            f'<div class="card-caption">Current AOI: '
            f'<b>{st.session_state["aoi_source"]}</b></div>',
            unsafe_allow_html=True,
        )

    # Overlay Tanager plume geometries on map if available
    _map_layers = []
    _tan_feats = st.session_state.get("tanager_features", [])
    if _tan_feats:
        _map_layers.append(
            ("Tanager-1 plumes", cm_plume_geojson(_tan_feats), SATELLITES["Tanager-1"]["color"])
        )

    map_data = st_folium(
        create_map(st.session_state.aoi, extra_layers=_map_layers or None),
        height=385,
        width=1000,
        key="aoi_map",
    )

    ignore_drawings = st.session_state.pop("_ignore_drawings_once", False)
    if not ignore_drawings and map_data and map_data.get("all_drawings"):
        new_aoi = normalize_geometry(
            {"type": "FeatureCollection", "features": map_data["all_drawings"]}
        )
        if new_aoi and new_aoi != st.session_state.aoi:
            st.session_state.aoi = new_aoi
            st.session_state["aoi_source"] = "Custom polygon (drawn)"
            st.rerun()

    last_clicked = map_data.get("last_clicked") if map_data else None
    if last_clicked:
        lat_c = last_clicked.get("lat")
        lon_c = last_clicked.get("lng")
        st.markdown(
            f'<div class="mouse-readout">'
            f'🖱️ Last click &nbsp;→&nbsp; '
            f'<b>Lat:</b> {lat_c:.5f} &nbsp;·&nbsp; <b>Lon:</b> {lon_c:.5f}'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="mouse-readout">'
            '🖱️ Live mouse coordinates shown in the bottom-right corner of the map. '
            'Click on the map to pin a coordinate here.'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

with control_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">02 · SEARCH</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Satellite Search</div>', unsafe_allow_html=True)

    default_end = datetime.now().date()
    default_start = default_end - timedelta(days=365)

    d1, d2 = st.columns(2, gap="small")
    with d1:
        start_date = st.date_input("Start date", default_start, key="start_date")
    with d2:
        end_date = st.date_input("End date", default_end, key="end_date")

    # ── Satellite selector ──
    satellite_choice = st.multiselect(
        "Satellites to search",
        options=list(SATELLITES.keys()),
        default=list(SATELLITES.keys()),
        format_func=lambda k: f"{SATELLITES[k]['icon']} {SATELLITES[k]['label']}",
        key="satellite_choice",
    )

    st.markdown(
        '<div class="card-caption">EMIT covers ~75 km swaths (60 m). '
        'Tanager-1 covers ~18 km swaths (30 m). '
        'A wider window improves the chance of finding overlapping data.</div>',
        unsafe_allow_html=True,
    )

    if st.button("🔎  Search satellites", type="primary", use_container_width=True):
        if not satellite_choice:
            st.warning("Select at least one satellite.")
        else:
            try:
                with st.spinner("Authenticating…"):
                    if "EMIT" in satellite_choice:
                        login_earthdata()

                with st.spinner("Searching EMIT and Tanager-1…"):
                    results = search_all_satellites(
                        st.session_state.aoi, start_date, end_date,
                        satellites=satellite_choice,
                    )

                st.session_state["search_results"] = results
                st.session_state["tanager_features"] = results.get("Tanager-1", [])
                st.session_state.pop("selected_granule", None)
                st.session_state.pop("selected_tanager", None)
                st.session_state.pop("emit_result", None)

                for err in results.get("errors", []):
                    st.warning(err)

                n_emit = len(results.get("EMIT", []))
                n_tan = len(results.get("Tanager-1", []))
                if n_emit or n_tan:
                    st.success(
                        f"Found {n_emit} EMIT granule(s) and {n_tan} Tanager-1 plume(s)"
                    )
                else:
                    st.warning(
                        "No data found for this AOI and time range. "
                        "Try a wider date range."
                    )
            except Exception as e:
                st.session_state["search_results"] = {"EMIT": [], "Tanager-1": [], "errors": [str(e)]}
                st.error(f"Search failed: {e}")

    search_results = st.session_state.get("search_results", {"EMIT": [], "Tanager-1": []})
    emit_results = search_results.get("EMIT", [])
    tanager_results = search_results.get("Tanager-1", [])

    # ── Unified results table ──
    if emit_results or tanager_results:
        rows = []
        for g in emit_results:
            dt = granule_datetime(g)
            rows.append({
                "date": dt,
                "satellite": "EMIT",
                "resolution": "60 m",
                "cloud": granule_cloud(g),
                "id": g.get("meta", {}).get("native-id", "unknown")[:40],
                "_idx": len(rows),
            })
        for f in tanager_results:
            dt = cm_plume_datetime(f)
            props = f.get("properties", {})
            rows.append({
                "date": dt,
                "satellite": "Tanager-1",
                "resolution": "30 m",
                "cloud": None,
                "id": props.get("plume_id", "unknown")[:40],
                "_idx": len(rows),
            })
        table = pd.DataFrame(rows).sort_values(["date", "satellite"], na_position="last")
        st.dataframe(
            table[["date", "satellite", "resolution", "cloud", "id"]],
            use_container_width=True,
            height=140,
            hide_index=True,
            column_config={
                "date": st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD HH:mm"),
                "cloud": st.column_config.NumberColumn("Cloud %", format="%.1f"),
            },
        )

        # ── Overlap detection ──
        overlap_dates = find_overlap_dates(emit_results, tanager_results)
        if overlap_dates:
            date_strs = ", ".join(d.strftime("%Y-%m-%d") for d in overlap_dates[:5])
            more = f" (+{len(overlap_dates)-5} more)" if len(overlap_dates) > 5 else ""
            st.markdown(
                f'<div class="overlap-banner">'
                f'🎯 {len(overlap_dates)} overlap day(s) with BOTH satellites: '
                f'{date_strs}{more}</div>',
                unsafe_allow_html=True,
            )
        elif "EMIT" in satellite_choice and "Tanager-1" in satellite_choice:
            st.info(
                "No calendar-day overlap found between EMIT and Tanager-1 "
                "in this window. Try a wider date range."
            )

        # ── Granule / plume selector ──
        def format_item(idx):
            r = table.iloc[idx]
            dt = r["date"]
            dt_text = dt.strftime("%Y-%m-%d %H:%M") if pd.notna(dt) else "unknown"
            return f"[{r['satellite']}] {dt_text} · {r['id']}"

        selected_idx = st.selectbox(
            "Select observation",
            list(range(len(table))),
            format_func=lambda x: format_item(x),
            key="observation_select",
        )
        chosen_row = table.iloc[selected_idx]

        if chosen_row["satellite"] == "EMIT":
            # Map back to EMIT granule
            emit_rows = table[table["satellite"] == "EMIT"].reset_index(drop=True)
            emit_pos = emit_rows[emit_rows["id"] == chosen_row["id"]].index
            if len(emit_pos):
                st.session_state["selected_granule"] = emit_results[emit_pos[0]]
                st.session_state.pop("selected_tanager", None)
        else:
            tan_rows = table[table["satellite"] == "Tanager-1"].reset_index(drop=True)
            tan_pos = tan_rows[tan_rows["id"] == chosen_row["id"]].index
            if len(tan_pos):
                st.session_state["selected_tanager"] = tanager_results[tan_pos[0]]
                st.session_state.pop("selected_granule", None)

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  03 · DETECTION  +  04 · PROCESS
# ══════════════════════════════════════════════════════════════════════

st.markdown('<div style="height:0.25rem"></div>', unsafe_allow_html=True)

settings_col, action_col = st.columns([1.65, 1.0], gap="small")

with settings_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">03 · DETECTION</div>', unsafe_allow_html=True)

    p1, p2 = st.columns(2, gap="small")
    with p1:
        PARAMS["plume_threshold_ppm_m"] = st.number_input(
            "Enhancement threshold (ppm·m)",
            min_value=100.0,
            max_value=10000.0,
            value=float(PARAMS["plume_threshold_ppm_m"]),
            step=100.0,
            key="plume_threshold",
        )
    with p2:
        PARAMS["min_plume_pixels"] = st.number_input(
            "Minimum plume pixels",
            min_value=1,
            max_value=500,
            value=int(PARAMS["min_plume_pixels"]),
            step=1,
            key="min_plume_pixels",
        )

    _emit_area = int(PARAMS["min_plume_pixels"]) * 60 * 60
    _tan_area = int(PARAMS["min_plume_pixels"]) * 30 * 30
    st.markdown(
        f'<div class="card-caption">'
        f'Minimum plume area: <b>EMIT</b> ≈ {_emit_area:,} m² (60 m) · '
        f'<b>Tanager-1</b> ≈ {_tan_area:,} m² (30 m). '
        f'Wind speed is fetched automatically from Open-Meteo (ERA5) for each '
        f'observation. If unavailable, a fallback of '
        f'{PARAMS["wind_speed_m_s"]:.1f} m/s is used.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

with action_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">04 · PROCESS</div>', unsafe_allow_html=True)

    selected_granule = st.session_state.get("selected_granule")
    selected_tanager = st.session_state.get("selected_tanager")

    has_selection = selected_granule is not None or selected_tanager is not None

    if has_selection:
        if selected_granule is not None:
            dt = granule_datetime(selected_granule)
            sat = "EMIT"
        else:
            dt = cm_plume_datetime(selected_tanager)
            sat = "Tanager-1"
        dt_text = dt.strftime("%Y-%m-%d %H:%M") if dt else "unknown date"
        st.markdown(
            f'<div class="card-title">Ready to detect</div>'
            f'<div class="card-caption">Satellite: <b>{sat}</b> · Observation: {dt_text}</div>',
            unsafe_allow_html=True,
        )

        run_detect = st.button(
            "🚀  Run Methane Detection",
            type="primary",
            use_container_width=True,
            key="run_detect",
        )

        if run_detect:
            progress = st.progress(0, text="Authenticating…")
            try:
                progress.progress(10, text="Logging in…")

                if selected_granule is not None:
                    # ── EMIT path ──
                    login_earthdata()
                    res = SATELLITES["EMIT"]["resolution"]

                    progress.progress(30, text="Loading EMIT enhancement…")
                    data, transform, crs = load_emit_enhancement(
                        selected_granule, st.session_state.aoi, resolution=res
                    )

                    if data is None or data.size == 0:
                        st.error("EMIT granule did not intersect the AOI.")
                        st.stop()

                    progress.progress(55, text="Fetching wind from Open-Meteo…")
                    _centroid = shape(st.session_state.aoi).centroid
                    _wind = get_wind_speed_openmeteo(
                        _centroid.y, _centroid.x, dt
                    ) if dt else None

                    if _wind is not None:
                        wind_speed_to_use = _wind
                        st.info(f"✅ Wind from Open-Meteo (ERA5): {wind_speed_to_use:.2f} m/s")
                    else:
                        wind_speed_to_use = PARAMS["wind_speed_m_s"]
                        st.warning(
                            f"⚠️ Open-Meteo wind unavailable — using fallback: "
                            f"{wind_speed_to_use:.2f} m/s"
                        )

                    progress.progress(70, text="Detecting plumes…")
                    plume_mask = detect_plume(
                        data,
                        PARAMS["plume_threshold_ppm_m"],
                        int(PARAMS["min_plume_pixels"]),
                    )

                    progress.progress(85, text="Estimating flux…")
                    flux = estimate_flux_ime(
                        data, plume_mask, wind_speed_to_use, resolution=res
                    )

                    st.session_state.emit_result = {
                        "enhancement": data,
                        "plume_mask": plume_mask,
                        "flux": flux,
                        "transform": transform,
                        "crs": crs,
                        "granule_dt": dt,
                        "threshold": PARAMS["plume_threshold_ppm_m"],
                        "wind_speed": wind_speed_to_use,
                        "satellite": "EMIT",
                        "resolution": res,
                    }
                    st.session_state.pop("tanager_result", None)

                else:
                    # ── Tanager-1 path ──
                    res = SATELLITES["Tanager-1"]["resolution"]
                    progress.progress(30, text="Loading Tanager-1 plume data…")

                    props = selected_tanager.get("properties", {})
                    reported_flux = cm_plume_emission(selected_tanager)
                    cm_wind = cm_plume_wind(selected_tanager)

                    # Try to load L2B raster (may be unpublished)
                    data, transform, crs = load_tanager_enhancement(
                        selected_tanager, st.session_state.aoi
                    )

                    has_raster = data is not None and data.size > 0

                    progress.progress(55, text="Fetching wind from Open-Meteo…")
                    _centroid = shape(st.session_state.aoi).centroid
                    _wind = get_wind_speed_openmeteo(
                        _centroid.y, _centroid.x, dt
                    ) if dt else None
                    if _wind is None:
                        _wind = cm_wind
                    if _wind is None:
                        _wind = PARAMS["wind_speed_m_s"]

                    wind_speed_to_use = _wind

                    if has_raster:
                        progress.progress(70, text="Detecting plumes…")
                        plume_mask = detect_plume(
                            data,
                            PARAMS["plume_threshold_ppm_m"],
                            int(PARAMS["min_plume_pixels"]),
                        )
                        progress.progress(85, text="Estimating flux…")
                        flux = estimate_flux_ime(
                            data, plume_mask, wind_speed_to_use, resolution=res
                        )
                    else:
                        # No raster — use Carbon Mapper reported values
                        plume_mask = np.zeros((1, 1), dtype=bool)
                        flux = {
                            "Q_kg_h": reported_flux,
                            "Q_ton_h": reported_flux / 1000.0,
                            "IME_ppm_m2": 0.0,
                            "IME_kg": 0.0,
                            "plume_area_m2": 0.0,
                            "length_m": 0.0,
                            "U_eff_m_s": ALPHA_IME * wind_speed_to_use + BETA_IME,
                            "n_pixels": 0,
                            "max_enhancement": 0.0,
                            "mean_enhancement": 0.0,
                            "reported_flux_kg_h": reported_flux,
                        }
                        st.info(
                            "ℹ️ Tanager-1 L2B raster not yet published for this scene. "
                            f"Using Carbon Mapper reported emission rate: "
                            f"{reported_flux:.1f} kg/h"
                        )

                    st.session_state.tanager_result = {
                        "enhancement": data,
                        "plume_mask": plume_mask,
                        "flux": flux,
                        "transform": transform,
                        "crs": crs,
                        "granule_dt": dt,
                        "threshold": PARAMS["plume_threshold_ppm_m"],
                        "wind_speed": wind_speed_to_use,
                        "satellite": "Tanager-1",
                        "resolution": res,
                        "plume_feature": selected_tanager,
                        "reported_flux_kg_h": reported_flux,
                        "has_raster": has_raster,
                    }
                    st.session_state.pop("emit_result", None)

                progress.progress(100, text="Done")
                st.success("Detection complete")
            except Exception as e:
                st.error(f"Detection failed: {e}")
    else:
        st.markdown(
            '<div class="card-title">Select an observation first</div>'
            '<div class="card-caption">Search satellites, select an observation, then run the detection.</div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  05 · RESULTS  (with comparison mode)
# ══════════════════════════════════════════════════════════════════════

if "emit_result" in st.session_state or "tanager_result" in st.session_state:
    has_emit = "emit_result" in st.session_state
    has_tan = "tanager_result" in st.session_state

    st.markdown('<div style="height:0.25rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">05 · RESULTS</div>', unsafe_allow_html=True)

    # ── Comparison mode ──
    if has_emit and has_tan:
        st.markdown(
            '<div class="card-title">🔬 Side-by-side comparison</div>'
            '<div class="card-caption">Both satellites have data for this observation. '
            'Compare enhancement maps and flux estimates below.</div>',
            unsafe_allow_html=True,
        )

        emit_res = st.session_state.emit_result
        tan_res = st.session_state.tanager_result

        # Comparison metrics table
        def _metric_row(name, emit_val, tan_val, unit=""):
            return {"Metric": name, "EMIT": emit_val, "Tanager-1": tan_val, "Unit": unit}

        comp_rows = [
            _metric_row("Resolution", "60", "30", "m"),
            _metric_row("Flux (IME)", f"{emit_res['flux']['Q_kg_h']:.1f}",
                        f"{tan_res['flux']['Q_kg_h']:.1f}", "kg/h"),
            _metric_row("Plume pixels", f"{emit_res['flux']['n_pixels']:,}",
                        f"{tan_res['flux']['n_pixels']:,}", "px"),
            _metric_row("Plume area", f"{emit_res['flux']['plume_area_m2']/1e6:.4f}",
                        f"{tan_res['flux']['plume_area_m2']/1e6:.4f}", "km²"),
            _metric_row("Max enhancement", f"{emit_res['flux']['max_enhancement']:.0f}",
                        f"{tan_res['flux']['max_enhancement']:.0f}", "ppm·m"),
            _metric_row("Mean enhancement", f"{emit_res['flux']['mean_enhancement']:.0f}",
                        f"{tan_res['flux']['mean_enhancement']:.0f}", "ppm·m"),
            _metric_row("Wind speed", f"{emit_res['wind_speed']:.2f}",
                        f"{tan_res['wind_speed']:.2f}", "m/s"),
        ]

        # Add reported flux if available
        if tan_res.get("reported_flux_kg_h"):
            comp_rows.append(_metric_row(
                "Reported flux (CM)", "—",
                f"{tan_res['reported_flux_kg_h']:.1f}", "kg/h"
            ))

        st.dataframe(
            pd.DataFrame(comp_rows),
            use_container_width=True,
            hide_index=True,
            height=min(280, 38 * len(comp_rows) + 40),
        )

        # Side-by-side enhancement maps
        vmin_e, vmax_e = _compute_vrange(emit_res["enhancement"])
        vmin_t, vmax_t = _compute_vrange(tan_res["enhancement"])

        cc1, cc2 = st.columns(2, gap="small")
        with cc1:
            st.markdown(
                f'<div class="compare-header">'
                f'<span class="satellite-badge emit">EMIT</span>'
                f' Enhancement (60 m)</div>',
                unsafe_allow_html=True,
            )
            st.image(
                enhancement_png(
                    emit_res["enhancement"], mask=emit_res["plume_mask"],
                    colormap="turbo", show_outline=True,
                    vmin=vmin_e, vmax=vmax_e,
                ),
                use_container_width=True,
                output_format="PNG",
            )
            st.image(colorbar_png(vmin_e, vmax_e, "turbo"), width=90)
            st.markdown(
                legend_html("plume", vmin_e, vmax_e,
                            n_pixels=emit_res["flux"]["n_pixels"],
                            mean_enh=emit_res["flux"]["mean_enhancement"],
                            satellite="EMIT"),
                unsafe_allow_html=True,
            )
        with cc2:
            st.markdown(
                f'<div class="compare-header">'
                f'<span class="satellite-badge tanager">Tanager-1</span>'
                f' Enhancement (30 m)</div>',
                unsafe_allow_html=True,
            )
            if tan_res.get("has_raster"):
                st.image(
                    enhancement_png(
                        tan_res["enhancement"], mask=tan_res["plume_mask"],
                        colormap="turbo", show_outline=True,
                        vmin=vmin_t, vmax=vmax_t,
                    ),
                    use_container_width=True,
                    output_format="PNG",
                )
                st.image(colorbar_png(vmin_t, vmax_t, "turbo"), width=90)
                st.markdown(
                    legend_html("plume", vmin_t, vmax_t,
                                n_pixels=tan_res["flux"]["n_pixels"],
                                mean_enh=tan_res["flux"]["mean_enhancement"],
                                satellite="Tanager-1"),
                    unsafe_allow_html=True,
                )
            else:
                st.image(
                    placeholder_png("Tanager-1 L2B raster not yet published\n"
                                    f"Reported flux: {tan_res.get('reported_flux_kg_h', 0):.1f} kg/h"),
                    use_container_width=True,
                )

        st.markdown(
            f'<div class="result-note">'
            f'<b>Comparison note:</b> EMIT (60 m) and Tanager-1 (30 m) have '
            f'different spatial resolutions and matched-filter implementations. '
            f'Flux estimates are derived independently and are not expected to match '
            f'exactly. Use the reported Carbon Mapper flux as the authoritative value '
            f'for Tanager-1 when available.'
            f'</div>',
            unsafe_allow_html=True,
        )

    else:
        # ── Single-satellite results (existing behaviour preserved) ──
        result = st.session_state.get("emit_result") or st.session_state.get("tanager_result")
        flux = result["flux"]
        enhancement = result["enhancement"]
        plume_mask = result["plume_mask"]
        sat = result.get("satellite", "EMIT")
        res = result.get("resolution", 60)

        if enhancement is not None and enhancement.size > 0:
            vmin_enh, vmax_enh = _compute_vrange(enhancement)
        else:
            vmin_enh, vmax_enh = 0.0, 1.0

        metrics = st.columns(6, gap="small")
        metrics[0].metric("Flux (kg/h)", f"{flux['Q_kg_h']:.1f}")
        metrics[1].metric("Flux (t/h)", f"{flux['Q_ton_h']:.2f}")
        metrics[2].metric("Plume pixels", f"{flux['n_pixels']:,}")
        metrics[3].metric("Plume area", f"{flux['plume_area_m2']/1e6:.3f} km²")
        metrics[4].metric("Max enh. (ppm·m)", f"{flux['max_enhancement']:.0f}")
        metrics[5].metric("Mean enh. (ppm·m)", f"{flux['mean_enhancement']:.0f}")

        rc1, rc2 = st.columns(2, gap="small")
        with rc1:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="result-tag">{sat}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="result-name">CH₄ Enhancement ({res} m)</div>',
                unsafe_allow_html=True,
            )
            img_col, legend_col = st.columns([3.4, 1.2], gap="small")
            with img_col:
                if enhancement is not None and enhancement.size > 0:
                    st.image(
                        enhancement_png(enhancement, mask=plume_mask,
                                        colormap="turbo", show_outline=True,
                                        vmin=vmin_enh, vmax=vmax_enh),
                        use_container_width=True,
                        output_format="PNG",
                    )
                else:
                    st.image(
                        placeholder_png(
                            f"{sat} raster not available\n"
                            f"Reported flux: {flux.get('reported_flux_kg_h', flux['Q_kg_h']):.1f} kg/h"
                        ),
                        use_container_width=True,
                    )
            with legend_col:
                st.markdown('<div style="padding-top:0.3rem;"></div>', unsafe_allow_html=True)
                if enhancement is not None and enhancement.size > 0:
                    st.image(colorbar_png(vmin_enh, vmax_enh, "turbo"), use_container_width=True)
                st.markdown(
                    legend_html("plume", vmin_enh, vmax_enh,
                                n_pixels=flux["n_pixels"],
                                mean_enh=flux["mean_enhancement"],
                                satellite=sat),
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

        with rc2:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="result-tag">Plume mask</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="result-name">Detected methane plume ({sat})</div>',
                unsafe_allow_html=True,
            )
            img_col, legend_col = st.columns([3.4, 1.2], gap="small")
            with img_col:
                if enhancement is not None and enhancement.size > 0:
                    st.image(
                        enhancement_png(enhancement, mask=plume_mask,
                                        colormap="turbo", show_outline=True,
                                        outline_color=(255, 255, 0),
                                        vmin=vmin_enh, vmax=vmax_enh),
                        use_container_width=True,
                        output_format="PNG",
                    )
                else:
                    st.image(placeholder_png("Plume mask unavailable"), use_container_width=True)
            with legend_col:
                st.markdown('<div style="padding-top:0.3rem;"></div>', unsafe_allow_html=True)
                st.markdown(
                    legend_html("plume", vmin_enh, vmax_enh,
                                n_pixels=flux["n_pixels"],
                                mean_enh=flux["mean_enhancement"],
                                satellite=sat),
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

        # IME note
        if flux["n_pixels"] > 0:
            st.markdown(
                f'<div class="result-note">'
                f'<b>IME method ({sat}, {res} m):</b> '
                f'IME = {flux["IME_ppm_m2"]:.2e} ppm·m·m² · '
                f'{flux["IME_kg"]:.2f} kg CH₄ · '
                f'U_eff = {flux["U_eff_m_s"]:.2f} m/s · '
                f'L = {flux["length_m"]:.0f} m · '
                f'Q = {flux["Q_kg_h"]:.1f} kg/h'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Downloads ──
    st.markdown("#### 📥 Download results")
    dt_str = result.get("granule_dt")
    dt_tag = dt_str.strftime("%Y%m%d") if dt_str else "granule"
    sat_tag = sat.replace("-", "").lower()

    dl1, dl2, dl3 = st.columns(3, gap="small")

    with dl1:
        if enhancement is not None and enhancement.size > 0:
            png_data = enhancement_png(enhancement, mask=plume_mask,
                                        colormap="turbo", show_outline=True,
                                        vmin=vmin_enh, vmax=vmax_enh)
        else:
            png_data = placeholder_png(f"{sat} raster unavailable")
        st.download_button(
            "⬇ Enhancement PNG",
            png_data,
            file_name=f"enhancement_{sat_tag}_{dt_tag}.png",
            mime="image/png",
            use_container_width=True,
            key="dl_enh_png",
        )

    with dl2:
        csv = pd.DataFrame([flux]).to_csv(index=False)
        st.download_button(
            "⬇ Flux CSV",
            csv,
            file_name=f"flux_{sat_tag}_{dt_tag}.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_flux_csv",
        )

    with dl3:
        try:
            import zipfile
            if enhancement is not None and enhancement.size > 0 and transform is not None:
                geo_pkg = io.BytesIO()
                with zipfile.ZipFile(geo_pkg, "w", zipfile.ZIP_DEFLATED) as zf:
                    enh_tif = io.BytesIO()
                    with rasterio.open(
                        enh_tif, "w", driver="GTiff",
                        height=enhancement.shape[0], width=enhancement.shape[1],
                        count=1, dtype="float32", crs=crs, transform=transform,
                        nodata=np.nan, compress="deflate",
                    ) as dst:
                        dst.write(enhancement.astype(np.float32), 1)
                    zf.writestr("enhancement_ppmm.tif", enh_tif.getvalue())

                    if plume_mask is not None and plume_mask.shape == enhancement.shape:
                        mask_tif = io.BytesIO()
                        with rasterio.open(
                            mask_tif, "w", driver="GTiff",
                            height=plume_mask.shape[0], width=plume_mask.shape[1],
                            count=1, dtype="uint8", crs=crs, transform=transform,
                            nodata=0, compress="deflate",
                        ) as dst:
                            dst.write(plume_mask.astype(np.uint8), 1)
                        zf.writestr("plume_mask.tif", mask_tif.getvalue())

                st.download_button(
                    "⬇ GeoTIFF bundle",
                    geo_pkg.getvalue(),
                    file_name=f"{sat_tag}_{dt_tag}.zip",
                    mime="application/zip",
                    use_container_width=True,
                    key="dl_geo_zip",
                )
            else:
                st.button("⬇ GeoTIFF (unavailable)", disabled=True, use_container_width=True)
        except Exception:
            st.button("⬇ GeoTIFF (unavailable)", disabled=True, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  06 · MULTI-DATE COMPARISON  (upgraded for both satellites)
# ══════════════════════════════════════════════════════════════════════

search_results = st.session_state.get("search_results", {})
emit_results = search_results.get("EMIT", [])
tanager_results = search_results.get("Tanager-1", [])

if emit_results or tanager_results:
    st.markdown('<div style="height:0.25rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">06 · MULTI-DATE COMPARISON</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Compare plumes over time</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-caption">'
        'Processes all observations in the current search window and displays '
        'them side by side. Useful for tracking emission evolution over months.'
        '</div>',
        unsafe_allow_html=True,
    )

    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2], gap="small")
    with btn_col1:
        batch_sat = st.multiselect(
            "Satellites",
            options=[k for k in ["EMIT", "Tanager-1"]
                     if (k == "EMIT" and emit_results) or (k == "Tanager-1" and tanager_results)],
            default=[k for k in ["EMIT", "Tanager-1"]
                     if (k == "EMIT" and emit_results) or (k == "Tanager-1" and tanager_results)],
            key="batch_satellites",
            label_visibility="collapsed",
        )
    with btn_col2:
        run_batch = st.button(
            "🔁  Process all",
            type="primary",
            use_container_width=True,
            key="run_batch",
        )
    with btn_col3:
        max_granules = st.slider(
            "Max observations to process",
            min_value=2,
            max_value=20,
            value=6,
            key="max_granules",
        )

    if run_batch:
        progress = st.progress(0, text="Processing observations…")
        batch_results = []
        _centroid_batch = shape(st.session_state.aoi).centroid

        # Build combined list
        jobs = []
        if "EMIT" in batch_sat:
            for g in emit_results:
                jobs.append(("EMIT", g))
        if "Tanager-1" in batch_sat:
            for f in tanager_results:
                jobs.append(("Tanager-1", f))

        jobs = jobs[: int(max_granules)]

        for i, (sat, item) in enumerate(jobs):
            progress.progress(
                int(100 * (i + 1) / max(len(jobs), 1)),
                text=f"Processing {i+1}/{len(jobs)}…",
            )
            try:
                if sat == "EMIT":
                    data, tform, tcrs = load_emit_enhancement(
                        item, st.session_state.aoi,
                        resolution=SATELLITES["EMIT"]["resolution"],
                    )
                    if data is None or data.size == 0:
                        continue
                    if valid_coverage(data) < 0.02:
                        continue

                    g_dt = granule_datetime(item)
                    wind = get_wind_speed_openmeteo(
                        _centroid_batch.y, _centroid_batch.x, g_dt
                    ) if g_dt else None
                    if wind is None:
                        wind = PARAMS["wind_speed_m_s"]

                    pm = detect_plume(
                        data,
                        PARAMS["plume_threshold_ppm_m"],
                        int(PARAMS["min_plume_pixels"]),
                    )
                    f = estimate_flux_ime(
                        data, pm, wind,
                        resolution=SATELLITES["EMIT"]["resolution"],
                    )
                    batch_results.append({
                        "satellite": "EMIT",
                        "resolution": SATELLITES["EMIT"]["resolution"],
                        "date": g_dt,
                        "enhancement": data,
                        "plume_mask": pm,
                        "flux": f,
                        "transform": tform,
                        "crs": tcrs,
                        "wind_speed": wind,
                    })
                else:
                    data, tform, tcrs = load_tanager_enhancement(
                        item, st.session_state.aoi
                    )
                    g_dt = cm_plume_datetime(item)
                    reported = cm_plume_emission(item)
                    cm_wind = cm_plume_wind(item)
                    wind = cm_wind
                    if wind is None:
                        wind = get_wind_speed_openmeteo(
                            _centroid_batch.y, _centroid_batch.x, g_dt
                        ) if g_dt else None
                    if wind is None:
                        wind = PARAMS["wind_speed_m_s"]

                    if data is not None and data.size > 0:
                        pm = detect_plume(
                            data,
                            PARAMS["plume_threshold_ppm_m"],
                            int(PARAMS["min_plume_pixels"]),
                        )
                        f = estimate_flux_ime(
                            data, pm, wind,
                            resolution=SATELLITES["Tanager-1"]["resolution"],
                        )
                        has_raster = True
                    else:
                        pm = np.zeros((1, 1), dtype=bool)
                        f = {
                            "Q_kg_h": reported, "Q_ton_h": reported / 1000.0,
                            "IME_ppm_m2": 0.0, "IME_kg": 0.0,
                            "plume_area_m2": 0.0, "length_m": 0.0,
                            "U_eff_m_s": ALPHA_IME * wind + BETA_IME,
                            "n_pixels": 0, "max_enhancement": 0.0,
                            "mean_enhancement": 0.0,
                            "reported_flux_kg_h": reported,
                        }
                        has_raster = False

                    batch_results.append({
                        "satellite": "Tanager-1",
                        "resolution": SATELLITES["Tanager-1"]["resolution"],
                        "date": g_dt,
                        "enhancement": data,
                        "plume_mask": pm,
                        "flux": f,
                        "transform": tform,
                        "crs": tcrs,
                        "wind_speed": wind,
                        "has_raster": has_raster,
                        "reported_flux_kg_h": reported,
                    })
            except Exception:
                continue

        st.session_state.batch_results = batch_results
        progress.progress(100, text="Done")
        st.success(f"Processed {len(batch_results)} observation(s)")

    if st.session_state.get("batch_results"):
        batch = st.session_state.batch_results

        chart_rows = []
        for r in batch:
            if r["date"] is not None:
                chart_rows.append({
                    "date": r["date"],
                    "satellite": r["satellite"],
                    "flux_kg_h": r["flux"]["Q_kg_h"],
                    "plume_pixels": r["flux"]["n_pixels"],
                    "plume_area_km2": r["flux"]["plume_area_m2"] / 1e6,
                })
        if chart_rows:
            chart_df = pd.DataFrame(chart_rows).sort_values("date")

            # Dual-line chart (EMIT vs Tanager-1)
            pivot = chart_df.pivot_table(
                index="date", columns="satellite", values="flux_kg_h", aggfunc="first"
            )
            st.markdown("##### Estimated flux over time (EMIT vs Tanager-1)")
            st.line_chart(pivot, use_container_width=True, height=240)

            st.dataframe(
                chart_df.set_index("date"),
                use_container_width=True,
                hide_index=False,
                column_config={
                    "satellite": st.column_config.TextColumn("Satellite"),
                    "flux_kg_h": st.column_config.NumberColumn("Flux (kg/h)", format="%.1f"),
                    "plume_pixels": st.column_config.NumberColumn("Pixels", format="%d"),
                    "plume_area_km2": st.column_config.NumberColumn("Area (km²)", format="%.3f"),
                },
            )
            st.download_button(
                "⬇ Download time series CSV",
                chart_df.to_csv(index=False),
                file_name="multisat_flux_timeseries.csv",
                mime="text/csv",
                key="dl_ts_csv",
                use_container_width=False,
            )

        # Visual comparison slider
        st.markdown("##### Visual comparison")
        dates_labels = [
            f"[{r['satellite']}] " +
            (r["date"].strftime("%Y-%m-%d") if r["date"] else f"#{i+1}")
            for i, r in enumerate(batch)
        ]
        _batch_key = f"batch_slider_{len(batch)}"
        selected_idx = st.select_slider(
            "Select observation",
            options=list(range(len(batch))),
            format_func=lambda x: dates_labels[x],
            value=0,
            key=_batch_key,
        )
        chosen = batch[selected_idx]

        # Shared color range across all batch results that have raster
        _all_vals = []
        for r in batch:
            e = r.get("enhancement")
            if e is not None and np.isfinite(e).any():
                _all_vals.append(e[np.isfinite(e)])
        if _all_vals:
            _all_vals = np.concatenate(_all_vals)
            cvmin, cvmax = np.percentile(_all_vals, [2, 98])
        else:
            cvmin, cvmax = 0.0, 1.0
        if cvmax <= cvmin:
            cvmax = cvmin + 1.0

        _has_plume = chosen["flux"]["n_pixels"] > 0
        _has_raster = chosen.get("enhancement") is not None and chosen["enhancement"].size > 0
        _cov_lbl, _cov_col = coverage_badge(chosen["enhancement"]) if _has_raster else ("No raster", "#e63946")

        cc1, cc2 = st.columns(2, gap="small")
        with cc1:
            st.markdown(
                f'<div class="card-caption" style="font-weight:700;">'
                f'{dates_labels[selected_idx]} · Enhancement</div>'
                f'<div class="card-caption" style="color:{_cov_col} !important;'
                f'font-weight:700;">{_cov_lbl}</div>',
                unsafe_allow_html=True,
            )
            if _has_raster:
                st.image(
                    enhancement_png(chosen["enhancement"],
                                    mask=chosen["plume_mask"],
                                    colormap="turbo",
                                    show_outline=_has_plume,
                                    vmin=cvmin, vmax=cvmax),
                    use_container_width=True,
                    output_format="PNG",
                )
                st.image(colorbar_png(cvmin, cvmax, "turbo"), width=90)
            else:
                st.image(placeholder_png("Raster unavailable"), use_container_width=True)
        with cc2:
            if _has_plume:
                status_html = (
                    f'<div class="card-caption" style="color:#2a9d8f !important;'
                    f'font-weight:700;">✓ Plume detected</div>'
                )
            else:
                status_html = (
                    f'<div class="card-caption" style="color:#e63946 !important;'
                    f'font-weight:700;">✗ No plume above threshold '
                    f'({PARAMS["plume_threshold_ppm_m"]:.0f} ppm·m)</div>'
                )
            st.markdown(
                f'<div class="card-caption" style="font-weight:700;">'
                f'{dates_labels[selected_idx]} · Plume outline</div>{status_html}',
                unsafe_allow_html=True,
            )
            if _has_raster:
                st.image(
                    enhancement_png(chosen["enhancement"],
                                    mask=chosen["plume_mask"],
                                    colormap="turbo",
                                    show_outline=_has_plume,
                                    vmin=cvmin, vmax=cvmax),
                    use_container_width=True,
                    output_format="PNG",
                )
            st.markdown(
                legend_html("plume", cvmin, cvmax,
                            n_pixels=chosen["flux"]["n_pixels"],
                            mean_enh=chosen["flux"]["mean_enhancement"],
                            satellite=chosen["satellite"]),
                unsafe_allow_html=True,
            )
        m1, m2, m3 = st.columns(3, gap="small")
        m1.metric("Flux (kg/h)", f"{chosen['flux']['Q_kg_h']:.1f}")
        m2.metric("Plume pixels", f"{chosen['flux']['n_pixels']:,}")
        m3.metric("Plume area (km²)", f"{chosen['flux']['plume_area_m2']/1e6:.3f}")

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  07 · PLUME EVOLUTION WINDOW  (upgraded for both satellites)
# ══════════════════════════════════════════════════════════════════════

if "emit_result" in st.session_state or "tanager_result" in st.session_state:
    _res = st.session_state.get("emit_result") or st.session_state.get("tanager_result")
    _ref_dt = _res.get("granule_dt")
    _ref_sat = _res.get("satellite", "EMIT")

    if _ref_dt is not None:
        st.markdown('<div style="height:0.25rem"></div>', unsafe_allow_html=True)
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-label">07 · PLUME EVOLUTION WINDOW</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="card-title">Methane plume changes around the detected date</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="card-caption">'
            f'Searches all EMIT and Tanager-1 observations within a ±<i>N</i>-day '
            f'window around <b>{_ref_dt.strftime("%Y-%m-%d %H:%M")}</b> and shows how '
            f'the plume appears, disappears, moves, and grows or shrinks across the window.'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.info(
            "💡 Methane plumes are transient. A source may appear on some "
            "overpasses and not others due to intermittent emission, cloud cover, "
            "or wind dispersion. Only some observations showing a plume is "
            "normal and expected."
        )

        ec1, ec2, ec3 = st.columns([1, 1, 1], gap="small")
        with ec1:
            window_days = st.slider(
                "Window around detected date (± days)",
                min_value=5, max_value=45, value=15, step=1,
                key="evo_window_days",
            )
        with ec2:
            max_evo = st.slider(
                "Max observations to process",
                min_value=2, max_value=40, value=12, step=1,
                key="evo_max_granules",
            )
        with ec3:
            evo_sat_filter = st.radio(
                "Show evolution for:",
                ["Both", "EMIT only", "Tanager-1 only"],
                horizontal=True,
                key="evo_satellite_filter",
            )

        run_evo = st.button(
            "🔁  Analyze plume evolution",
            type="primary",
            use_container_width=True,
            key="run_evolution",
        )

        if run_evo:
            start_d = (_ref_dt - timedelta(days=int(window_days))).date()
            end_d = (_ref_dt + timedelta(days=int(window_days))).date()

            progress = st.progress(0, text="Searching satellites…")
            try:
                # Search both
                if evo_sat_filter == "Both":
                    _evo_sats = ["EMIT", "Tanager-1"]
                elif evo_sat_filter == "EMIT only":
                    _evo_sats = ["EMIT"]
                else:
                    _evo_sats = ["Tanager-1"]

                results_evo = search_all_satellites(
                    st.session_state.aoi, start_d, end_d,
                    satellites=_evo_sats,
                )

                evo_jobs = []
                for g in results_evo.get("EMIT", []):
                    evo_jobs.append(("EMIT", g))
                for f in results_evo.get("Tanager-1", []):
                    evo_jobs.append(("Tanager-1", f))

                evo_jobs.sort(key=lambda x: (
                    granule_datetime(x[1]) if x[0] == "EMIT"
                    else cm_plume_datetime(x[1])
                ) or datetime.min)
                evo_jobs = evo_jobs[: int(max_evo)]

                if not evo_jobs:
                    progress.progress(100, text="No observations found")
                    st.warning(
                        "No data found in this window. Try a wider ± window."
                    )
                else:
                    evo_results = []
                    _centroid_evo = shape(st.session_state.aoi).centroid

                    for i, (sat, item) in enumerate(evo_jobs):
                        progress.progress(
                            int(100 * (i + 1) / len(evo_jobs)),
                            text=f"Processing {i+1}/{len(evo_jobs)}…",
                        )
                        try:
                            if sat == "EMIT":
                                login_earthdata()
                                res = SATELLITES["EMIT"]["resolution"]
                                data, tform, tcrs = load_emit_enhancement(
                                    item, st.session_state.aoi, resolution=res
                                )
                                if data is None or data.size == 0:
                                    continue
                                cov = valid_coverage(data)
                                if cov < 0.02:
                                    continue
                                g_dt = granule_datetime(item)
                                wind = get_wind_speed_openmeteo(
                                    _centroid_evo.y, _centroid_evo.x, g_dt
                                ) if g_dt else None
                                if wind is None:
                                    wind = PARAMS["wind_speed_m_s"]
                                pm = detect_plume(
                                    data,
                                    PARAMS["plume_threshold_ppm_m"],
                                    int(PARAMS["min_plume_pixels"]),
                                )
                                f = estimate_flux_ime(data, pm, wind, resolution=res)
                                has_raster = True
                            else:
                                res = SATELLITES["Tanager-1"]["resolution"]
                                data, tform, tcrs = load_tanager_enhancement(
                                    item, st.session_state.aoi
                                )
                                g_dt = cm_plume_datetime(item)
                                reported = cm_plume_emission(item)
                                cm_wind = cm_plume_wind(item)
                                wind = cm_wind
                                if wind is None:
                                    wind = get_wind_speed_openmeteo(
                                        _centroid_evo.y, _centroid_evo.x, g_dt
                                    ) if g_dt else None
                                if wind is None:
                                    wind = PARAMS["wind_speed_m_s"]

                                if data is not None and data.size > 0:
                                    cov = valid_coverage(data)
                                    if cov < 0.02:
                                        continue
                                    pm = detect_plume(
                                        data,
                                        PARAMS["plume_threshold_ppm_m"],
                                        int(PARAMS["min_plume_pixels"]),
                                    )
                                    f = estimate_flux_ime(
                                        data, pm, wind, resolution=res
                                    )
                                    has_raster = True
                                else:
                                    cov = 0.0
                                    pm = np.zeros((1, 1), dtype=bool)
                                    f = {
                                        "Q_kg_h": reported, "Q_ton_h": reported / 1000.0,
                                        "IME_ppm_m2": 0.0, "IME_kg": 0.0,
                                        "plume_area_m2": 0.0, "length_m": 0.0,
                                        "U_eff_m_s": ALPHA_IME * wind + BETA_IME,
                                        "n_pixels": 0, "max_enhancement": 0.0,
                                        "mean_enhancement": 0.0,
                                        "reported_flux_kg_h": reported,
                                    }
                                    has_raster = False

                            # Centroid
                            centroid_geo = None
                            if pm.any() and tform is not None:
                                ys, xs = np.nonzero(pm)
                                cx_px = float(xs.mean())
                                cy_px = float(ys.mean())
                                try:
                                    from rasterio.transform import xy as rio_xy
                                    gx, gy = rio_xy(tform, cy_px, cx_px, offset="center")
                                    centroid_geo = (float(gx), float(gy))
                                except Exception:
                                    pass

                            evo_results.append({
                                "satellite": sat,
                                "resolution": res,
                                "date": g_dt,
                                "enhancement": data,
                                "plume_mask": pm,
                                "flux": f,
                                "transform": tform,
                                "crs": tcrs,
                                "centroid_geo": centroid_geo,
                                "coverage": cov,
                                "wind_speed": wind,
                                "has_raster": has_raster,
                                "reported_flux_kg_h": reported if sat == "Tanager-1" else None,
                            })
                        except Exception:
                            continue

                    st.session_state.evo_results = evo_results
                    st.session_state.evo_ref_date = _ref_dt
                    st.session_state.evo_window_days_used = int(window_days)
                    progress.progress(100, text="Done")
                    st.success(
                        f"Processed {len(evo_results)} observation(s) in a "
                        f"±{int(window_days)}-day window"
                    )
            except Exception as e:
                st.error(f"Evolution analysis failed: {e}")

        if st.session_state.get("evo_results"):
            evo = st.session_state.evo_results
            used_window = st.session_state.get("evo_window_days_used", window_days)

            n_total = len(evo)
            n_with = sum(1 for r in evo if r["flux"]["n_pixels"] > 0)
            n_without = n_total - n_with
            n_emit = sum(1 for r in evo if r["satellite"] == "EMIT")
            n_tan = sum(1 for r in evo if r["satellite"] == "Tanager-1")

            st.markdown(
                f'<div class="result-note">'
                f'<b>{n_with}</b> of <b>{n_total}</b> observation(s) in the '
                f'±{used_window}-day window showed a detectable plume '
                f'({n_emit} EMIT · {n_tan} Tanager-1). '
                f'<b>{n_without}</b> observation(s) showed no plume above the '
                f'threshold of {PARAMS["plume_threshold_ppm_m"]:.0f} ppm·m.'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Coverage table
            cov_rows = []
            for r in evo:
                if r["has_raster"]:
                    lbl, _ = coverage_badge(r["enhancement"])
                else:
                    lbl = "No raster"
                cov_rows.append({
                    "date": r["date"].strftime("%Y-%m-%d") if r["date"] else "-",
                    "satellite": r["satellite"],
                    "coverage": lbl,
                    "flux_kg_h": r["flux"]["Q_kg_h"],
                    "wind_m_s": r.get("wind_speed"),
                })
            st.markdown("##### Data coverage per observation")
            st.dataframe(
                pd.DataFrame(cov_rows),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "wind_m_s": st.column_config.NumberColumn("Wind (m/s)", format="%.2f"),
                },
            )

            # Evolution dataframe
            rows = []
            for r in evo:
                rows.append({
                    "date": r["date"],
                    "satellite": r["satellite"],
                    "flux_kg_h": r["flux"]["Q_kg_h"],
                    "plume_pixels": r["flux"]["n_pixels"],
                    "plume_area_km2": r["flux"]["plume_area_m2"] / 1e6,
                    "max_enh_ppmm": r["flux"]["max_enhancement"],
                    "mean_enh_ppmm": r["flux"]["mean_enhancement"],
                    "has_plume": int(r["flux"]["n_pixels"] > 0),
                    "wind_m_s": r.get("wind_speed"),
                })
            evo_df = pd.DataFrame(rows)
            if not evo_df.empty and evo_df["date"].notna().any():
                evo_df = evo_df.sort_values("date").set_index("date")

                st.markdown("##### Flux evolution")
                pivot_evo = evo_df.reset_index().pivot_table(
                    index="date", columns="satellite",
                    values="flux_kg_h", aggfunc="first",
                )
                st.line_chart(pivot_evo, use_container_width=True, height=220)

                st.markdown("##### Plume area evolution")
                pivot_area = evo_df.reset_index().pivot_table(
                    index="date", columns="satellite",
                    values="plume_area_km2", aggfunc="first",
                )
                st.line_chart(pivot_area, use_container_width=True, height=200)

                st.dataframe(
                    evo_df,
                    use_container_width=True,
                    hide_index=False,
                    column_config={
                        "satellite": st.column_config.TextColumn("Satellite"),
                        "flux_kg_h": st.column_config.NumberColumn("Flux (kg/h)", format="%.1f"),
                        "plume_pixels": st.column_config.NumberColumn("Pixels", format="%d"),
                        "plume_area_km2": st.column_config.NumberColumn("Area (km²)", format="%.3f"),
                        "max_enh_ppmm": st.column_config.NumberColumn("Max enh.", format="%.0f"),
                        "mean_enh_ppmm": st.column_config.NumberColumn("Mean enh.", format="%.0f"),
                        "has_plume": st.column_config.NumberColumn("Plume?", format="%d"),
                        "wind_m_s": st.column_config.NumberColumn("Wind (m/s)", format="%.2f"),
                    },
                )

                st.download_button(
                    "⬇ Download evolution CSV",
                    evo_df.to_csv(),
                    file_name="multisat_plume_evolution.csv",
                    mime="text/csv",
                    key="dl_evo_csv",
                    use_container_width=False,
                )

            # Centroid movement
            geo_pts = [
                (r["date"], r["centroid_geo"])
                for r in evo
                if r.get("centroid_geo") is not None and r.get("date") is not None
            ]
            if len(geo_pts) >= 2:
                st.markdown("##### Plume centroid movement")
                crows = []
                for d, (gx, gy) in geo_pts:
                    crows.append({"date": d, "x": gx, "y": gy})
                cdf = pd.DataFrame(crows).sort_values("date")
                x0, y0 = cdf.iloc[0]["x"], cdf.iloc[0]["y"]
                cdf["dx_px"] = (cdf["x"] - x0) / 30  # approximate
                cdf["dy_px"] = (cdf["y"] - y0) / 30
                st.dataframe(
                    cdf[["date", "dx_px", "dy_px"]],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "dx_px": st.column_config.NumberColumn("ΔX (px)", format="%.2f"),
                        "dy_px": st.column_config.NumberColumn("ΔY (px)", format="%.2f"),
                    },
                )

            # Shared color range
            all_valid = []
            for r in evo:
                e = r.get("enhancement")
                if e is not None and np.isfinite(e).any():
                    all_valid.append(e[np.isfinite(e)])
            if all_valid:
                all_valid = np.concatenate(all_valid)
                shared_vmin, shared_vmax = np.percentile(all_valid, [2, 98])
            else:
                shared_vmin, shared_vmax = 0.0, 1.0
            if shared_vmax <= shared_vmin:
                shared_vmax = shared_vmin + 1.0

            # Visual evolution slider
            st.markdown("##### Visual evolution")
            dates_labels = [
                f"[{r['satellite']}] " +
                (r["date"].strftime("%Y-%m-%d") if r["date"] else f"#{i+1}")
                for i, r in enumerate(evo)
            ]
            _evo_key = f"evo_slider_{len(evo)}"
            sel_idx = st.select_slider(
                "Select observation",
                options=list(range(len(evo))),
                format_func=lambda x: dates_labels[x],
                value=0,
                key=_evo_key,
            )
            chosen = evo[sel_idx]
            _has_plume = chosen["flux"]["n_pixels"] > 0
            _has_raster = chosen.get("has_raster", False)

            cc1, cc2 = st.columns(2, gap="small")
            with cc1:
                st.markdown(
                    f'<div class="card-caption" style="font-weight:700;">'
                    f'{dates_labels[sel_idx]} · Enhancement</div>',
                    unsafe_allow_html=True,
                )
                if _has_raster:
                    st.image(
                        enhancement_png(
                            chosen["enhancement"],
                            mask=chosen["plume_mask"],
                            colormap="turbo",
                            show_outline=_has_plume,
                            vmin=shared_vmin, vmax=shared_vmax,
                        ),
                        use_container_width=True,
                        output_format="PNG",
                    )
                    st.image(colorbar_png(shared_vmin, shared_vmax, "turbo"), width=90)
                else:
                    st.image(placeholder_png("Raster unavailable"), use_container_width=True)
            with cc2:
                if _has_plume:
                    status_html = (
                        f'<div class="card-caption" style="color:#2a9d8f !important;'
                        f'font-weight:700;">✓ Plume detected</div>'
                    )
                else:
                    status_html = (
                        f'<div class="card-caption" style="color:#e63946 !important;'
                        f'font-weight:700;">✗ No plume above threshold '
                        f'({PARAMS["plume_threshold_ppm_m"]:.0f} ppm·m)</div>'
                    )
                st.markdown(
                    f'<div class="card-caption" style="font-weight:700;">'
                    f'{dates_labels[sel_idx]} · Plume outline</div>{status_html}',
                    unsafe_allow_html=True,
                )
                if _has_raster:
                    st.image(
                        enhancement_png(
                            chosen["enhancement"],
                            mask=chosen["plume_mask"],
                            colormap="turbo",
                            show_outline=_has_plume,
                            vmin=shared_vmin, vmax=shared_vmax,
                        ),
                        use_container_width=True,
                        output_format="PNG",
                    )
                st.markdown(
                    legend_html("plume",
                                n_pixels=chosen["flux"]["n_pixels"],
                                mean_enh=chosen["flux"]["mean_enhancement"],
                                satellite=chosen["satellite"]),
                    unsafe_allow_html=True,
                )
            em1, em2, em3, em4 = st.columns(4, gap="small")
            em1.metric("Flux (kg/h)", f"{chosen['flux']['Q_kg_h']:.1f}")
            em2.metric("Plume pixels", f"{chosen['flux']['n_pixels']:,}")
            em3.metric("Plume area (km²)", f"{chosen['flux']['plume_area_m2']/1e6:.3f}")
            em4.metric("Max enh. (ppm·m)", f"{chosen['flux']['max_enhancement']:.0f}")

            # Gallery
            st.markdown("##### Plume mask gallery (all observations)")
            n_cols = 4
            n_obs = len(evo)
            grid_rows = (n_obs + n_cols - 1) // n_cols
            for gr in range(grid_rows):
                gcols = st.columns(n_cols, gap="small")
                for gc in range(n_cols):
                    idx = gr * n_cols + gc
                    if idx >= n_obs:
                        break
                    r = evo[idx]
                    label = (
                        r["date"].strftime("%Y-%m-%d")
                        if r["date"] else f"#{idx+1}"
                    )
                    _has = r["flux"]["n_pixels"] > 0
                    _raster = r.get("has_raster", False)
                    with gcols[gc]:
                        st.markdown(
                            f'<div class="card-caption" style="font-weight:700; '
                            f'text-align:center; margin-bottom:0.15rem;">'
                            f'{label}<br/>'
                            f'<span style="font-weight:400;">'
                            f'{r["satellite"]} · '
                            f'{r["flux"]["Q_kg_h"]:.0f} kg/h · '
                            f'{r["flux"]["n_pixels"]} px</span></div>',
                            unsafe_allow_html=True,
                        )
                        if _raster:
                            st.image(
                                enhancement_png(
                                    r["enhancement"],
                                    mask=r["plume_mask"],
                                    colormap="turbo",
                                    show_outline=_has,
                                    vmin=shared_vmin, vmax=shared_vmax,
                                ),
                                use_container_width=True,
                                output_format="PNG",
                            )
                        else:
                            st.image(placeholder_png("—"), use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)
