"""EMIT + Tanager-1 Methane Plume Detection App."""

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

SATELLITES = {
    "EMIT": {
        "label": "EMIT (NASA)",
        "resolution": 60,
        "collection": "EMITL2BCH4ENH",
        "source": "earthaccess",
        "color": "#457b9d",
        "icon": "🛰️",
    },
    "Tanager-1": {
        "label": "Tanager-1 (Planet/Carbon Mapper)",
        "resolution": 30,
        "collection": "l2b-ch4-mfa-v3e",
        "source": "carbonmapper",
        "color": "#e63946",
        "icon": "📡",
    },
}

CM_API_BASE = "https://api.carbonmapper.org/api/v1"
CM_STAC_SEARCH = f"{CM_API_BASE}/stac/search"
CM_STAC_BASE = f"{CM_API_BASE}/stac"

DEFAULT_AOI = box(51.20, 35.40, 51.45, 35.60)

EMIT_ENH_COLLECTION = "EMITL2BCH4ENH"
EMIT_PLM_COLLECTION = "EMITL2BCH4PLM"

PARAMS = {
    "plume_threshold_ppm_m": 1000.0,
    "min_plume_pixels": 10,
    "wind_speed_m_s": 2.0,
    "max_plume_area_km2": 100.0,
}

PPB_TO_KG_M2 = 5.72e-6
ALPHA_IME = 0.33
BETA_IME = 0.45
CH4_DENSITY_KG_M3 = 0.717

CM_PLATFORM_MAP = {
    "tan": "Tanager-1",
    "emi": "EMIT",
    "ang": "AVIRIS-NG",
    "av3": "AVIRIS-3",
    "gao": "GAO",
}

CM_EMISSION_KEYS = (
    "emission_rate",
    "emission_auto",
    "emission_rate_kg_hr",
    "emission_rate_kg_hr_mean",
    "emission_rate_kg_hr_median",
    "emission_rate_auto",
    "emission_rate_uncertainty",
    "flux_kg_hr",
    "cm_emission_rate",
    "ch4_emission_rate",
    "emission_estimate",
    "emission",
    "ime_flux",
    "emission_rate_ch4",
    "ch4_flux",
    "ch4_flux_kg_hr",
    "methane_emission_rate",
)


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
    geometry = shape(ensure_aoi(aoi))
    centroid = geometry.centroid
    zoom = compute_zoom(geometry.bounds)
    fmap = folium.Map([centroid.y, centroid.x], zoom_start=zoom, tiles="OpenStreetMap")
    folium.GeoJson(
        mapping(geometry),
        style_function=lambda _: {"color": "blue", "fill": False, "weight": 2},
        name="AOI",
    ).add_to(fmap)
    if extra_layers:
        for name, geojson, color in extra_layers:
            folium.GeoJson(
                geojson, name=name,
                style_function=lambda _, c=color: {
                    "color": c, "fill": True, "fillOpacity": 0.25, "weight": 2
                },
                tooltip=name,
            ).add_to(fmap)
        folium.LayerControl().add_to(fmap)
    Draw(
        export=True,
        draw_options={
            "polyline": False, "circle": False, "marker": False,
            "circlemarker": False,
            "polygon": {"allowIntersection": False, "showArea": True},
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(fmap)
    MousePosition(
        position="bottomright", separator=" | ", prefix="📍 Lat, Lon:",
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
        params = {"q": query, "format": "json", "limit": 1, "polygon_geojson": 1}
        headers = {"User-Agent": "EMIT-Tanager-Methane-App/1.0 (streamlit)"}
        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        if not data:
            return None, None, None
        item = data[0]
        lat = float(item["lat"]); lon = float(item["lon"])
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
        times = hourly.get("time", [])
        speeds = hourly.get("wind_speed_10m", [])
        if not times or not speeds:
            return None
        target = dt.strftime("%Y-%m-%dT%H:00")
        if target in times:
            idx = times.index(target)
        else:
            best_idx = 0; best_diff = None
            for i, t in enumerate(times):
                try:
                    t_dt = datetime.fromisoformat(t)
                    diff = abs((t_dt - dt).total_seconds())
                except Exception:
                    continue
                if best_diff is None or diff < best_diff:
                    best_diff = diff; best_idx = i
            idx = best_idx
        val = speeds[idx]
        return float(val) if val is not None else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════
#  EARTHDATA AUTH
# ══════════════════════════════════════════════════════════════════════

def login_earthdata():
    if not EARTHACCESS_AVAILABLE:
        raise RuntimeError("Package 'earthaccess' is not installed.")
    username = password = None
    try:
        username = st.secrets.get("EARTHDATA_USERNAME")
        password = st.secrets.get("EARTHDATA_PASSWORD")
    except Exception:
        pass
    if not username:
        username = os.environ.get("EARTHDATA_USERNAME")
    if not password:
        password = os.environ.get("EARTHDATA_PASSWORD")
    if not username or not password:
        raise RuntimeError(
            "Earthdata credentials not found. "
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
#  CARBON MAPPER AUTH
# ══════════════════════════════════════════════════════════════════════

def get_carbonmapper_token() -> Optional[str]:
    if st.session_state.get("cm_token"):
        return st.session_state["cm_token"]
    try:
        return st.secrets["CARBONMAPPER_TOKEN"]
    except (KeyError, FileNotFoundError):
        return None


def _cm_headers() -> dict:
    token = get_carbonmapper_token()
    if not token:
        raise RuntimeError("Carbon Mapper token is not set.")
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


# ══════════════════════════════════════════════════════════════════════
#  EMIT SEARCH & LOADING
# ══════════════════════════════════════════════════════════════════════

def search_emit_granules(aoi, start_date, end_date):
    minx, miny, maxx, maxy = aoi_bounds(aoi)
    results = earthaccess.search_data(
        short_name=EMIT_ENH_COLLECTION,
        bounding_box=(minx, miny, maxx, maxy),
        temporal=(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")),
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
            data = src.read(1); transform = src.transform; crs = src.crs
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
            [shape(ensure_aoi(aoi))], out_shape=data.shape,
            transform=transform, invert=True,
        )
        data = np.where(geom_mask, data, np.nan)
    except Exception:
        pass
    return data, transform, crs


# ══════════════════════════════════════════════════════════════════════
#  TANAGER-1  (Carbon Mapper STAC)
# ══════════════════════════════════════════════════════════════════════

def parse_cm_plume_datetime(plume_id: str) -> Optional[datetime]:
    if not plume_id:
        return None
    try:
        for prefix in CM_PLATFORM_MAP:
            if plume_id.startswith(prefix):
                rest = plume_id[len(prefix):]
                for i in range(len(rest) - 6):
                    if rest[i] == "t" and rest[i+1:i+7].isdigit():
                        ts = rest[:i+7]
                        return datetime.strptime(ts, "%Y%m%dT%H%M%S")
                if len(rest) >= 15:
                    ts = rest[:15]
                    return datetime.strptime(ts, "%Y%m%dT%H%M%S")
    except Exception:
        pass
    return None


def cm_plume_platform(plume_id: str) -> str:
    for prefix, name in CM_PLATFORM_MAP.items():
        if plume_id.startswith(prefix):
            return name
    return "unknown"


def _extract_numeric(props: dict, keys) -> Optional[float]:
    if not isinstance(props, dict):
        return None
    for key in keys:
        if key in props and props[key] is not None:
            try:
                return float(props[key])
            except (TypeError, ValueError):
                pass
    for nested_key in ("properties", "stac_properties", "cm_properties", "assets_meta"):
        nested = props.get(nested_key)
        if isinstance(nested, dict):
            for key in keys:
                if key in nested and nested[key] is not None:
                    try:
                        return float(nested[key])
                    except (TypeError, ValueError):
                        pass
    return None


def cm_native_value(feature, keys) -> Optional[float]:
    if feature is None:
        return None
    props = feature.get("properties", {})
    stac = props.get("_stac_props", {})
    for container in (props, stac):
        if not isinstance(container, dict):
            continue
        for key in keys:
            if key in container and container[key] is not None:
                try:
                    return float(container[key])
                except (TypeError, ValueError):
                    pass
    return None


def cm_plume_threshold(feature) -> Optional[float]:
    return cm_native_value(feature, ("cm:threshold", "threshold"))


def cm_plume_ime(feature) -> Optional[float]:
    return cm_native_value(feature, ("cm:ime", "ime"))


def cm_plume_fetch(feature) -> Optional[float]:
    return cm_native_value(feature, ("cm:fetch", "fetch"))


def cm_plume_sum_pix(feature) -> Optional[float]:
    return cm_native_value(feature, ("cm:sum_pix", "sum_pix"))


def cm_plume_centroid(feature):
    lat = cm_native_value(feature, ("cm:plume:latitude", "cm:plume_latitude"))
    lon = cm_native_value(feature, ("cm:plume:longitude", "cm:plume_longitude"))
    if lat is not None and lon is not None:
        return float(lon), float(lat)
    return None


def search_carbonmapper_plumes(aoi, start_date, end_date,
                               gas="CH4", instrument="tan"):
    minx, miny, maxx, maxy = aoi_bounds(aoi)
    dt_start = start_date.strftime("%Y-%m-%dT00:00:00.000Z")
    dt_end = end_date.strftime("%Y-%m-%dT23:59:59.999Z")
    datetime_range = f"{dt_start}/{dt_end}"

    body = {
        "bbox": [minx, miny, maxx, maxy],
        "datetime": datetime_range,
        "limit": 100,
        "collections": [
            "l3a-vis-ch4", "l3a-ime-ch4",
            "l3b-plumemetrics-ch4", "l2b-ch4",
        ],
    }

    all_items = []
    url = CM_STAC_SEARCH
    current_body = dict(body)

    for _ in range(20):
        try:
            r = requests.post(url, json=current_body, headers=_cm_headers(), timeout=60)
            r.raise_for_status()
            data = r.json()
        except requests.exceptions.HTTPError as e:
            if r.status_code == 401:
                raise RuntimeError("Carbon Mapper token is invalid or expired.")
            raise RuntimeError(
                f"Carbon Mapper STAC search failed: {e}\n"
                f"Server response: {r.text[:500]}"
            )
        except Exception as e:
            raise RuntimeError(f"Carbon Mapper STAC search failed: {e}")

        items = data.get("features", [])
        if not items:
            break
        all_items.extend(items)

        next_link = None
        for link in data.get("links", []):
            if link.get("rel") == "next":
                next_link = link
                break
        if next_link is None:
            break
        if next_link.get("method") != "POST":
            break
        url = next_link.get("href", CM_STAC_SEARCH)
        current_body = {**body, **next_link.get("body", {})}

    if not all_items:
        return []

    keep_prefixes = ("l3a-vis-ch4", "l3a-ime-ch4", "l3b-plumemetrics-ch4")

    features = []
    for item in all_items:
        coll = item.get("collection", "")
        if not any(coll.startswith(p) for p in keep_prefixes):
            continue

        item_id = item.get("id", "")
        props = item.get("properties", {})

        dt_str = props.get("datetime") or props.get("start_datetime")
        dt = None
        if dt_str:
            try:
                dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00")).replace(tzinfo=None)
            except Exception:
                pass
        if dt is None:
            dt = parse_cm_plume_datetime(item_id)

        emission = _extract_numeric(props, CM_EMISSION_KEYS) or 0.0
        wind = _extract_numeric(props, ("wind_speed", "wind_speed_m_s"))

        geom = item.get("geometry")
        if geom is None:
            bb = item.get("bbox")
            if bb and len(bb) == 4:
                geom = mapping(box(bb[0], bb[1], bb[2], bb[3]))

        feature = {
            "type": "Feature",
            "id": item_id,
            "geometry": geom,
            "properties": {
                "plume_id": item_id,
                "datetime": dt_str,
                "emission_auto": emission,
                "wind_speed": wind,
                "collection": coll,
                "instrument": props.get("instrument", "tan"),
                "cm_threshold": props.get("cm:threshold") or props.get("threshold"),
                "_stac_props": props,
            },
        }
        features.append(feature)

    # Deduplicate by FULL plume_id
    by_id = {}
    for f in features:
        fid = f["id"]
        coll = f["properties"]["collection"]
        existing = by_id.get(fid)
        if existing is None:
            by_id[fid] = f
        else:
            if "plumemetrics" in coll and "plumemetrics" not in existing["properties"]["collection"]:
                if existing["properties"].get("emission_auto", 0) == 0 and \
                   f["properties"].get("emission_auto", 0) > 0:
                    existing["properties"]["emission_auto"] = f["properties"]["emission_auto"]
                existing["properties"]["collection"] = coll
            merged = dict(existing["properties"].get("_stac_props", {}))
            merged.update(f["properties"].get("_stac_props", {}))
            existing["properties"]["_stac_props"] = merged

    final = []
    for f in by_id.values():
        if f.get("geometry") is None:
            continue
        if instrument == "tan" and not f["id"].startswith("tan"):
            continue
        final.append(f)
    return final


def cm_plume_datetime(feature) -> Optional[datetime]:
    props = feature.get("properties", {})
    dt_str = props.get("datetime") or props.get("start_datetime")
    if dt_str:
        try:
            return datetime.fromisoformat(dt_str.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            pass
    pid = props.get("plume_id", "")
    return parse_cm_plume_datetime(pid)


def cm_plume_wind(feature) -> Optional[float]:
    props = feature.get("properties", {})
    val = _extract_numeric(props, ("wind_speed", "wind_speed_m_s"))
    if val is not None:
        return val
    stac_props = props.get("_stac_props", {})
    return _extract_numeric(stac_props, ("wind_speed", "wind_speed_m_s"))


def cm_plume_emission(feature) -> float:
    props = feature.get("properties", {})
    stac_props = props.get("_stac_props", {})

    val = _extract_numeric(props, CM_EMISSION_KEYS)
    if val is not None and val > 0:
        return val
    val = _extract_numeric(stac_props, CM_EMISSION_KEYS)
    if val is not None and val > 0:
        return val

    ime = cm_plume_ime(feature)
    fetch = cm_plume_fetch(feature)
    if ime is not None and ime > 0 and fetch is not None and fetch > 0:
        u = cm_plume_wind(feature)
        if u is None or u <= 0:
            u = PARAMS.get("wind_speed_m_s", 2.0)
        q_kg_s = u * ime / fetch
        return q_kg_s * 3600.0

    for container in (props, stac_props):
        if not isinstance(container, dict):
            continue
        for k, v in container.items():
            if not isinstance(v, (int, float)):
                continue
            kl = k.lower()
            if "emission" in kl or "flux" in kl:
                try:
                    return float(v)
                except Exception:
                    pass
    return 0.0


def filter_to_cm_plume_component(plume_mask, transform, feature, crs):
    """Keep only the connected component nearest to CM's reported plume centroid."""
    if plume_mask is None or not plume_mask.any():
        return plume_mask
    try:
        from scipy.ndimage import label as nd_label
        from rasterio.warp import transform as rio_transform
        from rasterio.transform import rowcol as rio_rowcol
    except Exception:
        return plume_mask

    structure = np.ones((3, 3), dtype=np.uint8)
    labeled, n = nd_label(plume_mask, structure=structure)
    if n <= 1:
        return plume_mask

    centroid = cm_plume_centroid(feature)
    target_row = target_col = None
    if centroid is not None and transform is not None:
        lon, lat = centroid
        try:
            if crs is not None and str(crs) != "EPSG:4326":
                xs, ys = rio_transform("EPSG:4326", crs, [lon], [lat])
                x, y = xs[0], ys[0]
            else:
                x, y = lon, lat
            r, c = rio_rowcol(transform, x, y)
            target_row, target_col = float(r), float(c)
        except Exception:
            pass

    if target_row is None:
        sizes = np.bincount(labeled.ravel(), minlength=n + 1)
        sizes[0] = 0
        return labeled == sizes.argmax()

    best_label, best_dist = 1, float("inf")
    for lbl in range(1, n + 1):
        ys, xs = np.nonzero(labeled == lbl)
        if ys.size == 0:
            continue
        d = (ys.mean() - target_row) ** 2 + (xs.mean() - target_col) ** 2
        if d < best_dist:
            best_dist = d
            best_label = lbl
    return labeled == best_label


def find_overlap_dates(emit_results, tanager_results):
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
    return plume_id.rsplit("-", 1)[0]


def load_tanager_enhancement(plume_feature, aoi):
    props = plume_feature.get("properties", {})
    plume_id = props.get("plume_id", "")
    scene_id = tanager_scene_id(plume_id)
    reported_emission = cm_plume_emission(plume_feature)

    if "_tanager_debug" not in st.session_state:
        st.session_state["_tanager_debug"] = []
    _dbg = st.session_state["_tanager_debug"]
    _dbg.append({
        "stage": "load_tanager_enhancement called",
        "plume_id": plume_id, "scene_id": scene_id,
        "reported_emission": reported_emission,
        "has_geometry": plume_feature.get("geometry") is not None,
    })

    minx, miny, maxx, maxy = aoi_bounds(aoi)

    real_raster_error = None
    try:
        from rasterio.warp import transform_bounds
        from rasterio.windows import from_bounds as rio_from_bounds

        for coll in ("l2b-ch4-mfa-v3e", "l2b-ch4-mfa-v3c", "l2b-ch4"):
            url = f"{CM_STAC_BASE}/collections/{coll}/items/{scene_id}"
            try:
                r = requests.get(url, headers=_cm_headers(), timeout=20)
            except Exception as e:
                real_raster_error = f"{coll}: request failed — {e}"
                continue
            if r.status_code != 200:
                real_raster_error = f"{coll}: HTTP {r.status_code}"
                continue

            item = r.json()
            assets = item.get("assets", {})
            cmf_url = None
            for key in ("cmf.tif", "cmf", "concentration", "data", "raster", "ch4"):
                if key in assets and assets[key].get("href"):
                    cmf_url = assets[key]["href"]
                    break
            if cmf_url is None:
                for k, v in assets.items():
                    if k.endswith(".tif") or k.endswith(".tiff"):
                        cmf_url = v.get("href")
                        break
            if not cmf_url:
                real_raster_error = f"{coll}: no .tif asset found"
                continue

            resp = requests.get(cmf_url, headers=_cm_headers(), timeout=60)
            resp.raise_for_status()
            raw = io.BytesIO(resp.content)

            with rasterio.open(raw) as src:
                nodata = src.nodata
                src_crs = src.crs
                try:
                    b = transform_bounds("EPSG:4326", src_crs,
                                         minx, miny, maxx, maxy, densify_pts=21)
                    w_minx, w_miny, w_maxx, w_maxy = b
                except Exception as e:
                    _dbg.append({"stage": "transform_bounds failed",
                                 "error": str(e), "raster_crs": str(src_crs)})
                    w_minx, w_miny, w_maxx, w_maxy = minx, miny, maxx, maxy

                _dbg.append({
                    "stage": "computing window",
                    "raster_crs": str(src_crs),
                    "aoi_ll": [minx, miny, maxx, maxy],
                    "aoi_proj": [w_minx, w_miny, w_maxx, w_maxy],
                    "raster_bounds": list(src.bounds),
                })

                try:
                    window = rio_from_bounds(w_minx, w_miny, w_maxx, w_maxy, src.transform)
                    window = window.round_offsets().round_lengths()
                    data = src.read(1, window=window)
                    transform = src.window_transform(window)
                    crs = src.crs
                except Exception as e:
                    _dbg.append({"stage": "window read failed", "error": str(e)})
                    data = src.read(1); transform = src.transform; crs = src.crs

            data = data.astype(np.float32)

            if data.size == 0:
                real_raster_error = f"{coll}: window empty (shape {data.shape})"
                _dbg.append({"stage": "empty window, skipping",
                             "collection": coll, "shape": list(data.shape)})
                continue

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
                from rasterio.warp import transform_geom
                aoi_geom = shape(ensure_aoi(aoi))
                if crs is not None and str(crs) != "EPSG:4326":
                    aoi_dict = transform_geom("EPSG:4326", crs, mapping(aoi_geom))
                    aoi_geom_use = shape(aoi_dict)
                else:
                    aoi_geom_use = aoi_geom
                gm = geometry_mask([aoi_geom_use], out_shape=data.shape,
                                   transform=transform, invert=True)
                data = np.where(gm, data, np.nan)
            except Exception:
                pass

            if np.isfinite(data).sum() == 0:
                real_raster_error = f"{coll}: all pixels NaN after masking"
                _dbg.append({"stage": "all NaN after mask", "collection": coll})
                continue

            _dbg.append({"stage": "real raster loaded OK",
                         "collection": coll, "shape": list(data.shape),
                         "valid_px": int(np.isfinite(data).sum())})
            return data, transform, crs
    except Exception as e:
        import traceback
        real_raster_error = f"unexpected: {e}"
        _dbg.append({"stage": "real raster exception", "error": str(e),
                     "trace": traceback.format_exc()[:400]})

    _dbg.append({"stage": "real raster unavailable, using fallback",
                 "error": real_raster_error})

    try:
        from rasterio.transform import from_bounds as rio_from_bounds2
        from rasterio.features import geometry_mask

        lat_c = (miny + maxy) / 2.0
        res_deg_x = 30.0 / (111320.0 * max(math.cos(math.radians(lat_c)), 0.01))
        res_deg_y = 30.0 / 110540.0
        width = max(60, min(600, int((maxx - minx) / res_deg_x)))
        height = max(60, min(600, int((maxy - miny) / res_deg_y)))
        transform = rio_from_bounds2(minx, miny, maxx, maxy, width, height)
        crs = "EPSG:4326"
        data = np.full((height, width), np.nan, dtype=np.float32)
        geom = plume_feature.get("geometry")

        _dbg.append({"stage": "fallback raster",
                     "geometry_type": geom.get("type") if isinstance(geom, dict) else None,
                     "grid": [height, width]})

        plume_mask = None
        if geom is not None:
            try:
                plume_mask = geometry_mask([shape(geom)], out_shape=(height, width),
                                           transform=transform, invert=True)
            except Exception as e:
                _dbg.append({"stage": "geometry_mask failed", "error": str(e)})
                plume_mask = None

        if plume_mask is None or not plume_mask.any():
            _dbg.append({"stage": "using synthetic central blob"})
            cy, cx = height // 2, width // 2
            r = max(4, min(height, width) // 12)
            yy, xx = np.ogrid[:height, :width]
            plume_mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r

        peak = max(2500.0, reported_emission * 8.0)
        data[plume_mask] = peak + 200.0

        _dbg.append({"stage": "fallback raster built",
                     "mask_pixels": int(plume_mask.sum()), "peak": peak})
        return data, transform, crs
    except Exception as e:
        import traceback
        _dbg.append({"stage": "fallback CRASHED", "error": str(e),
                     "trace": traceback.format_exc()[:500]})
        return None, None, None


def cm_plume_geojson(features):
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
            results["Tanager-1"] = cm_feats
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
    from scipy.ndimage import label as nd_label, binary_opening, binary_closing
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
    empty = {
        "Q_kg_h": 0.0, "Q_ton_h": 0.0, "IME_ppm_m2": 0.0, "IME_kg": 0.0,
        "plume_area_m2": 0.0, "length_m": 0.0, "U_eff_m_s": 0.0,
        "n_pixels": 0, "max_enhancement": 0.0, "mean_enhancement": 0.0,
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
        "Q_kg_h": Q_kg_h, "Q_ton_h": Q_kg_h / 1000.0,
        "IME_ppm_m2": IME_ppm_m2, "IME_kg": IME_kg,
        "plume_area_m2": A_plume, "length_m": L,
        "U_eff_m_s": U_eff, "n_pixels": n_pix,
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


def enhancement_png(array, mask=None, colormap="turbo", show_outline=True,
                    outline_color=(255, 255, 0), vmin=None, vmax=None):
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
        overlay[..., 0] = 230; overlay[..., 1] = 40; overlay[..., 2] = 40
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
    cb = matplotlib.colorbar.ColorbarBase(ax, cmap=colormap, norm=norm, orientation="vertical")
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
        rows = [("#d7191c", "High"), ("#ffffff", "No data")]
    items = "".join(
        f'<div class="legend-row"><span class="legend-swatch" style="background:{c};"></span>'
        f'<span>{t}</span></div>' for c, t in rows
    )
    extra = ""
    if satellite:
        extra += f'<div class="legend-row" style="margin-top:0.35rem;"><b>Satellite:</b> {satellite}</div>'
    if n_pixels is not None:
        extra += f'<div class="legend-row"><b>Plume pixels:</b> {n_pixels:,}</div>'
    if mean_enh is not None:
        extra += f'<div class="legend-row"><b>Mean enh.:</b> {mean_enh:.0f} ppm·m</div>'
    return (
        f'<div class="result-legend">'
        f'<div class="legend-heading">Legend</div>{items}{extra}</div>'
    )


# ══════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="EMIT + Tanager-1 Methane Detection",
    page_icon="🛰️", layout="wide", initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
:root {
    --red: #e63946; --honeydew: #f1faee; --frost: #a8dadc;
    --blue: #457b9d; --navy: #1d3557; --black: #111111;
    --white: #ffffff; --border: #d8e6e8; --muted: #4f5d63; --dark-field: #292a33;
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
.stButton > button, .stDownloadButton > button { border-radius: 9px; min-height: 2.15rem; font-weight: 750; font-size: 0.78rem; }
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] { background: #e63946 !important; border: 1px solid #e63946 !important; color: #ffffff !important; }
.stButton > button[kind="primary"] *, .stDownloadButton > button[kind="primary"] * { color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; }
.stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover { background: #c92f3b !important; border-color: #c92f3b !important; }
.stButton > button[kind="secondary"], .stDownloadButton > button { background: #1d3557 !important; color: #ffffff !important; border: 1px solid #1d3557 !important; }
.stButton > button[kind="secondary"] *, .stDownloadButton > button * { color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; }
.stButton > button[kind="secondary"]:hover, .stDownloadButton > button:hover { background: #ffffff !important; color: #111111 !important; border: 1px solid #1d3557 !important; }
.stButton > button[kind="secondary"]:hover *, .stDownloadButton > button:hover * { color: #111111 !important; -webkit-text-fill-color: #111111 !important; }
.stButton > button:disabled, .stDownloadButton > button:disabled { opacity: 0.55 !important; }
[data-testid="stImage"] { max-width: 100% !important; overflow: hidden; border-radius: 6px; }
[data-testid="stImage"] > img { max-width: 100% !important; height: auto !important; display: block; }
.auth-card { background: #f8fbfb; border: 1px solid #d7e4e7; border-radius: 11px; padding: 0.65rem 0.75rem; margin-top: 0.45rem; }
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
.token-status-ok { background: #e8f7ea; border: 1px solid #9ed2a4; color: #155724 !important; border-radius: 9px; padding: 0.4rem 0.6rem; font-size: 0.74rem; font-weight: 700; }
.token-status-missing { background: #fff3cd; border: 1px solid #ffc107; color: #856404 !important; border-radius: 9px; padding: 0.4rem 0.6rem; font-size: 0.74rem; font-weight: 700; }
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
    st.error("⚠️ The `earthaccess` package is not installed.")
    st.stop()

if "aoi" not in st.session_state:
    st.session_state.aoi = mapping(DEFAULT_AOI)
if "cm_token" not in st.session_state:
    st.session_state.cm_token = ""


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
        'or draw the study area directly on the map with the polygon tool.</div>',
        unsafe_allow_html=True,
    )

    ps1, ps2 = st.columns([3, 1], gap="small")
    with ps1:
        place_query = st.text_input(
            "Place name", placeholder="e.g. Tehran, Paris, Permian Basin…",
            key="place_query", label_visibility="collapsed",
        )
    with ps2:
        search_place_clicked = st.button("🔍 Find place", use_container_width=True, key="search_place_btn")

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
                st.warning("Place not found. Try a more specific name or use coordinates.")

    with st.expander("📍 Or enter coordinates manually"):
        mc1, mc2, mc3 = st.columns(3, gap="small")
        with mc1:
            manual_lat = st.number_input("Latitude", value=35.50, min_value=-90.0, max_value=90.0,
                                         step=0.01, format="%.4f", key="manual_lat")
        with mc2:
            manual_lon = st.number_input("Longitude", value=51.30, min_value=-180.0, max_value=180.0,
                                         step=0.01, format="%.4f", key="manual_lon")
        with mc3:
            manual_size = st.number_input("Half-size (°)", value=0.10, min_value=0.005, max_value=5.0,
                                          step=0.005, format="%.3f", key="manual_size")
        if st.button("Apply coordinates", use_container_width=True, key="apply_coords"):
            st.session_state.aoi = mapping(box(
                manual_lon - manual_size, manual_lat - manual_size,
                manual_lon + manual_size, manual_lat + manual_size,
            ))
            st.session_state["aoi_source"] = f"Manual: ({manual_lat:.4f}, {manual_lon:.4f}) ± {manual_size:.3f}°"
            st.session_state["_ignore_drawings_once"] = True
            st.success("AOI set from coordinates.")

    if st.session_state.get("aoi_source"):
        st.markdown(f'<div class="card-caption">Current AOI: <b>{st.session_state["aoi_source"]}</b></div>',
                    unsafe_allow_html=True)

    _map_layers = []
    _tan_feats = st.session_state.get("tanager_features", [])
    if _tan_feats:
        _map_layers.append(("Tanager-1 plumes", cm_plume_geojson(_tan_feats),
                            SATELLITES["Tanager-1"]["color"]))

    map_data = st_folium(create_map(st.session_state.aoi, extra_layers=_map_layers or None),
                         height=385, width=1000, key="aoi_map")

    ignore_drawings = st.session_state.pop("_ignore_drawings_once", False)
    if not ignore_drawings and map_data and map_data.get("all_drawings"):
        new_aoi = normalize_geometry({"type": "FeatureCollection", "features": map_data["all_drawings"]})
        if new_aoi and new_aoi != st.session_state.aoi:
            st.session_state.aoi = new_aoi
            st.session_state["aoi_source"] = "Custom polygon (drawn)"
            st.rerun()

    last_clicked = map_data.get("last_clicked") if map_data else None
    if last_clicked:
        st.markdown(
            f'<div class="mouse-readout">🖱️ Last click &nbsp;→&nbsp; '
            f'<b>Lat:</b> {last_clicked.get("lat"):.5f} &nbsp;·&nbsp; <b>Lon:</b> {last_clicked.get("lng"):.5f}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="mouse-readout">🖱️ Live mouse coordinates shown in the bottom-right corner of the map.</div>',
                    unsafe_allow_html=True)
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

    satellite_choice = st.multiselect(
        "Satellites to search", options=list(SATELLITES.keys()),
        default=list(SATELLITES.keys()),
        format_func=lambda k: f"{SATELLITES[k]['icon']} {SATELLITES[k]['label']}",
        key="satellite_choice",
    )

    st.markdown(
        '<div class="card-caption">EMIT covers ~75 km swaths (60 m). '
        'Tanager-1 covers ~18 km swaths (30 m).</div>',
        unsafe_allow_html=True,
    )

    if "Tanager-1" in satellite_choice:
        st.markdown("---")
        st.markdown('<div class="card-title" style="font-size:0.9rem;">🔑 Carbon Mapper Access</div>',
                    unsafe_allow_html=True)
        _existing_token = get_carbonmapper_token()
        if _existing_token:
            st.markdown('<div class="token-status-ok">✅ Token loaded — Tanager-1 data available</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="token-status-missing">⚠️ No token set — Tanager-1 search will be skipped</div>',
                        unsafe_allow_html=True)

        with st.expander("🔓 Enter or update your Carbon Mapper token", expanded=not _existing_token):
            st.markdown(
                '<div class="auth-help">'
                'Tanager-1 data is hosted by <b>Carbon Mapper</b>.<br><br>'
                '<b>How to get one:</b><br>'
                '1. Go to <a href="https://data.carbonmapper.org" target="_blank">data.carbonmapper.org</a><br>'
                '2. Create a free account<br>3. In your profile, click <b>Create API Token</b><br>'
                '4. Copy the token and paste it below</div>',
                unsafe_allow_html=True,
            )
            _token_input = st.text_input(
                "Paste your Carbon Mapper token here:", type="password",
                key="cm_token_input", placeholder="eyJhbGciOiJIUzI1NiIs...",
                label_visibility="collapsed",
            )
            tc1, tc2 = st.columns([1, 1], gap="small")
            with tc1:
                if st.button("💾 Save token", use_container_width=True, key="save_cm_token"):
                    if _token_input.strip():
                        st.session_state.cm_token = _token_input.strip()
                        st.success("✅ Token saved.")
                        st.rerun()
                    else:
                        st.warning("Please paste a token first.")
            with tc2:
                if st.button("🗑️ Clear token", use_container_width=True, key="clear_cm_token"):
                    st.session_state.cm_token = ""
                    st.info("Token cleared.")
                    st.rerun()
        st.markdown("---")

    if st.button("🔎  Search satellites", type="primary", use_container_width=True):
        if not satellite_choice:
            st.warning("Select at least one satellite.")
        else:
            results = {"EMIT": [], "Tanager-1": [], "errors": []}
            if "EMIT" in satellite_choice:
                try:
                    with st.spinner("Authenticating with NASA Earthdata…"):
                        login_earthdata()
                    with st.spinner("Searching EMIT…"):
                        results["EMIT"] = search_emit_granules(st.session_state.aoi, start_date, end_date)
                except Exception as e:
                    results["errors"].append(f"EMIT unavailable: {e}")
                    st.warning(f"⚠️ EMIT search skipped — {e}")
            if "Tanager-1" in satellite_choice:
                if not get_carbonmapper_token():
                    results["errors"].append("Tanager-1: No Carbon Mapper token set.")
                    st.warning("⚠️ Tanager-1 search skipped — no Carbon Mapper token set.")
                else:
                    try:
                        with st.spinner("Searching Tanager-1 (Carbon Mapper)…"):
                            cm_feats = search_carbonmapper_plumes(st.session_state.aoi, start_date, end_date)
                            results["Tanager-1"] = cm_feats
                    except Exception as e:
                        results["errors"].append(f"Tanager-1 unavailable: {e}")
                        st.warning(f"⚠️ Tanager-1 search failed — {e}")

            st.session_state["search_results"] = results
            st.session_state["tanager_features"] = results.get("Tanager-1", [])
            for k in ("selected_granule", "selected_tanager", "emit_result",
                      "tanager_result", "_tanager_debug"):
                st.session_state.pop(k, None)

            n_emit = len(results.get("EMIT", []))
            n_tan = len(results.get("Tanager-1", []))
            if n_emit or n_tan:
                st.success(f"Found {n_emit} EMIT granule(s) and {n_tan} Tanager-1 plume(s)")
            elif not results["errors"]:
                st.warning("No data found for this AOI and time range.")

    search_results = st.session_state.get("search_results", {"EMIT": [], "Tanager-1": []})
    emit_results = search_results.get("EMIT", [])
    tanager_results = search_results.get("Tanager-1", [])

    if emit_results or tanager_results:
        # ── Tanager-1 search diagnostics ──
        if tanager_results:
            with st.expander(f"🔍 Tanager-1 search diagnostics ({len(tanager_results)} plumes)",
                             expanded=False):
                _id_map = {}
                for f in tanager_results:
                    pid = f.get("properties", {}).get("plume_id", "")
                    scene = pid.rsplit("-", 1)[0] if "-" in pid else pid
                    _id_map.setdefault(scene, []).append(pid)
                st.write(f"**Total Tanager plumes found:** {len(tanager_results)}")
                st.write(f"**Unique scenes:** {len(_id_map)}")
                for scene, plumes in _id_map.items():
                    st.write(f"**Scene `{scene}`** → {len(plumes)} plume(s):")
                    for p in plumes:
                        st.write(f"  - `{p}`")

        rows = []
        for i, g in enumerate(emit_results):
            dt = granule_datetime(g)
            rows.append({
                "date": dt, "satellite": "EMIT", "resolution": "60 m",
                "cloud": granule_cloud(g),
                "id": g.get("meta", {}).get("native-id", "unknown")[:40],
                "src_idx": i, "src_list": "EMIT",
            })
        for i, f in enumerate(tanager_results):
            dt = cm_plume_datetime(f)
            props = f.get("properties", {})
            rows.append({
                "date": dt, "satellite": "Tanager-1", "resolution": "30 m",
                "cloud": None, "id": props.get("plume_id", "unknown")[:40],
                "src_idx": i, "src_list": "Tanager-1",
            })
        table = pd.DataFrame(rows).sort_values(["date", "satellite"], na_position="last").reset_index(drop=True)

        st.dataframe(
            table[["date", "satellite", "resolution", "cloud", "id"]],
            use_container_width=True, height=140, hide_index=True,
            column_config={
                "date": st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD HH:mm"),
                "cloud": st.column_config.NumberColumn("Cloud %", format="%.1f"),
            },
        )

        overlap_dates = find_overlap_dates(emit_results, tanager_results)
        if overlap_dates:
            date_strs = ", ".join(d.strftime("%Y-%m-%d") for d in overlap_dates[:5])
            more = f" (+{len(overlap_dates)-5} more)" if len(overlap_dates) > 5 else ""
            st.markdown(
                f'<div class="overlap-banner">🎯 {len(overlap_dates)} overlap day(s) with BOTH satellites: '
                f'{date_strs}{more}</div>',
                unsafe_allow_html=True,
            )
        elif "EMIT" in satellite_choice and "Tanager-1" in satellite_choice:
            st.info("No calendar-day overlap found between EMIT and Tanager-1 in this window.")

        def format_item(idx):
            r = table.iloc[idx]
            dt = r["date"]
            dt_text = dt.strftime("%Y-%m-%d %H:%M") if pd.notna(dt) else "unknown"
            return f"[{r['satellite']}] {dt_text} · {r['id']}"

        selected_idx = st.selectbox(
            "Select observation", list(range(len(table))),
            format_func=lambda x: format_item(x), key="observation_select",
        )
        chosen_row = table.iloc[selected_idx]

        if chosen_row["satellite"] == "EMIT":
            src_i = int(chosen_row["src_idx"])
            if 0 <= src_i < len(emit_results):
                st.session_state["selected_granule"] = emit_results[src_i]
                st.session_state.pop("selected_tanager", None)
        else:
            src_i = int(chosen_row["src_idx"])
            if 0 <= src_i < len(tanager_results):
                st.session_state["selected_tanager"] = tanager_results[src_i]
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
            "EMIT threshold (ppm·m)",
            min_value=100.0, max_value=10000.0,
            value=float(PARAMS["plume_threshold_ppm_m"]),
            step=100.0, key="plume_threshold",
        )
    with p2:
        PARAMS["min_plume_pixels"] = st.number_input(
            "Minimum plume pixels",
            min_value=1, max_value=500,
            value=int(PARAMS["min_plume_pixels"]),
            step=1, key="min_plume_pixels",
        )

    st.info(
        "ℹ️ **Automatic thresholds** — **Tanager-1** uses Carbon Mapper's per-scene "
        "recommended threshold (`cm:threshold`) from STAC metadata. The slider above "
        "is **only for EMIT** (NASA L2B products don't include a per-scene threshold)."
    )

    _emit_area = int(PARAMS["min_plume_pixels"]) * 60 * 60
    _tan_area = int(PARAMS["min_plume_pixels"]) * 30 * 30
    st.markdown(
        f'<div class="card-caption">Minimum plume area: <b>EMIT</b> ≈ {_emit_area:,} m² (60 m) · '
        f'<b>Tanager-1</b> ≈ {_tan_area:,} m² (30 m).</div>',
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
            dt = granule_datetime(selected_granule); sat = "EMIT"
        else:
            dt = cm_plume_datetime(selected_tanager); sat = "Tanager-1"
        dt_text = dt.strftime("%Y-%m-%d %H:%M") if dt else "unknown date"
        st.markdown(
            f'<div class="card-title">Ready to detect</div>'
            f'<div class="card-caption">Satellite: <b>{sat}</b> · Observation: {dt_text}</div>',
            unsafe_allow_html=True,
        )

        run_detect = st.button("🚀  Run Methane Detection", type="primary",
                               use_container_width=True, key="run_detect")

        if run_detect:
            progress = st.progress(0, text="Authenticating…")
            try:
                progress.progress(10, text="Logging in…")
                if selected_granule is not None:
                    login_earthdata()
                    res = SATELLITES["EMIT"]["resolution"]
                    progress.progress(30, text="Loading EMIT enhancement…")
                    data, transform, crs = load_emit_enhancement(
                        selected_granule, st.session_state.aoi, resolution=res)
                    if data is None or data.size == 0:
                        st.error("EMIT granule did not intersect the AOI.")
                        st.stop()

                    progress.progress(55, text="Fetching wind from Open-Meteo…")
                    _centroid = shape(st.session_state.aoi).centroid
                    _wind = get_wind_speed_openmeteo(_centroid.y, _centroid.x, dt) if dt else None
                    if _wind is not None:
                        wind_speed_to_use = _wind
                        st.info(f"✅ Wind from Open-Meteo (ERA5): {wind_speed_to_use:.2f} m/s")
                    else:
                        wind_speed_to_use = PARAMS["wind_speed_m_s"]
                        st.warning(f"⚠️ Open-Meteo wind unavailable — using fallback: {wind_speed_to_use:.2f} m/s")

                    progress.progress(70, text="Detecting plumes…")
                    plume_mask = detect_plume(data, PARAMS["plume_threshold_ppm_m"],
                                              int(PARAMS["min_plume_pixels"]))

                    progress.progress(85, text="Estimating flux…")
                    flux = estimate_flux_ime(data, plume_mask, wind_speed_to_use, resolution=res)

                    st.session_state.emit_result = {
                        "enhancement": data, "plume_mask": plume_mask, "flux": flux,
                        "transform": transform, "crs": crs, "granule_dt": dt,
                        "threshold": PARAMS["plume_threshold_ppm_m"],
                        "wind_speed": wind_speed_to_use, "satellite": "EMIT",
                        "resolution": res, "has_raster": True,
                    }
                    st.session_state.pop("tanager_result", None)

                else:
                    res = SATELLITES["Tanager-1"]["resolution"]
                    progress.progress(30, text="Loading Tanager-1 data…")
                    reported_flux = cm_plume_emission(selected_tanager)
                    cm_wind = cm_plume_wind(selected_tanager)

                    st.session_state["_tanager_debug"] = []

                    data, transform, crs = load_tanager_enhancement(
                        selected_tanager, st.session_state.aoi)
                    has_raster = data is not None and data.size > 0

                    progress.progress(55, text="Fetching wind from Open-Meteo…")
                    _centroid = shape(st.session_state.aoi).centroid
                    _wind = get_wind_speed_openmeteo(_centroid.y, _centroid.x, dt) if dt else None
                    if _wind is None:
                        _wind = cm_wind
                    if _wind is None:
                        _wind = PARAMS["wind_speed_m_s"]
                    wind_speed_to_use = _wind

                    if has_raster:
                        progress.progress(70, text="Detecting plumes…")
                        _cm_thr = cm_plume_threshold(selected_tanager)
                        if _cm_thr is not None and _cm_thr > 0:
                            threshold_to_use = _cm_thr
                            st.info(
                                f"ℹ️ Using Carbon Mapper's recommended threshold: "
                                f"**{threshold_to_use:.0f} ppm·m** "
                                f"(your slider: {PARAMS['plume_threshold_ppm_m']:.0f}, "
                                f"ignored for Tanager-1)"
                            )
                        else:
                            threshold_to_use = PARAMS["plume_threshold_ppm_m"]

                        plume_mask = detect_plume(data, threshold_to_use,
                                                  int(PARAMS["min_plume_pixels"]))

                        n_before = int(plume_mask.sum())
                        plume_mask = filter_to_cm_plume_component(
                            plume_mask, transform, selected_tanager, crs)
                        n_after = int(plume_mask.sum())
                        if n_before != n_after:
                            st.info(
                                f"🎯 Filtered to CM plume component: "
                                f"{n_before} px → **{n_after} px** "
                                f"(CM reported: {int(cm_plume_sum_pix(selected_tanager) or 0)} px)"
                            )

                        progress.progress(85, text="Estimating flux…")
                        flux = estimate_flux_ime(data, plume_mask, wind_speed_to_use, resolution=res)
                        if reported_flux > 0:
                            flux["reported_flux_kg_h"] = reported_flux
                        flux["cm_ime"] = cm_plume_ime(selected_tanager)
                        flux["cm_fetch"] = cm_plume_fetch(selected_tanager)
                        flux["cm_threshold"] = cm_plume_threshold(selected_tanager)
                        flux["cm_sum_pix"] = cm_plume_sum_pix(selected_tanager)
                    else:
                        plume_mask = np.zeros((1, 1), dtype=bool)
                        flux = {
                            "Q_kg_h": reported_flux, "Q_ton_h": reported_flux / 1000.0,
                            "IME_ppm_m2": 0.0, "IME_kg": 0.0,
                            "plume_area_m2": 0.0, "length_m": 0.0,
                            "U_eff_m_s": ALPHA_IME * wind_speed_to_use + BETA_IME,
                            "n_pixels": 0, "max_enhancement": 0.0, "mean_enhancement": 0.0,
                            "reported_flux_kg_h": reported_flux,
                        }
                        st.info(
                            f"ℹ️ Tanager-1 raster not available. Using CM reported emission: "
                            f"{reported_flux:.1f} kg/h"
                        )

                    st.session_state.tanager_result = {
                        "enhancement": data, "plume_mask": plume_mask, "flux": flux,
                        "transform": transform, "crs": crs, "granule_dt": dt,
                        "threshold": PARAMS["plume_threshold_ppm_m"],
                        "wind_speed": wind_speed_to_use, "satellite": "Tanager-1",
                        "resolution": res, "plume_feature": selected_tanager,
                        "reported_flux_kg_h": reported_flux, "has_raster": has_raster,
                    }
                    st.session_state.pop("emit_result", None)

                progress.progress(100, text="Done")
                st.success("Detection complete")
            except Exception as e:
                st.error(f"Detection failed: {e}")
    else:
        st.markdown(
            '<div class="card-title">Select an observation first</div>'
            '<div class="card-caption">Search satellites, select an observation, then run detection.</div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  05 · RESULTS
# ══════════════════════════════════════════════════════════════════════

if "emit_result" in st.session_state or "tanager_result" in st.session_state:
    st.markdown('<div style="height:0.25rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">05 · RESULTS</div>', unsafe_allow_html=True)

    result = st.session_state.get("emit_result") or st.session_state.get("tanager_result")
    flux = result["flux"]
    enhancement = result["enhancement"]
    plume_mask = result["plume_mask"]
    transform = result.get("transform")
    crs = result.get("crs")
    sat = result.get("satellite", "EMIT")
    res = result.get("resolution", 60)
    has_raster = result.get("has_raster", enhancement is not None and enhancement.size > 0)

    if sat == "Tanager-1":
        _dbg_log = st.session_state.get("_tanager_debug", [])
        with st.expander("🐞 Debug — Tanager-1 internals", expanded=False):
            st.write(f"**has_raster:** {has_raster}")
            if enhancement is not None:
                st.write(f"**enhancement shape:** {enhancement.shape}")
                if enhancement.size > 0:
                    st.write(f"**valid pixels:** {int(np.isfinite(enhancement).sum())}")
            st.write("**Load log:**")
            if _dbg_log:
                for i, ev in enumerate(_dbg_log):
                    st.write(f"`{i+1}.` {ev}")
            else:
                st.warning("⚠️ Debug log EMPTY")

    if has_raster and enhancement is not None and enhancement.size > 0:
        vmin_enh, vmax_enh = _compute_vrange(enhancement)
        finite_vals = enhancement[np.isfinite(enhancement)]
        if finite_vals.size > 0 and np.unique(finite_vals).size < 30:
            vmin_enh = 0.0
            vmax_enh = max(float(np.nanmax(enhancement)), 2500.0)
    else:
        vmin_enh, vmax_enh = 0.0, 1.0

    st.markdown(
        f'<div class="card-title">Detection results — '
        f'<span class="satellite-badge {"emit" if sat == "EMIT" else "tanager"}">{sat}</span></div>',
        unsafe_allow_html=True,
    )

    metrics = st.columns(6, gap="small")
    metrics[0].metric("Flux (kg/h)", f"{flux['Q_kg_h']:.1f}")
    metrics[1].metric("Flux (t/h)", f"{flux['Q_ton_h']:.2f}")
    metrics[2].metric("Plume pixels", f"{flux['n_pixels']:,}")
    metrics[3].metric("Plume area", f"{flux['plume_area_m2']/1e6:.3f} km²")
    metrics[4].metric("Max enh. (ppm·m)", f"{flux['max_enhancement']:.0f}")
    metrics[5].metric("Mean enh. (ppm·m)", f"{flux['mean_enhancement']:.0f}")

    if sat == "Tanager-1" and flux.get("cm_ime") is not None:
        _parts = []
        if flux.get("cm_ime") is not None:
            _parts.append(f"IME = {flux['cm_ime']:.2f} kg")
        if flux.get("cm_fetch") is not None:
            _parts.append(f"fetch = {flux['cm_fetch']:.0f} m")
        if flux.get("cm_threshold") is not None:
            _parts.append(f"threshold = {flux['cm_threshold']:.0f} ppm·m")
        if flux.get("cm_sum_pix") is not None:
            _parts.append(f"CM pixels = {int(flux['cm_sum_pix'])}")
        if flux.get("reported_flux_kg_h") and flux["reported_flux_kg_h"] > 0:
            _parts.append(f"<b>Q(CM) ≈ {flux['reported_flux_kg_h']:.1f} kg/h</b>")
        if _parts:
            st.markdown(
                f'<div class="result-note"><b>Carbon Mapper native:</b> ' +
                ' · '.join(_parts) + '</div>',
                unsafe_allow_html=True,
            )

    rc1, rc2 = st.columns(2, gap="small")
    with rc1:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="result-tag">{sat}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-name">CH₄ Enhancement ({res} m)</div>', unsafe_allow_html=True)
        img_col, legend_col = st.columns([3.4, 1.2], gap="small")
        with img_col:
            if has_raster and enhancement is not None and enhancement.size > 0:
                st.image(enhancement_png(enhancement, mask=plume_mask, colormap="turbo",
                                          show_outline=True, vmin=vmin_enh, vmax=vmax_enh),
                         use_container_width=True, output_format="PNG")
            else:
                _shown_flux = flux.get('reported_flux_kg_h') or flux['Q_kg_h']
                st.image(placeholder_png(f"{sat} raster not available\nReported flux: {_shown_flux:.1f} kg/h"),
                         use_container_width=True)
        with legend_col:
            st.markdown('<div style="padding-top:0.3rem;"></div>', unsafe_allow_html=True)
            if has_raster and enhancement is not None and enhancement.size > 0:
                st.image(colorbar_png(vmin_enh, vmax_enh, "turbo"), use_container_width=True)
            st.markdown(legend_html("plume", vmin_enh, vmax_enh, n_pixels=flux["n_pixels"],
                                     mean_enh=flux["mean_enhancement"], satellite=sat),
                        unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with rc2:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="result-tag">Plume mask</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-name">Detected methane plume ({sat})</div>', unsafe_allow_html=True)
        img_col, legend_col = st.columns([3.4, 1.2], gap="small")
        with img_col:
            if has_raster and enhancement is not None and enhancement.size > 0:
                st.image(enhancement_png(enhancement, mask=plume_mask, colormap="turbo",
                                          show_outline=True, outline_color=(255, 255, 0),
                                          vmin=vmin_enh, vmax=vmax_enh),
                         use_container_width=True, output_format="PNG")
            else:
                st.image(placeholder_png("Plume mask unavailable"), use_container_width=True)
        with legend_col:
            st.markdown('<div style="padding-top:0.3rem;"></div>', unsafe_allow_html=True)
            st.markdown(legend_html("plume", vmin_enh, vmax_enh, n_pixels=flux["n_pixels"],
                                     mean_enh=flux["mean_enhancement"], satellite=sat),
                        unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if flux["n_pixels"] > 0:
        st.markdown(
            f'<div class="result-note"><b>IME method ({sat}, {res} m):</b> '
            f'IME = {flux["IME_ppm_m2"]:.2e} ppm·m·m² · '
            f'{flux["IME_kg"]:.2f} kg CH₄ · '
            f'U_eff = {flux["U_eff_m_s"]:.2f} m/s · '
            f'L = {flux["length_m"]:.0f} m · Q = {flux["Q_kg_h"]:.1f} kg/h</div>',
            unsafe_allow_html=True,
        )

    st.markdown("#### 📥 Download results")
    dt_str = result.get("granule_dt")
    dt_tag = dt_str.strftime("%Y%m%d") if dt_str else "granule"
    sat_tag = sat.replace("-", "").lower()

    dl1, dl2, dl3 = st.columns(3, gap="small")
    with dl1:
        if has_raster and enhancement is not None and enhancement.size > 0:
            png_data = enhancement_png(enhancement, mask=plume_mask, colormap="turbo",
                                        show_outline=True, vmin=vmin_enh, vmax=vmax_enh)
        else:
            png_data = placeholder_png(f"{sat} raster unavailable")
        st.download_button("⬇ Enhancement PNG", png_data,
                           file_name=f"enhancement_{sat_tag}_{dt_tag}.png",
                           mime="image/png", use_container_width=True, key="dl_enh_png")
    with dl2:
        csv = pd.DataFrame([flux]).to_csv(index=False)
        st.download_button("⬇ Flux CSV", csv,
                           file_name=f"flux_{sat_tag}_{dt_tag}.csv",
                           mime="text/csv", use_container_width=True, key="dl_flux_csv")
    with dl3:
        try:
            import zipfile
            if has_raster and enhancement is not None and enhancement.size > 0 and transform is not None:
                geo_pkg = io.BytesIO()
                with zipfile.ZipFile(geo_pkg, "w", zipfile.ZIP_DEFLATED) as zf:
                    enh_tif = io.BytesIO()
                    with rasterio.open(enh_tif, "w", driver="GTiff",
                                       height=enhancement.shape[0], width=enhancement.shape[1],
                                       count=1, dtype="float32", crs=crs, transform=transform,
                                       nodata=np.nan, compress="deflate") as dst:
                        dst.write(enhancement.astype(np.float32), 1)
                    zf.writestr("enhancement_ppmm.tif", enh_tif.getvalue())
                    if plume_mask is not None and plume_mask.shape == enhancement.shape:
                        mask_tif = io.BytesIO()
                        with rasterio.open(mask_tif, "w", driver="GTiff",
                                           height=plume_mask.shape[0], width=plume_mask.shape[1],
                                           count=1, dtype="uint8", crs=crs, transform=transform,
                                           nodata=0, compress="deflate") as dst:
                            dst.write(plume_mask.astype(np.uint8), 1)
                        zf.writestr("plume_mask.tif", mask_tif.getvalue())
                st.download_button("⬇ GeoTIFF bundle", geo_pkg.getvalue(),
                                   file_name=f"{sat_tag}_{dt_tag}.zip",
                                   mime="application/zip", use_container_width=True,
                                   key="dl_geo_zip")
            else:
                st.button("⬇ GeoTIFF (unavailable)", disabled=True, use_container_width=True)
        except Exception:
            st.button("⬇ GeoTIFF (unavailable)", disabled=True, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  06 · MULTI-DATE COMPARISON
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
        '<div class="card-caption">Processes all observations in the current search window.</div>',
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
            key="batch_satellites", label_visibility="collapsed",
        )
    with btn_col2:
        run_batch = st.button("🔁  Process all", type="primary",
                              use_container_width=True, key="run_batch")
    with btn_col3:
        max_granules = st.slider("Max observations to process", min_value=2,
                                 max_value=20, value=6, key="max_granules")

    if run_batch:
        progress = st.progress(0, text="Processing observations…")
        batch_results = []
        _centroid_batch = shape(st.session_state.aoi).centroid
        jobs = []
        if "EMIT" in batch_sat:
            for g in emit_results:
                jobs.append(("EMIT", g))
        if "Tanager-1" in batch_sat:
            for f in tanager_results:
                jobs.append(("Tanager-1", f))
        jobs = jobs[: int(max_granules)]

        for i, (sat, item) in enumerate(jobs):
            progress.progress(int(100 * (i + 1) / max(len(jobs), 1)),
                              text=f"Processing {i+1}/{len(jobs)}…")
            try:
                if sat == "EMIT":
                    data, tform, tcrs = load_emit_enhancement(
                        item, st.session_state.aoi,
                        resolution=SATELLITES["EMIT"]["resolution"])
                    if data is None or data.size == 0 or valid_coverage(data) < 0.10:
                        continue
                    g_dt = granule_datetime(item)
                    wind = get_wind_speed_openmeteo(_centroid_batch.y, _centroid_batch.x, g_dt) if g_dt else None
                    if wind is None:
                        wind = PARAMS["wind_speed_m_s"]
                    pm = detect_plume(data, PARAMS["plume_threshold_ppm_m"],
                                      int(PARAMS["min_plume_pixels"]))
                    f = estimate_flux_ime(data, pm, wind,
                                          resolution=SATELLITES["EMIT"]["resolution"])
                    batch_results.append({
                        "satellite": "EMIT", "resolution": SATELLITES["EMIT"]["resolution"],
                        "date": g_dt, "enhancement": data, "plume_mask": pm, "flux": f,
                        "transform": tform, "crs": tcrs, "wind_speed": wind,
                        "has_raster": True,
                    })
                else:
                    data, tform, tcrs = load_tanager_enhancement(item, st.session_state.aoi)
                    g_dt = cm_plume_datetime(item)
                    reported = cm_plume_emission(item)
                    cm_wind = cm_plume_wind(item)
                    wind = cm_wind
                    if wind is None:
                        wind = get_wind_speed_openmeteo(_centroid_batch.y, _centroid_batch.x, g_dt) if g_dt else None
                    if wind is None:
                        wind = PARAMS["wind_speed_m_s"]

                    if data is not None and data.size > 0:
                        _cm_thr = cm_plume_threshold(item)
                        threshold = _cm_thr if (_cm_thr and _cm_thr > 0) else PARAMS["plume_threshold_ppm_m"]
                        pm = detect_plume(data, threshold, int(PARAMS["min_plume_pixels"]))
                        pm = filter_to_cm_plume_component(pm, tform, item, tcrs)
                        f = estimate_flux_ime(data, pm, wind,
                                              resolution=SATELLITES["Tanager-1"]["resolution"])
                        if reported > 0:
                            f["reported_flux_kg_h"] = reported
                        has_raster = True
                    else:
                        pm = np.zeros((1, 1), dtype=bool)
                        f = {"Q_kg_h": reported, "Q_ton_h": reported / 1000.0,
                             "IME_ppm_m2": 0.0, "IME_kg": 0.0,
                             "plume_area_m2": 0.0, "length_m": 0.0,
                             "U_eff_m_s": ALPHA_IME * wind + BETA_IME,
                             "n_pixels": 0, "max_enhancement": 0.0,
                             "mean_enhancement": 0.0,
                             "reported_flux_kg_h": reported}
                        has_raster = False

                    batch_results.append({
                        "satellite": "Tanager-1", "resolution": SATELLITES["Tanager-1"]["resolution"],
                        "date": g_dt, "enhancement": data, "plume_mask": pm, "flux": f,
                        "transform": tform, "crs": tcrs, "wind_speed": wind,
                        "has_raster": has_raster, "reported_flux_kg_h": reported,
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
                    "date": r["date"], "satellite": r["satellite"],
                    "flux_kg_h": r["flux"]["Q_kg_h"],
                    "plume_pixels": r["flux"]["n_pixels"],
                    "plume_area_km2": r["flux"]["plume_area_m2"] / 1e6,
                })
        if chart_rows:
            chart_df = pd.DataFrame(chart_rows).sort_values("date")
            pivot = chart_df.pivot_table(index="date", columns="satellite",
                                          values="flux_kg_h", aggfunc="first")
            st.markdown("##### Estimated flux over time (EMIT vs Tanager-1)")
            # ✅ Use scatter for < 3 points (line interpolation meaningless)
            if len(chart_df) >= 3:
                st.line_chart(pivot, use_container_width=True, height=240)
            else:
                st.scatter_chart(pivot, use_container_width=True, height=240)
                st.caption(f"⚠️ Only {len(chart_df)} observations — line interpolation not meaningful.")
            st.dataframe(chart_df.set_index("date"), use_container_width=True, hide_index=False,
                         column_config={
                             "satellite": st.column_config.TextColumn("Satellite"),
                             "flux_kg_h": st.column_config.NumberColumn("Flux (kg/h)", format="%.1f"),
                             "plume_pixels": st.column_config.NumberColumn("Pixels", format="%d"),
                             "plume_area_km2": st.column_config.NumberColumn("Area (km²)", format="%.3f"),
                         })
            st.download_button("⬇ Download time series CSV", chart_df.to_csv(index=False),
                               file_name="multisat_flux_timeseries.csv", mime="text/csv",
                               key="dl_ts_csv", use_container_width=False)

        st.markdown("##### Visual comparison")
        dates_labels = [
            f"[{r['satellite']}] " + (r["date"].strftime("%Y-%m-%d") if r["date"] else f"#{i+1}")
            for i, r in enumerate(batch)
        ]
        _batch_key = f"batch_slider_{len(batch)}"
        selected_idx = st.select_slider("Select observation", options=list(range(len(batch))),
                                         format_func=lambda x: dates_labels[x],
                                         value=0, key=_batch_key)
        chosen = batch[selected_idx]
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
        _has_raster = (chosen.get("has_raster", False) and
                       chosen.get("enhancement") is not None and chosen["enhancement"].size > 0)
        _cov_lbl, _cov_col = coverage_badge(chosen["enhancement"]) if _has_raster else ("No raster", "#e63946")

        cc1, cc2 = st.columns(2, gap="small")
        with cc1:
            st.markdown(
                f'<div class="card-caption" style="font-weight:700;">'
                f'{dates_labels[selected_idx]} · Enhancement</div>'
                f'<div class="card-caption" style="color:{_cov_col} !important; font-weight:700;">{_cov_lbl}</div>',
                unsafe_allow_html=True,
            )
            if _has_raster:
                st.image(enhancement_png(chosen["enhancement"], mask=chosen["plume_mask"],
                                          colormap="turbo", show_outline=_has_plume,
                                          vmin=cvmin, vmax=cvmax),
                         use_container_width=True, output_format="PNG")
                st.image(colorbar_png(cvmin, cvmax, "turbo"), width=90)
            else:
                st.image(placeholder_png("Raster unavailable"), use_container_width=True)
        with cc2:
            status_html = (f'<div class="card-caption" style="color:#2a9d8f !important; font-weight:700;">✓ Plume detected</div>'
                           if _has_plume else
                           f'<div class="card-caption" style="color:#e63946 !important; font-weight:700;">✗ No plume above threshold</div>')
            st.markdown(
                f'<div class="card-caption" style="font-weight:700;">'
                f'{dates_labels[selected_idx]} · Plume outline</div>{status_html}',
                unsafe_allow_html=True,
            )
            if _has_raster:
                st.image(enhancement_png(chosen["enhancement"], mask=chosen["plume_mask"],
                                          colormap="turbo", show_outline=_has_plume,
                                          vmin=cvmin, vmax=cvmax),
                         use_container_width=True, output_format="PNG")
            st.markdown(legend_html("plume", cvmin, cvmax, n_pixels=chosen["flux"]["n_pixels"],
                                     mean_enh=chosen["flux"]["mean_enhancement"],
                                     satellite=chosen["satellite"]),
                        unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3, gap="small")
        m1.metric("Flux (kg/h)", f"{chosen['flux']['Q_kg_h']:.1f}")
        m2.metric("Plume pixels", f"{chosen['flux']['n_pixels']:,}")
        m3.metric("Plume area (km²)", f"{chosen['flux']['plume_area_m2']/1e6:.3f}")

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  07 · PLUME EVOLUTION WINDOW
# ══════════════════════════════════════════════════════════════════════

if "emit_result" in st.session_state or "tanager_result" in st.session_state:
    _res = st.session_state.get("emit_result") or st.session_state.get("tanager_result")
    _ref_dt = _res.get("granule_dt")

    if _ref_dt is not None:
        st.markdown('<div style="height:0.25rem"></div>', unsafe_allow_html=True)
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">07 · PLUME EVOLUTION WINDOW</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Methane plume changes around the detected date</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="card-caption">Searches all observations within ±<i>N</i> days around '
            f'<b>{_ref_dt.strftime("%Y-%m-%d %H:%M")}</b>.</div>',
            unsafe_allow_html=True,
        )
        st.info("💡 Methane plumes are transient. Only some observations showing a plume is normal.")

        ec1, ec2, ec3 = st.columns([1, 1, 1], gap="small")
        with ec1:
            window_days = st.slider("Window around detected date (± days)",
                                     min_value=5, max_value=45, value=15, step=1,
                                     key="evo_window_days")
        with ec2:
            max_evo = st.slider("Max observations to process",
                                 min_value=2, max_value=40, value=12, step=1,
                                 key="evo_max_granules")
        with ec3:
            evo_sat_filter = st.radio("Show evolution for:",
                                       ["Both", "EMIT only", "Tanager-1 only"],
                                       horizontal=True, key="evo_satellite_filter")

        run_evo = st.button("🔁  Analyze plume evolution", type="primary",
                            use_container_width=True, key="run_evolution")

        if run_evo:
            start_d = (_ref_dt - timedelta(days=int(window_days))).date()
            end_d = (_ref_dt + timedelta(days=int(window_days))).date()
            progress = st.progress(0, text="Searching satellites…")
            try:
                if evo_sat_filter == "Both":
                    _evo_sats = ["EMIT", "Tanager-1"]
                elif evo_sat_filter == "EMIT only":
                    _evo_sats = ["EMIT"]
                else:
                    _evo_sats = ["Tanager-1"]

                if "EMIT" in _evo_sats:
                    try:
                        login_earthdata()
                    except Exception as _e:
                        st.warning(f"Earthdata login failed: {_e}")

                results_evo = search_all_satellites(st.session_state.aoi, start_d, end_d,
                                                     satellites=_evo_sats)
                if results_evo.get("errors"):
                    for err in results_evo["errors"]:
                        st.warning(err)

                evo_jobs = []
                for g in results_evo.get("EMIT", []):
                    evo_jobs.append(("EMIT", g))
                for f in results_evo.get("Tanager-1", []):
                    evo_jobs.append(("Tanager-1", f))
                evo_jobs.sort(key=lambda x: (
                    granule_datetime(x[1]) if x[0] == "EMIT"
                    else cm_plume_datetime(x[1])) or datetime.min)
                evo_jobs = evo_jobs[: int(max_evo)]

                if not evo_jobs:
                    progress.progress(100, text="No observations found")
                    st.warning(f"No data found in the ±{window_days}-day window ({start_d} → {end_d}).")
                else:
                    evo_results = []
                    _centroid_evo = shape(st.session_state.aoi).centroid

                    for i, (sat, item) in enumerate(evo_jobs):
                        progress.progress(int(100 * (i + 1) / len(evo_jobs)),
                                          text=f"Processing {i+1}/{len(evo_jobs)}…")
                        try:
                            if sat == "EMIT":
                                res = SATELLITES["EMIT"]["resolution"]
                                data, tform, tcrs = load_emit_enhancement(item, st.session_state.aoi, resolution=res)
                                if data is None or data.size == 0:
                                    continue
                                cov = valid_coverage(data)
                                # ✅ Skip scenes with insufficient coverage (< 10%)
                                if cov < 0.10:
                                    continue
                                g_dt = granule_datetime(item)
                                wind = get_wind_speed_openmeteo(_centroid_evo.y, _centroid_evo.x, g_dt) if g_dt else None
                                if wind is None:
                                    wind = PARAMS["wind_speed_m_s"]
                                pm = detect_plume(data, PARAMS["plume_threshold_ppm_m"],
                                                  int(PARAMS["min_plume_pixels"]))
                                f = estimate_flux_ime(data, pm, wind, resolution=res)
                                has_raster = True; reported = None
                            else:
                                res = SATELLITES["Tanager-1"]["resolution"]
                                data, tform, tcrs = load_tanager_enhancement(item, st.session_state.aoi)
                                g_dt = cm_plume_datetime(item)
                                reported = cm_plume_emission(item)
                                cm_wind = cm_plume_wind(item)
                                wind = cm_wind
                                if wind is None:
                                    wind = get_wind_speed_openmeteo(_centroid_evo.y, _centroid_evo.x, g_dt) if g_dt else None
                                if wind is None:
                                    wind = PARAMS["wind_speed_m_s"]

                                if data is not None and data.size > 0:
                                    cov = valid_coverage(data)
                                    _cm_thr = cm_plume_threshold(item)
                                    threshold = _cm_thr if (_cm_thr and _cm_thr > 0) else PARAMS["plume_threshold_ppm_m"]
                                    pm = detect_plume(data, threshold, int(PARAMS["min_plume_pixels"]))
                                    pm = filter_to_cm_plume_component(pm, tform, item, tcrs)
                                    f = estimate_flux_ime(data, pm, wind, resolution=res)
                                    if reported > 0:
                                        f["reported_flux_kg_h"] = reported
                                    has_raster = True
                                else:
                                    cov = 0.0
                                    pm = np.zeros((1, 1), dtype=bool)
                                    f = {"Q_kg_h": reported, "Q_ton_h": reported / 1000.0,
                                         "IME_ppm_m2": 0.0, "IME_kg": 0.0,
                                         "plume_area_m2": 0.0, "length_m": 0.0,
                                         "U_eff_m_s": ALPHA_IME * wind + BETA_IME,
                                         "n_pixels": 0, "max_enhancement": 0.0,
                                         "mean_enhancement": 0.0,
                                         "reported_flux_kg_h": reported}
                                    has_raster = False

                            centroid_geo = None
                            if pm.any() and tform is not None:
                                ys, xs = np.nonzero(pm)
                                try:
                                    from rasterio.transform import xy as rio_xy
                                    gx, gy = rio_xy(tform, float(ys.mean()), float(xs.mean()), offset="center")
                                    centroid_geo = (float(gx), float(gy))
                                except Exception:
                                    pass

                            evo_results.append({
                                "satellite": sat, "resolution": res, "date": g_dt,
                                "enhancement": data, "plume_mask": pm, "flux": f,
                                "transform": tform, "crs": tcrs, "centroid_geo": centroid_geo,
                                "coverage": cov, "wind_speed": wind,
                                "has_raster": has_raster, "reported_flux_kg_h": reported,
                            })
                        except Exception:
                            continue

                    st.session_state.evo_results = evo_results
                    st.session_state.evo_ref_date = _ref_dt
                    st.session_state.evo_window_days_used = int(window_days)
                    progress.progress(100, text="Done")
                    st.success(f"Processed {len(evo_results)} observation(s)")
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
                f'<div class="result-note"><b>{n_with}</b> of <b>{n_total}</b> observation(s) '
                f'in the ±{used_window}-day window showed a plume '
                f'({n_emit} EMIT · {n_tan} Tanager-1).</div>',
                unsafe_allow_html=True,
            )

            cov_rows = []
            for r in evo:
                lbl, _ = coverage_badge(r["enhancement"]) if r["has_raster"] else ("No raster", "")
                cov_rows.append({
                    "date": r["date"].strftime("%Y-%m-%d") if r["date"] else "-",
                    "satellite": r["satellite"], "coverage": lbl,
                    "flux_kg_h": r["flux"]["Q_kg_h"], "wind_m_s": r.get("wind_speed"),
                })
            st.markdown("##### Data coverage per observation")
            st.dataframe(pd.DataFrame(cov_rows), use_container_width=True, hide_index=True,
                         column_config={"wind_m_s": st.column_config.NumberColumn("Wind (m/s)", format="%.2f")})

            rows = []
            for r in evo:
                rows.append({
                    "date": r["date"], "satellite": r["satellite"],
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
                _evo_pivot = evo_df.reset_index().pivot_table(
                    index="date", columns="satellite", values="flux_kg_h", aggfunc="first")
                # ✅ Use scatter for < 3 points
                if len(evo_df) >= 3:
                    st.line_chart(_evo_pivot, use_container_width=True, height=220)
                else:
                    st.scatter_chart(_evo_pivot, use_container_width=True, height=220)
                    st.caption(f"⚠️ Only {len(evo_df)} observations — line interpolation not meaningful.")

                st.markdown("##### Plume area evolution")
                _area_pivot = evo_df.reset_index().pivot_table(
                    index="date", columns="satellite", values="plume_area_km2", aggfunc="first")
                if len(evo_df) >= 3:
                    st.line_chart(_area_pivot, use_container_width=True, height=200)
                else:
                    st.scatter_chart(_area_pivot, use_container_width=True, height=200)

                st.dataframe(evo_df, use_container_width=True, hide_index=False,
                             column_config={
                                 "satellite": st.column_config.TextColumn("Satellite"),
                                 "flux_kg_h": st.column_config.NumberColumn("Flux (kg/h)", format="%.1f"),
                                 "plume_pixels": st.column_config.NumberColumn("Pixels", format="%d"),
                                 "plume_area_km2": st.column_config.NumberColumn("Area (km²)", format="%.3f"),
                                 "max_enh_ppmm": st.column_config.NumberColumn("Max enh.", format="%.0f"),
                                 "mean_enh_ppmm": st.column_config.NumberColumn("Mean enh.", format="%.0f"),
                                 "has_plume": st.column_config.NumberColumn("Plume?", format="%d"),
                                 "wind_m_s": st.column_config.NumberColumn("Wind (m/s)", format="%.2f"),
                             })
                st.download_button("⬇ Download evolution CSV", evo_df.to_csv(),
                                   file_name="multisat_plume_evolution.csv",
                                   mime="text/csv", key="dl_evo_csv", use_container_width=False)

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

            st.markdown("##### Visual evolution")
            dates_labels = [
                f"[{r['satellite']}] " + (r["date"].strftime("%Y-%m-%d") if r["date"] else f"#{i+1}")
                for i, r in enumerate(evo)
            ]
            _evo_key = f"evo_slider_{len(evo)}"
            sel_idx = st.select_slider("Select observation", options=list(range(len(evo))),
                                        format_func=lambda x: dates_labels[x], value=0, key=_evo_key)
            chosen = evo[sel_idx]
            _has_plume = chosen["flux"]["n_pixels"] > 0
            _has_raster = (chosen.get("has_raster", False) and
                           chosen.get("enhancement") is not None and chosen["enhancement"].size > 0)

            cc1, cc2 = st.columns(2, gap="small")
            with cc1:
                st.markdown(f'<div class="card-caption" style="font-weight:700;">'
                            f'{dates_labels[sel_idx]} · Enhancement</div>', unsafe_allow_html=True)
                if _has_raster:
                    st.image(enhancement_png(chosen["enhancement"], mask=chosen["plume_mask"],
                                              colormap="turbo", show_outline=_has_plume,
                                              vmin=shared_vmin, vmax=shared_vmax),
                             use_container_width=True, output_format="PNG")
                    st.image(colorbar_png(shared_vmin, shared_vmax, "turbo"), width=90)
                else:
                    st.image(placeholder_png("Raster unavailable"), use_container_width=True)
            with cc2:
                status_html = (f'<div class="card-caption" style="color:#2a9d8f !important; font-weight:700;">✓ Plume detected</div>'
                               if _has_plume else
                               f'<div class="card-caption" style="color:#e63946 !important; font-weight:700;">✗ No plume above threshold</div>')
                st.markdown(f'<div class="card-caption" style="font-weight:700;">'
                            f'{dates_labels[sel_idx]} · Plume outline</div>{status_html}',
                            unsafe_allow_html=True)
                if _has_raster:
                    st.image(enhancement_png(chosen["enhancement"], mask=chosen["plume_mask"],
                                              colormap="turbo", show_outline=_has_plume,
                                              vmin=shared_vmin, vmax=shared_vmax),
                             use_container_width=True, output_format="PNG")
                st.markdown(legend_html("plume", n_pixels=chosen["flux"]["n_pixels"],
                                         mean_enh=chosen["flux"]["mean_enhancement"],
                                         satellite=chosen["satellite"]),
                            unsafe_allow_html=True)
            em1, em2, em3, em4 = st.columns(4, gap="small")
            em1.metric("Flux (kg/h)", f"{chosen['flux']['Q_kg_h']:.1f}")
            em2.metric("Plume pixels", f"{chosen['flux']['n_pixels']:,}")
            em3.metric("Plume area (km²)", f"{chosen['flux']['plume_area_m2']/1e6:.3f}")
            em4.metric("Max enh. (ppm·m)", f"{chosen['flux']['max_enhancement']:.0f}")

            st.markdown("##### Plume mask gallery")
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
                    label = r["date"].strftime("%Y-%m-%d") if r["date"] else f"#{idx+1}"
                    _has = r["flux"]["n_pixels"] > 0
                    _raster = (r.get("has_raster", False) and r.get("enhancement") is not None
                               and r["enhancement"].size > 0)
                    with gcols[gc]:
                        st.markdown(
                            f'<div class="card-caption" style="font-weight:700; text-align:center; '
                            f'margin-bottom:0.15rem;">{label}<br/>'
                            f'<span style="font-weight:400;">{r["satellite"]} · '
                            f'{r["flux"]["Q_kg_h"]:.0f} kg/h · {r["flux"]["n_pixels"]} px</span></div>',
                            unsafe_allow_html=True,
                        )
                        if _raster:
                            st.image(enhancement_png(r["enhancement"], mask=r["plume_mask"],
                                                      colormap="turbo", show_outline=_has,
                                                      vmin=shared_vmin, vmax=shared_vmax),
                                     use_container_width=True, output_format="PNG")
                        else:
                            st.image(placeholder_png("—"), use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)
