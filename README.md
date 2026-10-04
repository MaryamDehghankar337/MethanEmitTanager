# MethanEmitTanager

A comparative Streamlit application for methane plume detection and quantification using NASA's **EMIT** and Planet's **Tanager-1** hyperspectral satellite data.

## Overview

This app provides a unified interface for:
- Searching and processing **NASA EMIT** methane enhancement data.
- Searching and processing **Planet Tanager-1** radiance data.
- **Comparing** detection results from both satellites side-by-side for the same Area of Interest (AOI).

The core detection and flux estimation algorithms are based on the **Integrated Methane Enhancement (IME)** method, the same approach used by Carbon Mapper.

## Key Features

- **EMIT Pipeline**: Full support for searching, loading, detecting plumes, and estimating flux from EMIT data.
- **Tanager-1 Pipeline**: Search and load Tanager-1 scenes via STAC, with a placeholder for matched-filter enhancement retrieval.
- **Comparative Analysis**: A dedicated section to search for Tanager-1 scenes over the same AOI and directly compare plume metrics (flux, area, pixel count) with EMIT results.
- **Automatic Wind Data**: Wind speed is automatically fetched from the Open-Meteo API (based on ERA5) for each scene's time and location.
- **Interactive Map**: Define AOI by drawing, searching for place names, or entering coordinates.
- **Multi-Date & Evolution Analysis**: Tools to analyze plume changes over time using EMIT data.
- **Export Options**: Download results as PNG images, CSV data, or GeoTIFF bundles.

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd MethanEmitTanager
