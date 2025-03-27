# weightstack

Python toolkit for generating weight maps, confidence maps, and masks for astronomical FITS images. These maps are essential for proper image stacking, coaddition, and source detection in astronomical data processing pipelines.


**THIS IS WORK IN PROGRESS ---  CHECK IN LATER**


## Features

- Generate various types of maps from astronomical FITS images:
  - Weight maps (inverse variance maps with masked bad pixels)
  - Confidence maps (binary good/bad pixel flags)
  - Bitmasks for different defect types
  - Sky background maps
- Automatic detection of:
  - Bad pixels from flat fields
  - Saturated pixels
  - Cosmic rays (using astroscrappy)
  - Astronomical objects
  - Satellite streaks (using Hough transform)
- Compatible with Multi-Extension FITS (MEF) files
- Configurable via YAML files
- SWarp and casutools compatible weight map outputs

## License

GPL-3.0