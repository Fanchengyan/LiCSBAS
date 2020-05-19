# LiCSBAS

LiCSBAS is an open-source package in Python and bash to carry out InSAR time series analysis using LiCSAR products (i.e., unwrapped interferograms and coherence) which are freely available on the [COMET-LiCS web portal](https://comet.nerc.ac.uk/COMET-LiCS-portal/).



Users can easily derive the time series and velocity of the displacement if sufficient LiCSAR products are available in the area of interest. LiCSBAS also contains visualization tools to interactively display the time series of displacement to help investigation and interpretation of the results.

[<img src="https://raw.githubusercontent.com/wiki/yumorishita/LiCSBAS/images/comet-lics-web.png"  height="240">](https://comet.nerc.ac.uk/COMET-LiCS-portal/) <img src="https://raw.githubusercontent.com/wiki/yumorishita/LiCSBAS/images/sample_vel.png"  height="240">  <img src="https://raw.githubusercontent.com/wiki/yumorishita/LiCSBAS/images/sample_ts.png"  height="240">

THIS IS RESEARCH CODE PROVIDED TO YOU "AS IS" WITH NO WARRANTIES OF CORRECTNESS. USE AT YOUR OWN RISK.

## Documentation and Bug Reports

See the [**wiki**](https://github.com/Fanchengyan/LiCSBAS/wiki) pages and [quick start](https://github.com/Fanchengyan/LiCSBAS/wiki/2_0_workflow).

If you have found an issue or bug, please report it on the [issues page](https://github.com/yumorishita/LiCSBAS/issues), rather than emailing me.

## Sample Products and Tutorial

- Frame ID: 124D_04854_171313 (Italy)
- Time: 2016/09/09-2018/05/08 (~1.7 years, ~60 images, ~170 interferograms)
- Clipped around Campi Flegrei (14.03/14.22/40.78/40.90)

- Tutorial: [LiCSBAS_sample_CF.pdf](https://raw.githubusercontent.com/wiki/yumorishita/LiCSBAS/documents/LiCSBAS_sample_CF.pdf) (1.3MB)

- Sample batch script: [batch_LiCSBAS_sample_CF.sh](https://raw.githubusercontent.com/wiki/yumorishita/LiCSBAS/documents/batch_LiCSBAS_sample_CF.sh)
- Sample results: [LiCSBAS_sample_CF.tar.gz](https://raw.githubusercontent.com/wiki/yumorishita/LiCSBAS/sample/LiCSBAS_sample_CF.tar.gz) (40MB)

## Citation

Morishita, Y.; Lazecky, M.; Wright, T.J.; Weiss, J.R.; Elliott, J.R.; Hooper, A. LiCSBAS: An Open-Source InSAR Time Series Analysis Package Integrated with the LiCSAR Automated Sentinel-1 InSAR Processor. *Remote Sens.* **2020**, *12*, 424, https://doi.org/10.3390/RS12030424.

M. Lazecky et al. LiCSAR: An Automatic InSAR Tool for Monitoring Tectonic and Volcanic Activities. *in prep*.

## Acknowledgements

This work has been accomplished during Y. Morishita’s visit at University of Leeds, funded by JSPS Overseas Research Fellowship.

COMET is the UK Natural Environment Research Council's Centre for the Observation and Modelling of Earthquakes, Volcanoes and Tectonics. LiCSAR is developed as part of the NERC large grant, "Looking inside the continents from Space" (NE/K010867/1). LiCSAR contains modified Copernicus Sentinel data [2014-] analysed by the COMET. LiCSAR uses [JASMIN](http://jasmin.ac.uk), the UK’s collaborative data analysis environment.

The [Scientific Colour Maps](http://www.fabiocrameri.ch/colourmaps.php) ([Crameri, 2018](https://doi.org/10.5194/gmd-11-2541-2018)) is used in LiCSBAS.

## Significant Updates

##### April 2020

- Add -c (coherence based mask) option in Step 0-4.
- Add `LiCSBAS_color_geotiff.py` which creates a colored GeoTIFF from a data GeoTIFF.
- Add `LiCSBAS_contour.py` which creates a contour GeoJSON from a data GeoTIFF.
- Add `edit_batch_LiCSBAS.sh` for easy edit of batch_LiCSBAS.sh.

##### 2 March, 2020

- Add an option to remove topography-correlated tropospheric noise at Step 1-6. 
- Find a stable reference point at Steps 1-3 and 1-6.
- Compatible with a new file structure of LiCSAR including GACOS data auto-download.
- Use the uint8 format for coherence instead of float32.
- Add `LiCSBAS_plot_network.py`.



*Yu Morishita (PhD)\
JSPS Overseas Research Fellow (June 2018-June 2020)\
Visiting Researcher, COMET, School of Earth and Environment, University of Leeds (June 2018-June 2020)\
Chief Researcher, Geography and Crustal Dynamics Research Center, Geospatial Information Authority of Japan (GSI)*

[<img src="https://raw.githubusercontent.com/wiki/yumorishita/LiCSBAS/images/COMET_logo.png"  height="60">](https://comet.nerc.ac.uk/)   [<img src="https://raw.githubusercontent.com/wiki/yumorishita/LiCSBAS/images/logo-leeds.png"  height="60">](https://environment.leeds.ac.uk/see/)  [<img src="https://raw.githubusercontent.com/wiki/yumorishita/LiCSBAS/images/LiCS_logo.jpg"  height="60">](https://comet.nerc.ac.uk/COMET-LiCS-portal/)   [<img src="https://raw.githubusercontent.com/wiki/yumorishita/LiCSBAS/images/GSI_logo.png"  height="60">](https://www.gsi.go.jp/)

