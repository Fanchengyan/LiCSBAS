#!/usr/bin/env python3
"""
v1.3 20200311 Yu Morishita, Uni of Leeds and GSI

========
Overview
========
This script downloads GeoTIFF files of unw (unwrapped interferogram) and cc (coherence) in the specified frame ID from COMET-LiCS web portal. The -f option is not necessary when the frame ID can be automatically identified from the name of the working directory. GACOS data can also be downloaded if available. Existing GeoTIFF files are not re-downloaded to save time, i.e., only the newly available data will be downloaded.

============
Output files
============
 - GEOC/
   - yyyymmdd_yyyymmdd/
     - yyyymmdd_yyyymmdd.geo.unw.tif
     - yyyymmdd_yyyymmdd.geo.cc.tif
  [- *.geo.mli.tif (using just one first epoch)]
   - *.geo.E.tif
   - *.geo.N.tif
   - *.geo.U.tif
   - *.geo.hgt.tif
   - baselinestime_format = lambda a:time.strftime('%H:%M:%S',time.localtime(a))
Usage
=====
LiCSBAS01_get_geotiff.py [-f frameID] [-s yyyymmdd] [-e yyyymmdd] [--get_gacos]

 -f  Frame ID (e.g., 021D_04972_131213). (Default: Read from directory name)
 -s  Start date (Default: 20141001)
 -e  End date (Default: Today)
 --get_gacos  Download GACOS data as well if available
 
"""
#%% Change log
'''
v1.3 2020031 Yu Morishita, Uni of Leeds and GSI
 - Deal with only new LiCSAR file structure
v1.2 20200302 Yu Morishita, Uni of Leeds and GSI
 - Compatible with new LiCSAR file structure (backward-compatible)
 - Add --get_gacos option
v1.1 20191115 Yu Morishita, Uni of Leeds and GSI
 - Download mli and hgt
v1.0 20190729 Yu Morishita, Uni of Leeds and GSI
 - Original implementation
'''


#%% Import
import getopt
import os
import re
import sys
import time
import requests
from bs4 import BeautifulSoup
import numpy as np
import datetime as dt
import LiCSBAS_tools_lib as tools_lib
from data_downloader import downloader

class Usage(Exception):
    """Usage context manager"""
    def __init__(self, msg):
        self.msg = msg


#%% Main
def main(argv=None):
   
    #%% Check argv
    if argv == None:
        argv = sys.argv
        
    start = time.time()
    ver=1.2; date=20200227; author="Y. Morishita"
    print("\n{} ver{} {} {}".format(os.path.basename(argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(argv[0]), ' '.join(argv[1:])), flush=True)


    #%% Set default
    frameID = []
    startdate = 20141001
    enddate = int(dt.date.today().strftime("%Y%m%d"))
    get_gacos = False
    

    #%% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hl:f:s:e:", ["help", "get_gacos"])
        except getopt.error as msg:
            raise Usage(msg)
        for o, a in opts:
            if o == '-h' or o == '--help':
                print(__doc__)
                return 0
            else:
                if o == '-l':
                    limit = int(a)
                if o == '-f':
                    frameID = a
                if o == '-s':
                    startdate = int(a)
                if o == '-e':
                    enddate = int(a)
                if o == '--get_gacos':
                    get_gacos = True


    except Usage as err:
        print("\nERROR:", file=sys.stderr, end='')
        print("  "+str(err.msg), file=sys.stderr)
        print("\nFor help, use -h or --help.\n", file=sys.stderr)
        return 2
 

    #%% Determine frameID
    wd = os.getcwd()
    if not frameID: ## if frameID not indicated
        _tmp = re.findall(r'\d{3}[AD]_\d{5}_\d{6}', wd)
        ##e.g., 021D_04972_131213
        if len(_tmp)==0:
            print('\nFrame ID cannot be identified from dir name!', file=sys.stderr)
            print('Use -f option', file=sys.stderr)
            return
        else:
            frameID = _tmp[0]
            print('\nFrame ID is {}\n'.format(frameID), flush=True)
    trackID = str(int(frameID[0:3]))


    #%% Directory and file setting
    outdir = os.path.join(wd, 'GEOC')
    if not os.path.exists(outdir): os.mkdir(outdir)
    os.chdir(outdir)

    LiCSARweb = 'http://gws-access.ceda.ac.uk/public/nceo_geohazards/LiCSAR_products/'

    # All metadatas
    #%% ENU and hgt
    urls = []
    file_names = []
    for ENU in ['E', 'N', 'U', 'hgt']:
        enutif = '{}.geo.{}.tif'.format(frameID, ENU)
        file_names.append(enutif)

        url = os.path.join(LiCSARweb, trackID, frameID, 'metadata', enutif)
        urls.append(url)

    #%% baselines and metadata.txt
    url = os.path.join(LiCSARweb, trackID, frameID, 'metadata', 'baselines')
    urls.append(url)
    file_names.append('baselines')

    url = os.path.join(LiCSARweb, trackID, frameID, 'metadata', 'metadata.txt')
    urls.append(url)
    file_names.append('metadata.txt')

    #%% mli
    ### Get available dates
    url = os.path.join(LiCSARweb, trackID, frameID, 'epochs')
    response = requests.get(url)
    
    response.encoding = response.apparent_encoding #avoid garble
    html_doc = response.text
    soup = BeautifulSoup(html_doc, "html.parser")
    tags = soup.find_all(href=re.compile(r"\d{8}"))
    imdates_all = [tag.get("href")[0:8] for tag in tags]
    _imdates = np.int32(np.array(imdates_all))
    _imdates = (_imdates[(_imdates>=startdate)*(_imdates<=enddate)]).astype('str').tolist()
    
    ## Find earliest date in which mli is available

    urls_mli = np.array([os.path.join(url, imd, imd+'.geo.mli.tif') for imd in _imdates])
    status_ok = downloader.status_ok(urls_mli)
    try:
        url_mli = urls_mli[status_ok][0]
        print('Find the mli file in {}'.format(url_mli))
        mlitif = frameID+'.geo.mli.tif'

        urls.append(url_mli)
        file_names.append(mlitif)
    except:
        print('No mli available on {}'.format(url), file=sys.stderr, flush=True)
        
    # download metadatas
    downloader.async_download_datas(urls,None,file_names,limit,'All metadatas')


    #%% GACOS if specified
    if get_gacos:
        gacosdir = os.path.join(wd, 'GACOS')
        if not os.path.exists(gacosdir): os.mkdir(gacosdir)

        ### Get available dates
        print('\nDownload GACOS data', flush=True)
        url = os.path.join(LiCSARweb, trackID, frameID, 'epochs')
        response = requests.get(url)
        response.encoding = response.apparent_encoding #avoid garble
        html_doc = response.text
        soup = BeautifulSoup(html_doc, "html.parser")
        tags = soup.find_all(href=re.compile(r"\d{8}"))
        imdates_all = [tag.get("href")[0:8] for tag in tags]
        _imdates = np.int32(np.array(imdates_all))
        _imdates = (_imdates[(_imdates>=startdate)*(_imdates<=enddate)]).astype('str').tolist()

        urls = np.array([os.path.join(url, i, i+'.sltd.geo.tif') for i in _imdates])
        status_ok = downloader.status_ok(urls)
        urls_sltd = urls[status_ok]
        
        n_im = len(urls_sltd)
        if n_im > 0:
            extract_time = lambda a:os.path.basename(a).split('.')[0]
            print('{} GACOS data available from {} to {}'.format(n_im, extract_time(urls_sltd[0]),
                extract_time(urls_sltd[-1])), flush=True)
            ### Download
            files_sltd = [os.path.basename(url) for url in urls_sltd]
            downloader.async_download_datas(urls_sltd,gacosdir,files_sltd,limit,'GACOS')
        else:
            print('No GACOS data available from {} to {}'.format(startdate, enddate), flush=True)
        
        

    #%% unw and cc
    ### Get available dates
    print('\nDownload geotiff of unw and cc', flush=True)
    url = os.path.join(LiCSARweb, trackID, frameID, 'interferograms')
    response = requests.get(url)
    
    response.encoding = response.apparent_encoding #avoid garble
    html_doc = response.text
    soup = BeautifulSoup(html_doc, "html.parser")
    tags = soup.find_all(href=re.compile(r"\d{8}_\d{8}"))
    ifgdates_all = [tag.get("href")[0:17] for tag in tags]
    
    ### Extract during start_date to end_date
    ifgdates = []
    for ifgd in ifgdates_all:
        mimd = int(ifgd[:8])
        simd = int(ifgd[-8:])
        if mimd >= startdate and simd <= enddate:
            ifgdates.append(ifgd)
    
    n_ifg = len(ifgdates)
    imdates = tools_lib.ifgdates2imdates(ifgdates)
    print('{} IFGs available from {} to {}'.format(n_ifg, imdates[0], imdates[-1]), flush=True)
    
    ### Download
    urls_unw = []
    paths_unw = []
    urls_cc = []
    paths_cc = []
    for i, ifgd in enumerate(ifgdates):
        print('  Downloading {} ({}/{})...'.format(ifgd, i+1, n_ifg), flush=True)
        url_unw = os.path.join(url, ifgd, ifgd+'.geo.unw.tif')
        path_unw = os.path.join(ifgd, ifgd+'.geo.unw.tif')

        #new
        urls_unw.append(url_unw)
        paths_unw.append(path_unw)

        if not os.path.exists(ifgd):
            os.mkdir(ifgd)

        url_cc = os.path.join(url, ifgd, ifgd+'.geo.cc.tif')
        path_cc = os.path.join(ifgd, ifgd+'.geo.cc.tif')

        #new
        urls_cc.append(url_cc)
        paths_cc.append(path_cc)
        # if not tools_lib.download_data(url_cc, path_cc):
        #     print('    Error while downloading from {}'.format(url_cc), file=sys.stderr, flush=True)
    downloader.async_download_datas(urls_unw,None,paths_unw,limit,'unwraped interferograms')
    downloader.async_download_datas(urls_cc,None,paths_cc,limit,'coherences')

    #%% Finish
    elapsed_time = time.time()-start
    hour = int(elapsed_time/3600)
    minute = int(np.mod((elapsed_time/60),60))
    sec = int(np.mod(elapsed_time,60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour,minute,sec))

    print('\n{} Successfully finished!!\n'.format(os.path.basename(argv[0])))
    print('Output directory: {}\n'.format(outdir))


#%% main
if __name__ == "__main__":
    sys.exit(main())

