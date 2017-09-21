#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""   """

from __future__ import print_function
import argparse
import hjson
import numpy as np
import pandas as pd
import warnings
# import matplotlib.pyplot as plt

import getwth

__autor__ = 'Kyosuke Yamamoto (kyon)'
__date__ = '20 Sep 2017'


def load_cultivar_params(cultivar):
    ''' Load cultivar parameters '''

    with open(cultivar_params_file, 'r') as f:
        cultivar_params = hjson.load(f)

    if cultivar not in cultivar_params.keys():
        raise Exception('Unknown cultivar:', cultivar + '.', 'Choose from', cultivar_params.keys())

    return cultivar_params[cultivar]


def daylength(lat, doy):
    # if (class(doy) == "Date" | class(doy) == "character") {
    #     doy <- doyFromDate(doy)
    # }

    if lat > 90 or lat < -90:
        lat = np.nan

    doy.loc[doy == 366] = 365  # is this ok? uruudoshi?
    if np.any(doy < 1) or np.any(doy > 365):
        raise Exception('doy must be between 1 and 365')

    P = np.arcsin(0.39795 * np.cos(0.2163108 + 2 * np.arctan(0.9671396 * np.tan(0.0086 * (doy - 186)))))
    a = (np.sin(0.8333 * np.pi/180) + np.sin(lat * np.pi/180) * np.sin(P))/(np.cos(lat * np.pi/180) * np.cos(P))
    a[a < -1] = -1
    a[a > 1] = 1
    DL = 24 - (24/np.pi) * np.arccos(a)
    return DL


if __name__ == '__main__':

    #argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cultivar', '-c', default='Nipponbare', type=str)
    parser.add_argument('--weather', '-w', default='./daily_weather_28368.nasa.csv', type=str)
    parser.add_argument('--startday', '-s', default='2000-05-15', type=str)
    parser.add_argument('--co2', default=350, type=int)
    parser.add_argument('--transplant', action='store_true')
    args = parser.parse_args()

    #init
    cultivar_params_file = './cultivars.hjson'
    # plt.style.use('ggplot')

    #load cultivar parameters
    cultivar = load_cultivar_params(args.cultivar)

    #load weather data
    wth = getwth.main(args.weather)

    #Constants and parameters which may not be cutivar specific
    #Constants related to optcal properties of canopy
    SCAT = 0.25
    RSOL = 0.1
    RCAN = 0.22
    KREF = 0.5
    #Parameters related changes with DVI in radiation conversion efficiency
    CL = 0.001
    TAUC = 0.1
    #Conversion factors from rough to brown rice, and panicle to rough rice
    CVBP = 0.76
    CVPP = 0.9

    #Initial conditions for simulation
    if args.transplant:
        DVII = 0.15  # transplant
        LAII = 0.05
        DWI = 15
    else:
        DVII = 0     # emergence
        LAII = 0.0025
        DWI = 4

    IFLUG1 = 0
    IFLUG2 = 0
    CDED = 0.
    TCHECK = 0.
    STERP = 0.
    HTSUM = 0.
    HTDAY = 0.
    STHT = 0.0
    ATHHT = cultivar['HIMX']
    ATHLT = cultivar['HIMX']
    JJ = 1
    DWGRAIN = 0.0
    DVI = DVII
    LAI = LAII
    DW = DWI
    DWGRAIN = 0.0
    DWPAN = 0.0
    STHT = 0
    STLT = 0

    #weather data
    AVT = wth['w'].tavg
    RAD = wth['w'].srad
    TMX = wth['w'].tmax
    startday = pd.to_datetime(args.startday)
    endday = startday + pd.to_timedelta('200 days')
    days = pd.date_range(startday, endday)
    DL = daylength(wth['lat'], wth['w'].doy)

    startindex = wth['w'].date.eq(startday).idxmax()
    day = startindex - 1
    growing = True
    simday = 0

    res = {}
    # res = as.data.frame(matrix(ncol=9, nrow=length(days)))
    # colnames(res) = c('date','TMP', 'RAD','DL','DVI','LAI', 'DW', 'GY', 'PY')
    # class(res[,'date']) = 'Date'

    #Dynamic Section of The Model
    while growing:
        day += 1
        simday += 1
        if day >= wth['w'].shape[0]:
            warnings.warn('reached end of weather records')
            growing = Flse

        date = wth['w'].loc[day, 'date']
        res[date] = {
            'TMP': AVT[day],
            'RAD': RAD[day],
            'DL': DL[day],
            'DVI': DVI,
            'LAI': LAI,
            'DW': DW,
            'GY': DWGRAIN,
            'PY': DWPAN,
        }

        #Culculation of Developmental Index DVI
        if DVI < cultivar['DVIA']:
            EFT = AVT[day] - cultivar['TH']
            DVR = 1. / (cultivar['GV'] * (1.0 + exp(-cultivar@ALF * EFT)))
        } else if (DVI <= 1.0) {
            EFT = AVT[day]-cultivar@TH
            EFL = min(DL[day]-cultivar@LC,0.)
            DVR = (1.0-exp(cultivar@BDL*EFL))/(cultivar@GV*(1.0+exp(-cultivar@ALF*EFT)))
        } else {
            EFT = max(AVT[day]-cultivar@TCR,0.)
            DVR = (1.0-exp(-cultivar@KCR*EFT))/cultivar@GR
        }
        DVI = DVI+DVR

        #Culculation of LAI
        if (DVI < 0.95) {
            EFFTL = max(AVT[day]-cultivar@TCF,0.)
            GRLAI = LAI*cultivar@A*(1.0-exp(-cultivar@KF*EFFTL))*(1.0-(LAI/cultivar@FAS)**cultivar@ETA)
            GRL95 = GRLAI
            DVI95 = DVI
        } else if (GRLAI > 0.0  |  DVI <= 1.0) {
            GRLAI = GRL95*(1.0-(DVI-DVI95)/(1-DVI95))
            LAIMX = LAI
            DVI1 = DVI
        } else if (DVI < 1.1) {
            GRLAI = -(LAIMX*(1.0-cultivar@BETA)*(DVI-DVI1)/(1.1-DVI1))*DVR
        } else {
            GRLAI = -LAIMX*(1.0-cultivar@BETA)*DVR
        }
        LAI = LAI+GRLAI

        #Culuculation of Crop Dry Weight
        TAU = exp(-(1.0-SCAT)*cultivar@EXTC*LAI)
        REF = RCAN-(RCAN-RSOL)*exp(-KREF*LAI)
        ABSOP = 1.0-REF-(1.0-RSOL)*TAU
        ABSRAD = RAD[day]*ABSOP
        COVCO2 = cultivar@COVES*(1.54*(CO2-330.0)/(1787.0+(CO2-  330.0))+1.0)
        if (DVI < 1.0) {
            CONEF = COVCO2
        } else {
            CONEF = COVCO2*(1.0+CL)/(1.0+CL*exp((DVI-1.0)/TAUC))
        }
        DW = DW+CONEF*ABSRAD

        #Culuculation of Spikelet Sterility Percentage due to Cool Temerature
        if (DVI > 0.75  &  DVI < 1.2) {
            CDEG = max(cultivar@THOT-AVT[day],0.)
            CDED = CDED+CDEG
            SSTR = cultivar@STO+cultivar@BST*CDED**cultivar@PST
            STLT = min(100.0,SSTR)
            RIPEP = 1.0-STLT/100.0
            ATHLT = cultivar@HIMX*RIPEP
        }

        #Culculation of Spikelet Sterility Percentage due to Heat Stress
        if (DVI > 0.96  &  DVI < 1.20) {
            HTSUM = HTSUM+TMX[day]
            HTDAY = HTDAY+1
        }
        if (DVI >= 1.20  &  IFLUG1 == 0) {
            AVTMX = HTSUM/HTDAY
            STHT = 100.0/(1.0+exp(-0.853*(AVTMX-36.6)))
            ATHHT = (1.0-STHT/100.0)*cultivar@HIMX
            IFLUG1 = 1
        }

        #Culculation of Grain Yield
        ATHI = min(ATHLT,ATHHT)
        STERP = max(STHT,STLT)
        EFDVI = max(DVI-1.22, 0.0)
        HI = ATHI*(1.0-exp(-5.57*EFDVI))
        DWGRAIN = DW*HI
        DWPAN = DWGRAIN/CVBP/CVPP

        #Time Control and Terminal Condition of Simulation
        if (DVI > 1.0  &  AVT[day] < cultivar@CTR) {
            TCHECK = TCHECK+1
        }
        if (DVI > 2.0) {
            growing = FALSE
        }

    # simday <- simday + 1
    # res[simday,'date'] <- wth@w$date[day]
    # res[simday,'TMP'] <- AVT[day]
    # res[simday,'RAD'] <- RAD[day]
    # res[simday,'DL'] <- DL[day]
    # res[simday,'DVI'] <- DVI
    # res[simday,'LAI'] <- LAI
    # res[simday,'DW'] <- DW
    # res[simday,'GY'] <- DWGRAIN
    # res[simday,'PY'] <- DWPAN
    #
    # #Terminal Section of  Simulation
    # PYBROD <- DWGRAIN/100.0
    # PYBRON <- PYBROD/0.86
    # PYPADY <- PYBRON/CVBP
    # PANDW <- PYBROD/CVBP/CVPP
    # DWT <- DW/100.0
    #
    # r <- new('SIMRIW')
    # r@cultivar <- cultivar
    # r@PYBROD <- PYBROD
    # r@PYBRON <- PYBRON
    # r@PYPADY <- PYPADY
    # r@PANDW <- PANDW
    # r@DWT <- DWT
    # r@d <- res[1:simday,]
    #
    # return(r)
