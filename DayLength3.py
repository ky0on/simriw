# -*- coding: utf-8 -*-
from math import floor, sin, cos, tan, asin, acos, atan, pi
import numpy as np
from datetime import datetime, timedelta

class SUN(object):
    def __init__(self, datetimeobj=datetime(2000,1,1), latitude=35.0, longitude=135.0):
        self.time = datetimeobj
        year = self.time.year
        month = self.time.month
        day = self.time.day
        self.d = (datetimeobj - datetime(year,month,day,0,0,0)).seconds / 86400.0
        self.longitude = longitude
        self.latitude = latitude
        self.rad = pi/180.0 #[deg]の変数にこいつを掛けると[rad]表現になる
        self.deg = 180.0/pi #[rad]の変数にこいつを掛けると[deg]表現になる


    def settime(self,datetimeobj):
        self.time = datetimeobj
        year = self.time.year
        month = self.time.month
        day = self.time.day
        self.d = (datetimeobj - datetime(year,month,day,0,0,0)).seconds / 86400.0


    def setlat(self,latitude):
        self.latitude = latitude

    def setlon(self,longitude):
        self.longitude = longitude
        

    def adjusttime(self):
        # 日を単位とする時刻d(0~1)をSunクラスの日付オブジェクトの字分秒に反映するメソッド
        year = self.time.year
        month = self.time.month
        day = self.time.day
        d = self.d
        self.time = datetime(year,month,day,0,0,0) + timedelta(seconds=d*86400.0)
        return self.time


    def julius_year(self):
        # 2000年1月1日正午(J2000.0と標記)からの経過日数Kをユリウス年(365.25日)で割ったもの
        # K：2000年1月1日正午(J2000.0と標記)からの経過日。
        # implementation of eqs.(3.13)(3.14)(3.15)
        year = self.time.year
        month = self.time.month
        day = self.time.day
        d = self.d
        if month == 1 or month == 2:
            year = year - 1
            month = month + 12
        kp = 365.0*(year-2000) + 30.0*month + day -33.875+ floor(3*(month+1)/5) + floor((year-2000)/4.0)
        dt = 65.0 + (year-2000)   #地球の自転はだんだん遅くなっている
        k = kp + d + dt/86400.0
        yjul = k/365.25
        return yjul


    def distance(self):
        # 太陽-地球間距離[A.U.]
        # implementation of r in table 3.11
        rad = self.rad
        t = self.julius_year()
        q = (0.007256 - 0.0000002*t)*sin((267.54 + 359.991*t)*rad)
        q += 0.000091*sin((265.1+ 719.98*t)*rad)
        q += 0.000030*sin((90.0)*rad)
        q += 0.000013*sin(( 27.8+ 4452.67*t)*rad)
        q += 0.000007*sin((254.0+  450.40*t)*rad)
        q += 0.000007*sin((156.0+  329.60*t)*rad)
        r = 10**q
        return r

    def visalradius(self):
        # 太陽視半径(太陽の見かけの大きさ)[deg]
        # implementation of eq.()
        return 0.266994 / self.distance()


    def lambda_s(self):
        # 太陽の視黄経[deg](0～360)
        # implementation of lambda_s in table 3.11
        rad = self.rad
        t = self.julius_year()
        l_s = 280.4603 + 360.00769*t
        l_s += (1.9146 - 0.00005*t)*sin((357.538+359.991*t)*rad)
        l_s += 0.0200*sin((355.05+719.981*t)*rad)
        l_s += 0.0048*sin((234.95+ 19.341*t)*rad)
        l_s += 0.0020*sin((247.1+  329.64*t)*rad)
        l_s += 0.0018*sin((297.8+ 4452.67*t)*rad)
        l_s += 0.0018*sin((251.3+    0.20*t)*rad)
        l_s += 0.0015*sin((343.2+  450.37*t)*rad)
        l_s += 0.0013*sin(( 81.4+  225.18*t)*rad)
        l_s += 0.0008*sin((132.5+  659.29*t)*rad)
        l_s += 0.0007*sin((153.3+   90.38*t)*rad)
        l_s += 0.0007*sin((206.8+   30.35*t)*rad)
        l_s += 0.0006*sin(( 29.8+  337.18*t)*rad)
        l_s += 0.0005*sin((207.4+    1.50*t)*rad)
        l_s += 0.0005*sin((291.2+   22.81*t)*rad)
        l_s += 0.0004*sin((234.9+  315.56*t)*rad)
        l_s += 0.0004*sin((157.3+  299.30*t)*rad)
        l_s += 0.0004*sin(( 21.1+  720.02*t)*rad)
        l_s += 0.0003*sin((352.5+ 1079.97*t)*rad)
        l_s += 0.0003*sin((329.7+   44.43*t)*rad)
        l_s = l_s % 360.0
        return l_s


    def epsilon(self):
        # 黄道傾斜角[deg]
        # implementation of eq.(3.17)
        t = self.julius_year()
        eps = 23.439291 - 0.000130042 * t
        return eps


    def alpha(self):
        # 赤経[deg](0～360)
        # implementation of eq.(3.16)
        rad = self.rad
        deg = self.deg
        eps = self.epsilon()*rad
        l_s = self.lambda_s()*rad
        alp = atan(tan(l_s)*cos(eps))*deg
        if l_s < 180.0*rad :
            alp = alp % 180
        else:
            if alp > 0.0 :
                alp = alp + 180.0
            else:
                alp = alp + 360.0
        return alp


    def delta(self):
        # 赤緯[deg]
        # implementation of eq.(3.16)
        rad = self.rad
        deg = self.deg
        eps = self.epsilon()*rad
        l_s = self.lambda_s()*rad
        dlt = asin(sin(l_s)*sin(eps))*deg
        return dlt


    def theta(self):
        # 地方恒星時[deg]
        # implementation of eq.(3.18)
        lon = self.longitude
        t = self.julius_year()
        d = self.d
        the = 325.4606 + 360.007700536*t + 0.00000003879*t*t + 360.0*d + lon
        the = the % 360.0
        return the


    def hour_angle(self):
        # implementation of eq.(2.18)
        # ---
        t = self.theta() - self.alpha()
        return t


    def elevation(self):
        #太陽高度角[deg](0～90)
        # implementation of eq.(1.20)
        deg = self.deg
        rad = self.rad
        dlt = self.delta()*rad
        t = self.hour_angle()*rad
        lat = self.latitude*rad
        sinh = sin(dlt)*sin(lat) + cos(dlt)*cos(lat)*cos(t)
        h = asin(sinh)*deg
        return h


    def azimus(self):
        #　太陽方位角[deg]
        # implementation of eq.(2.19)
        deg = self.deg
        rad = self.rad
        dlt = self.delta()*rad
        t = self.hour_angle()*rad
        lat = self.latitude*rad
        denominator = sin(dlt)*cos(lat)-cos(dlt)*sin(lat)*cos(t)
        if denominator > 0.0 :
            tanA =  -cos(dlt)*sin(t) / denominator
            azi = atan(tanA)*deg
        elif denominator < 0.0 :
            tanA =  -cos(dlt)*sin(t) / denominator
            azi = atan(tanA)*deg + 180.0
        else:
            if t > 0.0 :
                azi = -90.0
            elif t < 0.0 :
                azi = 90.0
            else:
                azi = np.nan
        azi = azi % 360.0
        return azi


    def parallax(self):
        #太陽中心が h[deg] にあるとき、見かけの高度角を得るために加えるべき視差角para[deg]を求めるメソッド。
        rad = self.rad
        h = self.elevation()
        r = self.distance()
        R = 0.0167/tan((h+8.6/(h+4.4))*rad)   #大気差(大気の影響による太陽の見かけの浮き上がり)[deg]
        p = 0.0024428/r     #視差(観測者が地球中心にいないことによる仰角の違い)[deg])
        para = R - p
        return  para


    def hour_angle_at_(self,h):
        # 太陽中心の高度角が h[deg] となる時角 ｔｈ[deg] を求めるメソッド
        # 二つある値のうち正のものだけを返す
        # implementation of eq.(2.17)
        deg = self.deg
        rad = self.rad
        dlt = self.delta()*rad
        lat = self.latitude*rad
        costh = ( sin(h*rad)-sin(dlt)*sin(lat) )/( cos(dlt)*cos(lat) )
        th = acos(costh)*deg
        return th


    def at_geometoric(self,h,isforenoon=True):
        """
        太陽の高度角がh[deg] となる時刻 t[1日で規格化した時刻] を求めるメソッド。
        引数：
           h：太陽高度角[deg]
           isforenoon：
               True:南中前の時刻を計算する
               False:南中後のの時刻を計算する。
        戻り値：
        　　計算時刻結果の時刻を、日を単位とする浮動小数で返す。また、Sunクラス
          の時刻オブジェクトをこの時刻に設定する。
        """
        if isforenoon :
            sig = -1
        else:
            sig = 1
        d = 0.5
        self.d = d
        diff = 0.0
        for i in range(10):
            thc = self.hour_angle()
            th = sig * abs(self.hour_angle_at_(h))
            diff = (th - thc) % 360.0 / 360.0
            if abs(diff) > 0.5 :
                diff = diff - 1.0
            d = (d + diff) % 1.0
            #print(i,d,diff,th,thc)
            if abs(diff) < 0.00005:
                break
            self.d = d
        self.adjusttime()
        return d

        
    def at_apparent(self,ha,isforenoon=True):
        """
        見かけの太陽高度が指定値になる時刻を求めるメソッド。大気差と視差を考慮する。
        引数：
           h：太陽高度角[deg]
           isforenoon[論理値]：
               True:南中前の時刻を計算する
               False:南中後のの時刻を計算する。
        戻り値：
        　　計算時刻結果の時刻を、日を単位とする浮動小数で返す。また、Sunクラス
          の時刻オブジェクトをこの時刻に設定する。
        """
        self.at_geometoric(ha,isforenoon)
        for i in range(10):
            d = self.d
            p = self.parallax()
            h = ha - p
            self.at_geometoric(h,isforenoon)
            diff = d - self.d
            if abs(diff) < 0.00005:
                break
        return d

        
    def at_meridian(self):
        """
        南中時刻を求めるメソッド。
        """
        d = 0.5
        self.d = d
        for i in range(10):
            tc = self.hour_angle()
            diff = - tc / 360.0
            if abs(diff) > 0.5 :
                diff = diff - 1.0
            d = (d + diff) % 1.0
            self.d = d
            if abs(diff) < 0.000002:
                break
        self.adjusttime()
        return self.d


    def at_sunrise(self):
        """
        日の出または日の入りの時刻を求めるメソッド。
        引数：
           第一引数：太陽高度角[deg]
           第一引数：文字列"forenoon"を指定すると日の出の時刻、
                   文字列"afternoon"を指定すると日の入りの時刻を計算する。
                   それ以外の文字列を与えると文句だけ言って終了する。
        戻り値：
        　　計算時刻結果の時刻を、日を単位とする浮動小数で返す。また、Sunクラス
          の時刻オブジェクトをこの時刻に設定する。
        """
        s = self.visalradius()
        p = 0.0024428 / self.distance()
        r = 0.585556
        h = - s - r + p
        d = self.at_geometoric(h,True)
        return d

    def at_sunset(self):
        """
        日の出または日の入りの時刻を求めるメソッド。
        引数：
           第一引数：太陽高度角[deg]
           第一引数：文字列"forenoon"を指定すると日の出の時刻、
                   文字列"afternoon"を指定すると日の入りの時刻を計算する。
                   それ以外の文字列を与えると文句だけ言って終了する。
        戻り値：
        　　計算時刻結果の時刻を、日を単位とする浮動小数で返す。また、Sunクラス
          の時刻オブジェクトをこの時刻に設定する。
        """
        s = self.visalradius()
        p = 0.0024428 / self.distance()
        r = 0.585556
        h = - s - r + p
        d = self.at_geometoric(h,False)
        return d


def daylength(tim,lat,lon,elevangle=np.nan ):
    """
概要：
    　長澤(1999)の式に基づいて、配列tim、lat、lonで指定される時空間範囲における
    日長[時間]の分布を配列で返す関数。
    長澤　工, 1999: 日の出日の入りの計算. 160p,　地人書館(東京).　
引数(必須)：
    tim：時間の格子点を定義するdatetimeobjectの配列。
    lat：緯度の格子点を定義する実数の配列
    lon：経度の格子点を定義する実数の配列
引数(必要に応じ)：
    elevangle：昼間とみなす太陽高度角[degree]
        正は仰角(日長は短くなる)、負は伏角(日長は長くなる)を示す。
        np.nanを与えると太陽高度角ではなく、太陽上端の出没に基づいて日長を計算する。
        この引数を省略した場合は太陽上端の出没に基づいて日長を計算する。
戻り値：
    引数で指定した時空間範囲における日長[時間]の3次元配列
改変履歴：
        20170601 クラスSunを導入し全面的に改訂
        20170518 コメント、文法の見直し
        20161201 Python Version 3 版作成
        20120629 初版作成
Copyright (C)  OHNO, Hiroyuki
    """
    print( "Calculating day lengh over the domain.")
    ntim = len(tim)
    nlat = len(lat)
    nlon = len(lon)
    dlarr = np.zeros((ntim, nlat, nlon))
    if np.isnan(elevangle):
        for j in range(nlat):
            for t in range(ntim):
                sun = SUN(tim[t],lat[j],lon[0])
                d1 = sun.at_sunrise()
                d2 = sun.at_sunset()
                dlarr[t,j,:] = (d2-d1)*24.0
    else:
        for j in range(nlat):
            for t in range(ntim):
                sun = SUN(tim[t],lat[j],lon[0])
                d1 = sun.at_apparent(elevangle,True)
                d2 = sun.at_apparent(elevangle,False)
                dlarr[t,j,:] = (d2-d1)*24.0
    return dlarr

        
if __name__ == '__main__':
    tim = [datetime(2017,1,1)+timedelta(days=oo) for oo in range(365)]  #2017年1月1日から12月31日までの日付オブジェクトのリストを作成
    lat = 35.6544    # [deg]
    lon = 139.7447   # [deg]

    # －－茨城県つくば市館野における2017年の日の出、南中、日の入りのグラフ－－
    sun = SUN()  #太陽オブジェクトの新規作成
    sun.setlat(lat) #観測地点の緯度の設定
    sun.setlon(lon) #観測地点の経度の設定
    sunrise = []   #日の出時刻を格納するための空のリスト
    sunset = []    #日の入時刻を格納するための空のリスト
    meridian = []  #南中時刻を格納するための空のリスト
    for dd in tim:  #1年分の値をループで計算する
        sun.settime(dd) #太陽観測日の設定
        t = sun.at_sunrise()*24. #メソッドat_sunrise()は、日の出時刻を24時間を1とする時刻で返す
        sunrise.append(t)        #リストに値を追加
        t = sun.at_sunset()*24. #メソッドat_sunset()は、日の出時刻を24時間を1とする時刻で返す
        sunset.append(t)         #リストに値を追加
        t = sun.at_meridian()*24. #メソッドat_meridian()は、南中時刻を24時間を1とする時刻で返す
        meridian.append(t)       #リストに値を追加
    print(sunrise)
    print(sunset)
    print(meridian)
