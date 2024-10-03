# Adapted from
# https://github.com/blender/blender-addons/blob/bb16aba5bd3873794eefe167497118b6063b9a85/sun_position/sun_calc.py

import bpy
import datetime
import math

from math import degrees, radians, pi
from mathutils import *


class SunCalc():
    def __init__(
            self,
            latitude=45.939,
            longitude=7.866,
            utc_zone=1,
            year=2021,
            month=7,
            day=1):
        self.latitude = latitude
        self.longitude = longitude
        self.utc_zone = utc_zone
        self.year = year
        self.month = month
        self.day = day

    def get_sun_position(self, distance=50):
        locX = math.sin(self.phi) * math.sin(-self.theta) * distance
        locY = math.sin(self.theta) * math.cos(self.phi) * distance
        locZ = math.cos(self.theta) * distance
        return Vector((locX, locY, locZ))

    def get_sun_pose(self, local_time):
        self.longitude *= -1                 # for internal calculations
        utc_time = local_time + self.utc_zone   # Set Greenwich Meridian Time

        self.latitude = radians(self.latitude)

        t = self.julian_time_from_y2k(
            utc_time, self.year, self.month, self.day)

        e = radians(self.obliquity_correction(t))
        L = self.apparent_longitude_of_sun(t)
        solar_dec = self.sun_declination(e, L)
        eqtime = self.calc_equation_of_time(t)

        time_correction = (eqtime - 4 * self.longitude) + 60 * self.utc_zone
        true_solar_time = ((utc_time - self.utc_zone) *
                           60.0 + time_correction) % 1440

        hour_angle = true_solar_time / 4.0 - 180.0
        if hour_angle < -180.0:
            hour_angle += 360.0

        csz = (math.sin(self.latitude) * math.sin(solar_dec) +
               math.cos(self.latitude) * math.cos(solar_dec) *
               math.cos(radians(hour_angle)))
        if csz > 1.0:
            csz = 1.0
        elif csz < -1.0:
            csz = -1.0

        zenith = math.acos(csz)

        az_denom = math.cos(self.latitude) * math.sin(zenith)

        if abs(az_denom) > 0.001:
            az_rad = ((math.sin(self.latitude) *
                       math.cos(zenith)) - math.sin(solar_dec)) / az_denom
            if abs(az_rad) > 1.0:
                az_rad = -1.0 if (az_rad < 0.0) else 1.0
            azimuth = 180.0 - degrees(math.acos(az_rad))
            if hour_angle > 0.0:
                azimuth = -azimuth
        else:
            azimuth = 180.0 if (self.latitude > 0.0) else 0.0

        if azimuth < 0.0:
            azimuth = azimuth + 360.0

        exoatm_elevation = 90.0 - degrees(zenith)

        if exoatm_elevation > 85.0:
            refraction_correction = 0.0
        else:
            te = math.tan(radians(exoatm_elevation))
            if exoatm_elevation > 5.0:
                refraction_correction = (
                    58.1 / te - 0.07 / (te ** 3) + 0.000086 / (te ** 5))
            elif (exoatm_elevation > -0.575):
                s1 = (-12.79 + exoatm_elevation * 0.711)
                s2 = (103.4 + exoatm_elevation * (s1))
                s3 = (-518.2 + exoatm_elevation * (s2))
                refraction_correction = 1735.0 + exoatm_elevation * (s3)
            else:
                refraction_correction = -20.774 / te

        refraction_correction = refraction_correction / 3600
        solar_elevation = 90.0 - (degrees(zenith) - refraction_correction)

        solar_azimuth = azimuth
        north_offset = 0.0
        solar_azimuth += north_offset

        self.az_north = solar_azimuth

        self.theta = math.pi / 2 - radians(solar_elevation)
        self.phi = radians(solar_azimuth) * -1
        self.azimuth = azimuth
        self.elevation = solar_elevation

        return Euler((radians(self.elevation - 90),
                      0, radians(-self.az_north)))

    def julian_time_from_y2k(self, utc_time, year, month, day):
        century = 36525.0  # Days in Julian Century
        epoch = 2451545.0  # Julian Day for 1/1/2000 12:00 gmt
        jd = self.get_julian_day(year, month, day)
        return ((jd + (utc_time / 24)) - epoch) / century

    def get_julian_day(self, year, month, day):
        if month <= 2:
            year -= 1
            month += 12
        A = math.floor(year / 100)
        B = 2 - A + math.floor(A / 4.0)
        jd = (math.floor((365.25 * (year + 4716.0))) +
              math.floor(30.6001 * (month + 1)) + day + B - 1524.5)
        return jd

    def obliquity_correction(self, t):
        ec = self.obliquity_of_ecliptic(t)
        omega = 125.04 - 1934.136 * t
        return (ec + 0.00256 * math.cos(radians(omega)))

    def obliquity_of_ecliptic(self, t):
        return ((23.0 + 26.0 / 60 + (21.4480 - 46.8150) / 3600 * t -
                 (0.00059 / 3600) * t ** 2 + (0.001813 / 3600) * t ** 3))

    def true_longitude_of_sun(self, t):
        return (self.mean_longitude_sun(t) + self.equation_of_sun_center(t))

    def calc_sun_apparent_long(self, t):
        o = self.true_longitude_of_sun(t)
        omega = 125.04 - 1934.136 * t
        lamb = o - 0.00569 - 0.00478 * math.sin(radians(omega))
        return lamb

    def apparent_longitude_of_sun(self, t):
        return (radians(self.true_longitude_of_sun(t) - 0.00569 - 0.00478 *
                        math.sin(radians(125.04 - 1934.136 * t))))

    def mean_longitude_sun(self, t):
        return (280.46646 + 36000.76983 * t + 0.0003032 * t ** 2) % 360

    def equation_of_sun_center(self, t):
        m = radians(self.mean_anomaly_sun(t))
        c = ((1.914602 - 0.004817 * t - 0.000014 * t ** 2) * math.sin(m) +
             (0.019993 - 0.000101 * t) * math.sin(m * 2) +
             0.000289 * math.sin(m * 3))
        return c

    def mean_anomaly_sun(self, t):
        return (357.52911 + t * (35999.05029 - 0.0001537 * t))

    def eccentricity_earth_orbit(self, t):
        return (0.016708634 - 0.000042037 * t - 0.0000001267 * t ** 2)

    def sun_declination(self, e, L):
        return (math.asin(math.sin(e) * math.sin(L)))

    def calc_equation_of_time(self, t):
        epsilon = self.obliquity_correction(t)
        ml = radians(self.mean_longitude_sun(t))
        e = self.eccentricity_earth_orbit(t)
        m = radians(self.mean_anomaly_sun(t))
        y = math.tan(radians(epsilon) / 2.0)
        y = y * y
        sin2ml = math.sin(2.0 * ml)
        cos2ml = math.cos(2.0 * ml)
        sin4ml = math.sin(4.0 * ml)
        sinm = math.sin(m)
        sin2m = math.sin(2.0 * m)
        etime = (y * sin2ml - 2.0 * e * sinm + 4.0 * e * y *
                 sinm * cos2ml - 0.5 * y ** 2 * sin4ml - 1.25 * e ** 2 * sin2m)
        return (degrees(etime) * 4)
