import os
import sys

# Fixed by Claude: Set GDAL_LIBRARY_PATH before importing Django GIS
# Django looks for versioned DLLs (gdal311.dll) but pixi installs gdal.dll
gdal_dll_path = os.path.join(sys.prefix, 'Library', 'bin', 'gdal.dll')
if os.path.exists(gdal_dll_path):
    os.environ['GDAL_LIBRARY_PATH'] = gdal_dll_path

import django
from django.conf import settings
from django.contrib.gis.geoip2 import GeoIP2
import geoip2.database


# -----------------------------------------------------------------------------
# Django GeoIP2 Configuration
# Configure Django settings for IP geolocation using MaxMind GeoLite2 database
# The database must be downloaded from MaxMind and placed in ./geolite2_db/
# -----------------------------------------------------------------------------
if not settings.configured:
    settings.configure(
        GEOIP_PATH="./geolite2_db",
        INSTALLED_APPS=["django.contrib.gis"],
        GDAL_LIBRARY_PATH=gdal_dll_path if os.path.exists(gdal_dll_path) else None,
    )
django.setup()

# Initialize GeoIP2 lookup service
geoIP = GeoIP2()

# Reference URLs for GeoIP2 setup and documentation
maxmind_geoip2_db_url = "https://www.maxmind.com/en/accounts/1263991/geoip/downloads"
geoip2_doc_url = "https://geoip2.readthedocs.io/en/latest/"
geoip2_django_doc_url = "https://docs.djangoproject.com/en/5.2/ref/contrib/gis/geoip2/"