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
        GEOIP_PATH="./geolite2_db", INSTALLED_APPS=["django.contrib.gis"]
    )
django.setup()

# Initialize GeoIP2 lookup service
geoIP = GeoIP2()

# Reference URLs for GeoIP2 setup and documentation
maxmind_geoip2_db_url = "https://www.maxmind.com/en/accounts/1263991/geoip/downloads"
geoip2_doc_url = "https://geoip2.readthedocs.io/en/latest/"
geoip2_django_doc_url = "https://docs.djangoproject.com/en/5.2/ref/contrib/gis/geoip2/"