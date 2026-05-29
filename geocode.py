import json
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from geopy.geocoders import Nominatim


@dataclass(frozen=True)
class LocationResult:
    display_name: str
    city: str | None
    country: str | None
    country_code: str | None


def root():
    path = Path.home() / ".photo-archivist"
    path.mkdir(exist_ok=True)
    return path


def db():
    con = sqlite3.connect(root() / "geocode_cache.db")
    con.execute("create table if not exists geocode_cache (lat_rounded real, lon_rounded real, result_json text, primary key (lat_rounded, lon_rounded))")
    return con


def reverse_geocode(lat: float, lon: float) -> LocationResult | None:
    lat_r = round(lat, 4)
    lon_r = round(lon, 4)
    try:
        con = db()
        row = con.execute("select result_json from geocode_cache where lat_rounded = ? and lon_rounded = ?", (lat_r, lon_r)).fetchone()
        if row:
            return LocationResult(**json.loads(row[0]))
        time.sleep(1.1)
        place = Nominatim(user_agent="photo-archivist/1.0 (personal use)").reverse((lat_r, lon_r), exactly_one=True, timeout=10)
        if not place:
            return None
        address = place.raw.get("address", {})
        result = LocationResult(place.address, address.get("city") or address.get("town") or address.get("village"), address.get("country"), address.get("country_code"))
        con.execute("insert or replace into geocode_cache values (?, ?, ?)", (lat_r, lon_r, json.dumps(asdict(result))))
        con.commit()
        return result
    except Exception as e:
        logging.warning("reverse geocode failed for %s,%s: %s", lat, lon, e)
        return None
