import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

MINUTELY_15_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "shortwave_radiation",
    "shortwave_radiation_instant",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_direction_80m",
    "wind_speed_80m",
]


def fetch_openmeteo_forecast(
    latitude: float,
    longitude: float,
    forecast_days: int = 3,
    variables: list[str] | None = None,
) -> pd.DataFrame:
    """
    Fetch 15-minutely weather forecast from Open-Meteo API.

    Parameters
    ----------
    latitude : float
        Latitude in degrees.
    longitude : float
        Longitude in degrees.
    forecast_days : int
        Number of forecast days (default 3).
    variables : list[str] | None
        Minutely variables to request. Defaults to standard set.

    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index and weather columns.
    """
    if variables is None:
        variables = MINUTELY_15_VARIABLES

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "minutely_15": variables,
        "forecast_days": forecast_days,
    }
    responses = openmeteo.weather_api(url, params=params)

    response = responses[0]
    minutely_15 = response.Minutely15()

    data = {
        "date": pd.date_range(
            start=pd.to_datetime(minutely_15.Time(), unit="s", utc=True),
            end=pd.to_datetime(minutely_15.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=minutely_15.Interval()),
            inclusive="left",
        )
    }
    for i, var in enumerate(variables):
        data[var] = minutely_15.Variables(i).ValuesAsNumpy()

    return pd.DataFrame(data=data)


if __name__ == "__main__":
    df = fetch_openmeteo_forecast(latitude=52.52, longitude=13.41, forecast_days=3)
    print(df)
