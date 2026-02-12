import requests
from datetime import datetime

DUCKLING_URL = "http://localhost:8000/parse"


def parse_dates_with_duckling(text: str):
    """
    Calls local Duckling server and extracts date ranges.
    Returns (start_datetime, end_datetime)
    """

    try:
        r = requests.post(
            DUCKLING_URL,
            data={
                "text": text,
                "locale": "en_US",
                "dims": '["time"]'
            },
            timeout=3,
        )

        if r.status_code != 200:
            return None, None

        results = r.json()

        for item in results:
            if item["dim"] != "time":
                continue

            val = item["value"]

            # interval (range)
            if val["type"] == "interval":
                start = val["from"]["value"]
                end = val["to"]["value"]

                return (
                    datetime.fromisoformat(start.replace("Z", "")),
                    datetime.fromisoformat(end.replace("Z", "")),
                )

            # single date
            if val["type"] == "value":
                dt = val["value"]
                dt = datetime.fromisoformat(dt.replace("Z", ""))
                return dt, dt

    except Exception as e:
        print("Duckling error:", e)

    return None, None
