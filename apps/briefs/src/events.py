import requests
import os
from datetime import date
from pydantic import BaseModel, field_validator
from typing import Optional
import pandas as pd
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv

load_dotenv()


class Source(BaseModel):
    id: int
    name: str


class Event(BaseModel):
    id: int
    sourceId: int
    url: str
    title: str
    publishDate: datetime  # changed from date to datetime
    content: str
    location: str
    relevance: str
    completeness: str
    summary: str

    @field_validator("publishDate", mode="before")
    @classmethod
    def parse_date(cls, value):
        if value is None:
            return None

        # Handle ISO format with timezone info
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            # For older Python versions or non-standard formats
            # you might need dateutil
            from dateutil import parser

            return parser.parse(value)


def get_events(date: str = None):
    #url = f"https://meridian-production.alceos.workers.dev/events"
    url = f"http://localhost:8787/events"

    if date:
        url += f"?date={date}"

    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {os.environ.get('MERIDIAN_SECRET_KEY')}"},
    )
    data = response.json()
    # Inside events.py, around line 60:

    print("----- DEBUG RESPONSE -----") # Add this
    print(f"Status Code: {response.status_code}") # Add this
    print(f"Response Data: {data}") # Add this
    print("----- END DEBUG -----") # Add this

    sources = [Source(**source) for source in data["sources"]]
    events = [Event(**event) for event in data["events"]]

    return sources, events
