"""J-Quants provider — JPX's own data, and the recommended source.

Why this rather than Yahoo Finance:

*Point-in-time universe.*  ``/listed/info`` takes a ``date``, so the universe
can be reconstructed as it stood on any past day.  A backtest run against
today's listing list silently excludes every company that has since been taken
over or delisted.

*Split adjustment you can audit.*  Every bar carries an ``AdjustmentFactor``
and pre-computed ``Adjustment*`` prices, so corporate actions are explicit
rather than inferred.

*Unadjusted prices are available too.*  This matters more for technical work
than it first appears: a dividend-adjusted series shifts every historical
price, so a "52-week high" computed on it is not the level traders were
actually watching.  ``Open``/``High``/``Low``/``Close`` here are the raw traded
prices, and the ``Adjustment*`` columns handle splits separately.

*Limit flags.*  ``UpperLimit``/``LowerLimit`` mark bars that closed locked at
the daily price limit, which is exactly the information the simulator needs to
refuse a fill that was never available.

Authentication is two-step: a mail address and password buy a refresh token
(valid about a week), which buys an ID token (valid about a day).  Set
``JQUANTS_MAIL_ADDRESS`` and ``JQUANTS_PASSWORD``, or ``JQUANTS_REFRESH_TOKEN``
if you would rather not put the password in the environment.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterator, Sequence

import pandas as pd

from .base import empty_panel, normalise_panel

API_ROOT = "https://api.jquants.com/v1"

#: Free-plan history is short and delayed by 12 weeks; the paid tiers remove
#: the delay and extend the window (Light 5y, Standard 10y, Premium to 2008).
PLAN_NOTES = {
    "free": "2 years of history, delivered with a 12-week delay",
    "light": "5 years, no delay",
    "standard": "10 years, no delay",
    "premium": "full history from 2008-05-07, no delay",
}


class JQuantsError(RuntimeError):
    pass


class JQuantsProvider:
    name = "jquants"

    def __init__(
        self,
        mail_address: str | None = None,
        password: str | None = None,
        refresh_token: str | None = None,
        timeout: int = 60,
        pause: float = 0.2,
        retries: int = 3,
        use_adjusted: bool = True,
    ) -> None:
        self.mail_address = mail_address or os.environ.get("JQUANTS_MAIL_ADDRESS")
        self.password = password or os.environ.get("JQUANTS_PASSWORD")
        self._refresh_token = refresh_token or os.environ.get("JQUANTS_REFRESH_TOKEN")
        self._id_token: str | None = None
        self.timeout = timeout
        self.pause = pause
        self.retries = retries
        #: Split-adjusted prices make a long history continuous.  Turn this off
        #: only if you need the literally traded prices.
        self.use_adjusted = use_adjusted

    # ------------------------------------------------------------- auth

    def _get_refresh_token(self) -> str:
        if self._refresh_token:
            return self._refresh_token
        if not (self.mail_address and self.password):
            raise JQuantsError(
                "J-Quants credentials missing. Set JQUANTS_MAIL_ADDRESS and "
                "JQUANTS_PASSWORD (or JQUANTS_REFRESH_TOKEN) in the environment."
            )
        body = json.dumps(
            {"mailaddress": self.mail_address, "password": self.password}
        ).encode()
        data = self._request("POST", "/token/auth_user", body=body, authed=False)
        self._refresh_token = data["refreshToken"]
        return self._refresh_token

    def _get_id_token(self) -> str:
        if self._id_token:
            return self._id_token
        token = urllib.parse.quote(self._get_refresh_token(), safe="")
        data = self._request(
            "POST", f"/token/auth_refresh?refreshtoken={token}", authed=False
        )
        self._id_token = data["idToken"]
        return self._id_token

    # ---------------------------------------------------------- transport

    def _request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        authed: bool = True,
    ) -> dict[str, Any]:
        url = API_ROOT + path
        headers = {"Content-Type": "application/json"}
        if authed:
            headers["Authorization"] = f"Bearer {self._get_id_token()}"

        delay = self.pause
        last: Exception | None = None
        for attempt in range(self.retries):
            req = urllib.request.Request(url, data=body, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:400]
                if exc.code == 401 and authed and attempt < self.retries - 1:
                    # The ID token lives about a day; refresh once and retry.
                    self._id_token = None
                    headers["Authorization"] = f"Bearer {self._get_id_token()}"
                    last = exc
                elif exc.code in (429, 500, 502, 503, 504) and attempt < self.retries - 1:
                    last = exc
                else:
                    raise JQuantsError(f"{method} {path} -> HTTP {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = exc
                if attempt == self.retries - 1:
                    raise JQuantsError(f"{method} {path} failed: {exc}") from exc
            time.sleep(delay)
            delay *= 2
        raise JQuantsError(f"{method} {path} failed after {self.retries} attempts: {last}")

    def _paged(self, path: str, key: str, **params) -> Iterator[dict]:
        """Follow ``pagination_key`` until the server stops handing one back."""
        query = {k: v for k, v in params.items() if v is not None}
        while True:
            qs = urllib.parse.urlencode(query)
            data = self._request("GET", f"{path}?{qs}" if qs else path)
            yield from data.get(key, [])
            nxt = data.get("pagination_key")
            if not nxt:
                return
            query["pagination_key"] = nxt
            time.sleep(self.pause)

    # ------------------------------------------------------------ public

    def listed_info(self, date: pd.Timestamp | str | None = None) -> pd.DataFrame:
        """The listed universe, optionally as it stood on a past date.

        This is the survivorship-bias fix: pass the date a backtest bar belongs
        to and you get the companies that were actually listed then, including
        ones since delisted.
        """
        params = {}
        if date is not None:
            params["date"] = pd.Timestamp(date).strftime("%Y-%m-%d")
        rows = list(self._paged("/listed/info", "info", **params))
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        return df.rename(
            columns={
                "Code": "code_full",
                "CompanyName": "name",
                "MarketCodeName": "market_name",
                "Sector33CodeName": "sector33",
                "Sector17CodeName": "sector17",
                "ScaleCategory": "size_name",
                "Date": "as_of",
            }
        )

    def announcements(self) -> pd.DataFrame:
        """Upcoming earnings dates — used to avoid holding through a report."""
        rows = list(self._paged("/fins/announcement", "announcement"))
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def fetch(
        self,
        codes: Sequence[str],
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Daily bars for the given issue codes.

        Fetching by code rather than by date keeps the request count
        proportional to the universe instead of to the calendar, which is the
        cheaper direction for a multi-year first load.
        """
        codes = list(dict.fromkeys(codes))
        frames = []
        for code in codes:
            rows = list(
                self._paged(
                    "/prices/daily_quotes",
                    "daily_quotes",
                    code=code,
                    **{
                        "from": pd.Timestamp(start).strftime("%Y-%m-%d") if start is not None else None,
                        "to": pd.Timestamp(end).strftime("%Y-%m-%d") if end is not None else None,
                    },
                )
            )
            if rows:
                frames.append(self._to_panel(pd.DataFrame(rows), code))
            time.sleep(self.pause)
        if not frames:
            return empty_panel()
        return normalise_panel(pd.concat(frames, ignore_index=True))

    def fetch_by_date(self, date: pd.Timestamp | str) -> pd.DataFrame:
        """Every issue's bar for one date — the cheap shape for daily updates."""
        day = pd.Timestamp(date).strftime("%Y-%m-%d")
        rows = list(self._paged("/prices/daily_quotes", "daily_quotes", date=day))
        if not rows:
            return empty_panel()
        raw = pd.DataFrame(rows)
        return normalise_panel(self._to_panel(raw, None))

    def _to_panel(self, raw: pd.DataFrame, code: str | None) -> pd.DataFrame:
        prefix = "Adjustment" if self.use_adjusted else ""

        def col(field: str) -> pd.Series:
            for candidate in (f"{prefix}{field}", field):
                if candidate in raw.columns:
                    return raw[candidate]
            return pd.Series([None] * len(raw))

        # JPX issue codes are 5 characters in the API (a trailing 0 for common
        # stock); the rest of this package keys on the familiar 4-character form.
        codes = raw["Code"].astype(str) if "Code" in raw.columns else pd.Series([code] * len(raw))
        codes = codes.map(_short_code)

        return pd.DataFrame(
            {
                "code": codes,
                "date": raw["Date"],
                "open": col("Open"),
                "high": col("High"),
                "low": col("Low"),
                "close": col("Close"),
                "volume": col("Volume"),
            }
        )


def _short_code(code: str) -> str:
    """``"72030"`` -> ``"7203"``; leaves already-short codes alone."""
    code = str(code).strip()
    if len(code) == 5 and code.endswith("0"):
        return code[:-1]
    return code
