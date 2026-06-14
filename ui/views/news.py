"""News view — MLB news feed with on-demand scrape trigger."""

import requests
import streamlit as st

from ui.api.client import API_BASE
from ui.api.models import NewsAlert


def _scrape_news() -> dict | None:
    try:
        resp = requests.post(f"{API_BASE}/news/scrape", timeout=35)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error("Cannot reach the API at **http://localhost:8000**.")
    except Exception as exc:
        st.error(f"Scrape failed: {exc}")
    return None


def _load_recent(category: str | None) -> list[NewsAlert]:
    params: dict = {"limit": 50}
    if category:
        params["category"] = category
    try:
        resp = requests.get(f"{API_BASE}/news", params=params, timeout=10)
        resp.raise_for_status()
        return [NewsAlert.from_dict(a) for a in resp.json()]
    except Exception:
        return []


def render() -> None:
    st.header("MLB News Feed")

    with st.sidebar:
        st.subheader("News Filters")
        category_raw = st.selectbox(
            "Category",
            options=["All", "INJURY", "LINEUP", "TRANSACTION", "GENERAL"],
            index=0,
        )
        category_filter = None if category_raw == "All" else category_raw

        st.divider()
        if st.button("Scrape Latest News", use_container_width=True, type="primary"):
            with st.spinner("Scraping Rotowire, MLB.com, ESPN..."):
                result = _scrape_news()
            if result:
                st.success(
                    f"Done — {result['inserted']} new articles "
                    f"({result['skipped_duplicates']} duplicates skipped). "
                    f"Scraped in {result['scrape_duration_ms']}ms."
                )

    alerts = _load_recent(category_filter)

    if not alerts:
        st.info("No news stored yet. Click **Scrape Latest News** to fetch articles.")
        return

    for alert in alerts:
        source_badge = f"`{alert.source.upper()}`"
        ts = alert.published_at[:16].replace("T", " ") if alert.published_at else "—"
        teams_str = "  ·  ".join(alert.teams) if alert.teams else ""
        category_badge = f"`{alert.category}`" if alert.category else ""

        with st.container(border=True):
            col1, col2 = st.columns([8, 2])
            with col1:
                st.markdown(f"**{alert.headline}**")
                meta = f"{source_badge}  ·  {ts}"
                if category_badge:
                    meta += f"  ·  {category_badge}"
                if teams_str:
                    meta += f"  ·  {teams_str}"
                st.caption(meta)
            with col2:
                if alert.url and alert.url.startswith("http"):
                    st.link_button("Read", alert.url, use_container_width=True)
