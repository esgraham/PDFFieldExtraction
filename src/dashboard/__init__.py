"""
Dashboard and analytics modules.

This package contains:
- Analytics engine for KPI calculation
- Dashboard web application
- Real-time metrics and reporting
"""

from .dashboard_analytics import AnalyticsEngine
from .dashboard_app import app

__all__ = [
    'AnalyticsEngine',
    'app'
]