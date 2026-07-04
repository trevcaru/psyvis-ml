"""Pytest configuration: force a non-interactive matplotlib backend for plot tests."""

import matplotlib

matplotlib.use("Agg")
