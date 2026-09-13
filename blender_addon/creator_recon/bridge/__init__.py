"""Standard-library boundary to the future external core process.

Only read-only configuration validation exists today. RunProcessBridge remains
unimplemented until owned Windows process trees, cancellation, and atomic file
publication are verified. There is intentionally no bare subprocess fallback.
"""
