"""Reserved preview import boundary; no files or geometry are imported yet.

PreviewImportSession will validate versioned packed preview blocks and build
owned temporary collections in bounded main-thread steps. It must commit only
complete versions and preserve user-owned or edited data on cancellation.
The implementation is deferred until the core emits verified preview packages.
"""
