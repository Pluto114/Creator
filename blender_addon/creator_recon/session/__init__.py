"""Reserved host session boundary, with no running state machine yet.

AddonSession will own document generations and only this add-on's process
handles. VersionController will manage full preview versions and derived work
copies. Neither class is claimed implemented by the configuration scaffold.
See docs/architecture/04-runtime-and-blender.md for the reviewed contracts.
"""
