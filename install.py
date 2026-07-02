import launch

# Forge Neo already ships matplotlib; only install it if it is genuinely missing
# so we don't fight the host environment's pinned version.
if not launch.is_installed("matplotlib"):
    launch.run_pip("install matplotlib", desc="DAAM: installing matplotlib")
