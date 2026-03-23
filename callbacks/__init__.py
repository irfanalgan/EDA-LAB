from callbacks import (
    data_loading,
    precompute,
    preview,
    profiling,
    target_iv,
    outlier,
    deep_dive,
    correlation,
    stat_tests,
    var_summary,
    playground,
    results,
    help_overlay,
    profile,
)

# İzleme (Monitoring) callback'leri — tamamen bağımsız
from callbacks.izleme import nav as _mon_nav, data as _mon_data, profile as _mon_profile
from callbacks.izleme import tabs as _mon_tabs
