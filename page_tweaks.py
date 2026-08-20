#!/usr/bin/env python
"""Post-processing tweaks applied to jlens readout pages after build_page().

Kept in one place so both collect_readouts.py (pages rendered on a fresh run)
and patch_pages.py (pages already written to disk) apply the exact same edit.
"""

from __future__ import annotations

# The jlens page renders three columns: #col0 (.heatmap-column, the main grid)
# plus two right-hand panels #col1/#col2 (.panel-column: by-layer, by-context)
# with drag handles between them. We hide the two panels + the column handles
# and let the heatmap reflow to full width. !important beats any inline widths
# the resize code sets. The row-resize handle inside the heatmap (.v-resize) is
# a different class and is left alone.
HIDE_PANELS_STYLE = (
    '<style id="jlens-hide-panels">'
    '.panel-column,.resize-handle{display:none!important}'
    '.heatmap-column{flex:1 1 auto!important;max-width:none!important}'
    '</style>'
)

# Marker used to detect an already-tweaked page (keeps apply_tweaks idempotent).
_MARKER = 'jlens-hide-panels'


def apply_tweaks(html: str) -> str:
    """Idempotently inject the page tweaks; returns the (possibly) edited html."""
    if _MARKER not in html and '</head>' in html:
        html = html.replace('</head>', HIDE_PANELS_STYLE + '</head>', 1)
    return html
