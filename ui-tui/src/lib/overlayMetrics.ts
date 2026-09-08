// SPDX-License-Identifier: MIT
// Copyright (c) 2026 EverMind.
// See NOTICES.md.

// Rows a pane overlay cannot spend on its list: the pane's top padding and
// border, the overlay's own title, spacer and hint rows, a row for an error or
// a "more" marker, and the composer area that stays below the pane (spacer,
// overlay anchor and status bar).
export const OVERLAY_PANE_CHROME_ROWS = 13

export const overlayPaneRows = (rows: number) => Math.max(5, rows - OVERLAY_PANE_CHROME_ROWS)
