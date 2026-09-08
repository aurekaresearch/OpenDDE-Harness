// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import { useEffect } from 'react'

import type { GatewayClient } from '../gatewayClientStub.js'
import type { ConfigFullResponse } from '../gatewayTypes.js'

import { resolveDetailsMode, resolveSections } from '../domain/details.js'
import { asRpcResult } from '../lib/rpc.js'
import {
  type BusyInputMode,
  DEFAULT_INDICATOR_STYLE,
  INDICATOR_STYLES,
  type IndicatorStyle,
  type StatusBarMode
} from './interfaces.js'
import { patchUiState } from './uiStore.js'

const STATUSBAR_ALIAS: Record<string, StatusBarMode> = {
  bottom: 'bottom',
  off: 'off',
  on: 'top',
  top: 'top'
}

export const normalizeStatusBar = (raw: unknown): StatusBarMode =>
  raw === false
    ? 'off'
    : raw === true
      ? 'top'
      : typeof raw === 'string'
        ? (STATUSBAR_ALIAS[raw.trim().toLowerCase()] ?? 'bottom')
        : 'bottom'

const BUSY_MODES = new Set<BusyInputMode>(['interrupt', 'queue'])

// `queue` rather than `interrupt`: in a full-screen TUI you're
// typically authoring the next prompt while the agent is still
// streaming, and an unintended interrupt loses work.  Set
// `display.busy_input_mode: interrupt` explicitly to opt out per-config.
const TUI_BUSY_DEFAULT: BusyInputMode = 'queue'

export const normalizeBusyInputMode = (raw: unknown): BusyInputMode => {
  if (typeof raw !== 'string') {
    return TUI_BUSY_DEFAULT
  }

  const v = raw.trim().toLowerCase() as BusyInputMode

  return BUSY_MODES.has(v) ? v : TUI_BUSY_DEFAULT
}

const INDICATOR_STYLE_SET: ReadonlySet<IndicatorStyle> = new Set(INDICATOR_STYLES)

export const normalizeIndicatorStyle = (raw: unknown): IndicatorStyle => {
  if (typeof raw !== 'string') {
    return DEFAULT_INDICATOR_STYLE
  }

  const v = raw.trim().toLowerCase() as IndicatorStyle

  return INDICATOR_STYLE_SET.has(v) ? v : DEFAULT_INDICATOR_STYLE
}

const FALSEY_MOUSE = new Set(['0', 'false', 'no', 'off'])
const hasOwn = (obj: object, key: PropertyKey) => Object.prototype.hasOwnProperty.call(obj, key)

export const normalizeMouseTracking = (display: { mouse_tracking?: unknown; tui_mouse?: unknown }): boolean => {
  const raw = hasOwn(display, 'mouse_tracking') ? display.mouse_tracking : display.tui_mouse

  if (raw === false || raw === 0) {
    return false
  }

  return typeof raw === 'string' ? !FALSEY_MOUSE.has(raw.trim().toLowerCase()) : true
}

const quietRpc = async <T extends object = Record<string, unknown>>(
  gw: GatewayClient,
  method: string,
  params: Record<string, unknown> = {}
): Promise<null | T> => {
  try {
    return asRpcResult<T>(await gw.request<T>(method, params))
  } catch {
    return null
  }
}

/** Fetch ``config.get full`` and fan the result through ``applyDisplay``.
 *
 * Extracted so the fetch/apply plumbing can be exercised by the test suite
 * without a React runtime, and a regression in it fails a test rather than only
 * showing up at runtime. */
export async function hydrateFullConfig(
  gw: GatewayClient,
  setBell: (v: boolean) => void
): Promise<ConfigFullResponse | null> {
  const cfg = await quietRpc<ConfigFullResponse>(gw, 'config.get', { key: 'full' })
  applyDisplay(cfg, setBell)

  return cfg
}

export const applyDisplay = (cfg: ConfigFullResponse | null, setBell: (v: boolean) => void) => {
  const d = cfg?.config?.display ?? {}

  setBell(!!d.bell_on_complete)

  patchUiState({
    busyInputMode: normalizeBusyInputMode(d.busy_input_mode),
    compact: !!d.tui_compact,
    detailsMode: resolveDetailsMode(d),
    detailsModeCommandOverride: false,
    indicatorStyle: normalizeIndicatorStyle(d.tui_status_indicator),
    inlineDiffs: d.inline_diffs !== false,
    // Only when the config says something. Every other field here would fall back
    // to the value the store already holds, but mouse tracking does not: its
    // default comes from OPENDDE_HARNESS_TUI_DISABLE_MOUSE, and normalizing a silent config
    // to `true` turned the environment's opt-out back on at startup.
    //
    // The conditional is what makes that safe, not the fetch: the request below
    // asks `config.get {key:'full'}` while the handler reads `keys` (plural), so
    // no display block is served today and this whole block runs on `{}`.
    ...(hasOwn(d, 'mouse_tracking') || hasOwn(d, 'tui_mouse') ? { mouseTracking: normalizeMouseTracking(d) } : {}),
    sections: resolveSections(d.sections),
    showCost: !!d.show_cost,
    showReasoning: d.show_reasoning !== false,
    statusBar: normalizeStatusBar(d.tui_statusbar),
    streaming: d.streaming !== false
    // NOTE: `transcript` is intentionally NOT synced here. It is a runtime-only
    // session flag toggled by `/transcript` (`display` is not a persisted config
    // block), so hydrating it would revert the user's choice.
  })
}

export function useConfigSync({ gw, setBellOnComplete, sid }: UseConfigSyncOptions) {
  useEffect(() => {
    if (!sid) {
      return
    }

    void hydrateFullConfig(gw, setBellOnComplete)
  }, [gw, setBellOnComplete, sid])
}

export interface UseConfigSyncOptions {
  gw: GatewayClient
  setBellOnComplete: (v: boolean) => void
  sid: null | string
}
