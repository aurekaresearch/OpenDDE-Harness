// SPDX-License-Identifier: MIT
// Copyright (c) 2026 EverMind.
// See NOTICES.md.

import { type ArtScale, lockupSize, PANEL_FRAME_ROWS, resolveLockupScale } from '../banner.js'

export const FOOTER_HELP_TEXT = '/help for commands'
// Narrowest info column that still fits the section titles and footer.
export const INFO_MIN_COLS = 24
// Border (2) plus `paddingX={2}` on both sides.
const PANEL_CHROME_COLS = 6
// The factor-4 wordmark is too small to read, so the product name is spelled
// out on one line under it.
const NAME_LINE_FACTOR: ArtScale = 4

export type PanelSectionKey = 'mcp' | 'skills' | 'system' | 'tools'

export interface PanelSectionInput {
  // Rows the body takes when open.
  body: number
  // `null` means the section uses its default state; a boolean is the user's
  // explicit toggle, which the planner never overrides.
  open: boolean | null
  // An absent section costs no rows, but `show` still tracks the concession
  // chain for it, so a caller can gate on `show` and presence independently.
  present: boolean
}

export interface PanelPlanInput {
  // Rows the whole panel (frame included) may take.
  available: number
  cols: number
  footerMetaLength: number
  sections: Record<PanelSectionKey, PanelSectionInput>
}

export interface PanelPlan {
  // The working-directory line; among the first things to go when rows run out.
  cwd: boolean
  divider: boolean
  footerInline: boolean
  // 0 rows between blocks when the budget is tight, else 1.
  gap: 0 | 1
  infoRows: number
  // 0 means the brand is the one-line text form.
  lockup: ArtScale | 0
  lockupRows: number
  // The product name spelled out on its own line (text tier and factor 4).
  nameLine: boolean
  open: Record<PanelSectionKey, boolean>
  paddingY: 0 | 1
  // Rows reserved for the section area (leading gaps included); never fewer
  // than one line plus its gap, so the area holds a loader line before the
  // sections exist and keeps the same height once they do.
  sectionRows: number
  // Whether the concession chain kept the section; independent of `present`.
  show: Record<PanelSectionKey, boolean>
  // The version and date ride in the footer whenever the name has no line.
  versionInFooter: boolean
  width: number
}

const SECTION_ORDER: PanelSectionKey[] = ['tools', 'skills', 'system', 'mcp']
const DEFAULT_OPEN: Record<PanelSectionKey, boolean> = { mcp: false, skills: false, system: false, tools: true }

interface InfoState {
  cwd: boolean
  divider: boolean
  gap: 0 | 1
  open: Record<PanelSectionKey, boolean>
  paddingY: 0 | 1
  show: Record<PanelSectionKey, boolean>
}

// Successive concessions, cheapest first. A section the user opened by hand is
// exempt from every step, so a resize never undoes an explicit click.
const degradeSteps: ((state: InfoState, sections: PanelPlanInput['sections']) => InfoState | null)[] = [
  s => (s.gap ? { ...s, gap: 0 } : null),
  s => (s.paddingY ? { ...s, paddingY: 0 } : null),
  (s, sec) => (s.show.system && sec.system.open !== true ? { ...s, show: { ...s.show, system: false } } : null),
  (s, sec) => (s.show.mcp && sec.mcp.open !== true ? { ...s, show: { ...s.show, mcp: false } } : null),
  s => (s.cwd ? { ...s, cwd: false } : null),
  s => (s.divider ? { ...s, divider: false } : null),
  (s, sec) => (s.open.skills && sec.skills.open !== true ? { ...s, open: { ...s.open, skills: false } } : null),
  (s, sec) => (s.open.tools && sec.tools.open !== true ? { ...s, open: { ...s.open, tools: false } } : null),
  (s, sec) => (s.show.skills && sec.skills.open !== true ? { ...s, show: { ...s.show, skills: false } } : null),
  (s, sec) => (s.show.tools && sec.tools.open !== true ? { ...s, show: { ...s.show, tools: false } } : null)
]

// The brand outranks spacing, the system and mcp sections, the divider and the
// working-directory line, but never the tools and skills sections: a state past
// this one may only be reached to fit the text brand.
const LOCKUP_STATE_LIMIT = 6

const initialState = (sections: PanelPlanInput['sections']): InfoState => {
  const show = {} as Record<PanelSectionKey, boolean>
  const open = {} as Record<PanelSectionKey, boolean>

  for (const key of SECTION_ORDER) {
    show[key] = true
    open[key] = sections[key].open ?? DEFAULT_OPEN[key]
  }

  return { cwd: true, divider: true, gap: 1, open, paddingY: 1, show }
}

const sectionRowsOf = (state: InfoState, sections: PanelPlanInput['sections']) => {
  let rows = 0

  for (const key of SECTION_ORDER) {
    if (state.show[key] && sections[key].present) {
      rows += state.gap + 1 + (state.open[key] ? sections[key].body : 0)
    }
  }

  return Math.max(state.gap + 1, rows)
}

const infoRowsOf = (state: InfoState, sections: PanelPlanInput['sections'], footerRows: number) => {
  const divider = state.divider ? state.gap + 1 : 0
  const cwd = state.cwd ? state.gap + 1 : 0

  return sectionRowsOf(state, sections) + divider + footerRows + cwd
}

const frameRows = (state: InfoState) => PANEL_FRAME_ROWS - 2 + 2 * state.paddingY

const footerRowsFor = (width: number, metaLength: number) => (FOOTER_HELP_TEXT.length + 2 + metaLength <= width ? 1 : 2)

// Rows the brand block costs at a scale factor, name line included.
export const brandRows = (lockup: ArtScale | 0) =>
  lockup === 0 ? 1 : lockupSize(lockup).rows + (lockup === NAME_LINE_FACTOR ? 1 : 0)

export function planSessionPanel(input: PanelPlanInput): PanelPlan {
  const available = Math.max(PANEL_FRAME_ROWS, input.available)
  const width = Math.max(INFO_MIN_COLS, input.cols - PANEL_CHROME_COLS)
  const footerRows = footerRowsFor(width, input.footerMetaLength)
  const states: InfoState[] = [initialState(input.sections)]

  for (const step of degradeSteps) {
    const next = step(states[states.length - 1]!, input.sections)

    if (next) {
      states.push(next)
    }
  }

  const fits = (state: InfoState, lockup: ArtScale | 0) =>
    frameRows(state) + brandRows(lockup) + state.gap + infoRowsOf(state, input.sections, footerRows) <= available

  const finish = (state: InfoState, lockup: ArtScale | 0): PanelPlan => ({
    cwd: state.cwd,
    divider: state.divider,
    footerInline: footerRows === 1,
    gap: state.gap,
    infoRows: infoRowsOf(state, input.sections, footerRows),
    lockup,
    lockupRows: lockup === 0 ? 0 : lockupSize(lockup).rows,
    nameLine: lockup === 0 || lockup === NAME_LINE_FACTOR,
    open: state.open,
    paddingY: state.paddingY,
    sectionRows: sectionRowsOf(state, input.sections),
    show: state.show,
    versionInFooter: lockup !== 0 && lockup !== NAME_LINE_FACTOR,
    width
  })

  // Largest lockup the columns hold, then the roomiest section layout that
  // leaves it the rows. Anything smaller than the tools and skills sections is
  // reserved for the text brand.
  const widest = resolveLockupScale(available, width)

  if (widest !== 0) {
    for (const lockup of [1, 2, 4] as ArtScale[]) {
      if (lockupSize(lockup).cols > width) {
        continue
      }

      const state = states.slice(0, LOCKUP_STATE_LIMIT + 1).find(s => fits(s, lockup))

      if (state) {
        return finish(state, lockup)
      }
    }
  }

  return finish(states.find(s => fits(s, 0)) ?? states[states.length - 1]!, 0)
}
