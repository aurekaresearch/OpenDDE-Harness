// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Portions Copyright (c) 2026 mrzz (pi-claude-theme, MIT): the neutral greys,
// the semantic green/red/amber, the periwinkle, and the message and tool
// grounds are that theme's values.
// Modifications Copyright (c) 2026 EverMind.
// See LICENSES/README.md and LICENSES/MIT-hermes-agent.txt.
//
// One curated set per scheme x color tier, so a 256-color terminal gets
// hand-picked `ansi256(N)` values instead of chalk's lossy hex downsample.
// Every color the UI draws goes through a `Theme` instance; components never
// emit raw ANSI.
//
// The split the palette is built on: violet is identity, everything else is
// neutral. The brand violet paints the logo, the accent and primary, the
// picker's selected row, the busy indicator and the thinking level, and
// nothing else. Borders, rules, popups, the footer and the message and tool
// grounds are grey at the reference theme's own levels; links and field labels
// take a second hue (periwinkle) so a reference is not mistaken for a brand
// accent; green, red and amber mean success, failure and warning and are never
// chrome. Text is left unpainted wherever it can be, so it is the terminal's
// own foreground.

import type { EditorTheme, MarkdownTheme, RgbColor, SelectListTheme, SettingsListTheme } from '@earendil-works/pi-tui'
import type { ChalkInstance } from 'chalk'

import { Chalk } from 'chalk'

import type { ColorTier } from './lib/colorTier.js'

import { resolveColorTier } from './lib/colorTier.js'

export type ColorScheme = 'dark' | 'light'

/** Paint nothing: the terminal's own color stands. Used for body text and for
 *  the footer's ground, which both read better in whatever the user's profile
 *  already chose than in anything this palette could pick for them. */
export const INHERIT = 'inherit'

export interface ThemeColors {
  primary: string
  accent: string
  /** The accent one light step up: the band of the busy indicator's shimmer,
   *  the way Claude Code tints its verb. */
  shimmer: string
  border: string
  text: string
  muted: string

  label: string
  ok: string
  error: string
  warn: string

  prompt: string
  /** pi's `dim`: the grey a footer, a hint's key or a URL is painted in, one
   *  step below `muted`. Kept at pi's own two values. */
  dim: string

  selectionBg: string
  userMessageBg: string
  toolPendingBg: string
  toolSuccessBg: string
  toolErrorBg: string
}

export type ThemeToken = keyof ThemeColors

// ── Tier 3: truecolor (source of truth) ──────────────────────────────

const DARK_TRUECOLOR: ThemeColors = {
  // Identity.
  primary: '#A78BFA',
  accent: '#8B5CF6',
  shimmer: '#A97AFF',
  prompt: '#A78BFA',
  selectionBg: '#3A3350',

  // Text: the terminal's own, with grey for what is deliberately secondary.
  text: INHERIT,
  muted: '#999999',
  label: '#8FA0F0',
  dim: '#666666',

  // Chrome.
  border: '#505050',

  // Meaning.
  ok: '#4EBA65',
  error: '#FF6B80',
  warn: '#FFC107',

  // Grounds. Flat greys: a band is a ground, and success is said by what is in
  // the panel rather than by tinting it.
  userMessageBg: '#373737',
  toolPendingBg: '#2E2E2E',
  toolSuccessBg: '#2E2E2E',
  toolErrorBg: '#362B2D'
}

// Light-terminal palette: the same structure at the reference's light levels.
const LIGHT_TRUECOLOR: ThemeColors = {
  primary: '#7C3AED',
  accent: '#6D28D9',
  shimmer: '#8B46F7',
  prompt: '#7C3AED',
  selectionBg: '#E9E4F7',

  text: INHERIT,
  muted: '#666666',
  label: '#4B5BD4',
  dim: '#767676',

  border: '#AFAFAF',

  ok: '#2C7A39',
  error: '#AB2B3F',
  warn: '#966C1E',

  userMessageBg: '#F0F0F0',
  toolPendingBg: '#F5F5F5',
  toolSuccessBg: '#F5F5F5',
  toolErrorBg: '#F9EFEF'
}

// ── Tier 2: 256-color ────────────────────────────────────────────────
//
// Nearest slot to each truecolor value, except where two roles would land on
// the same grey: the picker's selected row keeps a violet-tinted ground rather
// than collapsing onto the user band's, and a failed tool keeps a red one.

const DARK_256: ThemeColors = {
  primary: 'ansi256(141)',
  accent: 'ansi256(99)',
  shimmer: 'ansi256(141)',
  prompt: 'ansi256(141)',
  selectionBg: 'ansi256(60)',

  text: INHERIT,
  muted: 'ansi256(246)',
  label: 'ansi256(111)',
  dim: 'ansi256(241)',

  border: 'ansi256(240)',

  ok: 'ansi256(71)',
  error: 'ansi256(204)',
  warn: 'ansi256(214)',

  userMessageBg: 'ansi256(237)',
  toolPendingBg: 'ansi256(236)',
  toolSuccessBg: 'ansi256(236)',
  toolErrorBg: 'ansi256(52)'
}

const LIGHT_256: ThemeColors = {
  primary: 'ansi256(93)',
  accent: 'ansi256(92)',
  shimmer: 'ansi256(99)',
  prompt: 'ansi256(93)',
  selectionBg: 'ansi256(189)',

  text: INHERIT,
  muted: 'ansi256(241)',
  label: 'ansi256(62)',
  dim: 'ansi256(243)',

  border: 'ansi256(145)',

  ok: 'ansi256(29)',
  error: 'ansi256(125)',
  warn: 'ansi256(94)',

  userMessageBg: 'ansi256(255)',
  toolPendingBg: 'ansi256(255)',
  toolSuccessBg: 'ansi256(255)',
  toolErrorBg: 'ansi256(224)'
}

// ── Tier 1: 16-color ─────────────────────────────────────────────────
//
// The same split with eight hues to spend it on: magenta is the identity,
// every piece of chrome is one of the two blacks or the terminal's own, and no
// ground is tinted toward the brand. A reverse highlight has no color to fall
// back to, so the picker's selected row is told apart by its accent text and
// its cursor rather than by a band.

const DARK_16: ThemeColors = {
  primary: 'ansi:magentaBright',
  accent: 'ansi:magentaBright',
  shimmer: 'ansi:white',
  prompt: 'ansi:magentaBright',
  selectionBg: 'ansi:blackBright',

  text: INHERIT,
  muted: 'ansi:blackBright',
  label: 'ansi:blueBright',
  dim: 'ansi:blackBright',

  border: 'ansi:blackBright',

  ok: 'ansi:greenBright',
  error: 'ansi:redBright',
  warn: 'ansi:yellow',

  userMessageBg: 'ansi:blackBright',
  toolPendingBg: 'ansi:black',
  toolSuccessBg: 'ansi:black',
  toolErrorBg: 'ansi:red'
}

const LIGHT_16: ThemeColors = {
  primary: 'ansi:magenta',
  accent: 'ansi:magenta',
  shimmer: 'ansi:magentaBright',
  prompt: 'ansi:magenta',
  selectionBg: 'ansi:white',

  text: INHERIT,
  muted: 'ansi:blackBright',
  label: 'ansi:blue',
  dim: 'ansi:blackBright',

  border: 'ansi:blackBright',

  ok: 'ansi:green',
  error: 'ansi:red',
  warn: 'ansi:yellow',

  userMessageBg: 'ansi:white',
  toolPendingBg: 'ansi:white',
  toolSuccessBg: 'ansi:white',
  toolErrorBg: 'ansi:redBright'
}

// ── The wordmark ramp ────────────────────────────────────────────────
//
// The four bands the startup wordmark is drawn in, top to bottom, and the only
// color the banner art is allowed to use. It is a ramp rather than a token
// because the wordmark is a gradient: its rows are split into as many bands as
// this has entries.
//
// Every band is a violet. The ramp used to open at the brand scale's lightest
// stops (#F5F3FF, #EDE9FE), which read as white rather than as the brand on a
// dark terminal, and to close on its darkest (#3B0764, #1E103F), which read as
// black on a light one. Both ends now stay inside the violet family, so the
// gradient reads as one color getting deeper and never as white-to-violet or
// violet-to-black.
//
// The light ramp sits a step deeper than the dark one: the same violet that is
// comfortable on a near-black ground is pale on a white one. Its lightest band
// measures 4.2:1 against white and its darkest 9.0:1; on dark the range is
// 3.0:1 to 9.3:1 against #1b1b1b. Nothing is near either ground.

const WORDMARK_RAMP_TRUECOLOR_DARK: readonly string[] = ['#C4B5FD', '#A78BFA', '#8B5CF6', '#7C3AED']

const WORDMARK_RAMP_TRUECOLOR_LIGHT: readonly string[] = ['#8B5CF6', '#7C3AED', '#6D28D9', '#5B21B6']

const WORDMARK_RAMP_256_DARK: readonly string[] = ['ansi256(183)', 'ansi256(141)', 'ansi256(99)', 'ansi256(93)']

const WORDMARK_RAMP_256_LIGHT: readonly string[] = ['ansi256(99)', 'ansi256(93)', 'ansi256(56)', 'ansi256(55)']

/** Eight hues have no gradient to give: every band is the one violet there is. */
const WORDMARK_RAMP_16_DARK: readonly string[] = Array.from({ length: 4 }, () => 'ansi:magentaBright')

const WORDMARK_RAMP_16_LIGHT: readonly string[] = Array.from({ length: 4 }, () => 'ansi:magenta')

/** The wordmark's bands for a scheme + tier, lightest first. */
export function resolveBrandRamp(scheme: ColorScheme, tier: ColorTier): readonly string[] {
  if (tier === 1) {
    return scheme === 'light' ? WORDMARK_RAMP_16_LIGHT : WORDMARK_RAMP_16_DARK
  }

  if (tier === 2) {
    return scheme === 'light' ? WORDMARK_RAMP_256_LIGHT : WORDMARK_RAMP_256_DARK
  }

  return scheme === 'light' ? WORDMARK_RAMP_TRUECOLOR_LIGHT : WORDMARK_RAMP_TRUECOLOR_DARK
}

/**
 * Pick the palette for a scheme + color tier. Tier 3 (truecolor) and tier 0
 * (no color — chalk strips the codes anyway) share the hex palette.
 */
export function resolvePalette(scheme: ColorScheme, tier: ColorTier): ThemeColors {
  if (scheme === 'light') {
    return tier === 2 ? LIGHT_256 : tier === 1 ? LIGHT_16 : LIGHT_TRUECOLOR
  }

  return tier === 2 ? DARK_256 : tier === 1 ? DARK_16 : DARK_TRUECOLOR
}

// ── Light/dark detection ─────────────────────────────────────────────

const TRUE_RE = /^(?:1|true|yes|on)$/
const FALSE_RE = /^(?:0|false|no|off)$/
const HEX_3_RE = /^[0-9a-f]{3}$/
const HEX_6_RE = /^[0-9a-f]{6}$/

// Rec. 709 luma above this reads as a light background.
const LUMA_LIGHT_THRESHOLD = 0.6

// TERM_PROGRAM allow-list for terminals whose default profile is light and
// which may not expose COLORFGBG. Empty in production: TERM_PROGRAM alone
// can't tell a light profile from a dark one (Terminal.app ships both and
// emits no COLORFGBG either way), so an undetectable terminal stays dark.
// Injectable so the precedence rules stay testable.
const LIGHT_DEFAULT_TERM_PROGRAMS = new Set<string>([])

function luminance(rgb: RgbColor): number {
  return (0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b) / 255
}

/** Parse a 3- or 6-digit hex string into RGB. Strict: `parseInt` truncates at
 *  the first non-hex character, so `fffgff` would otherwise read as white. */
function parseHex(raw: string): RgbColor | null {
  const v = raw.trim().toLowerCase()
  const hex = v.startsWith('#') ? v.slice(1) : v

  if (HEX_6_RE.test(hex)) {
    return {
      r: parseInt(hex.slice(0, 2), 16),
      g: parseInt(hex.slice(2, 4), 16),
      b: parseInt(hex.slice(4, 6), 16)
    }
  }

  if (HEX_3_RE.test(hex)) {
    return {
      r: parseInt(hex[0]! + hex[0]!, 16),
      g: parseInt(hex[1]! + hex[1]!, 16),
      b: parseInt(hex[2]! + hex[2]!, 16)
    }
  }

  return null
}

/** The scheme the user asked for outright, or null when they said nothing.
 *  `OPENDDE_HARNESS_TUI_LIGHT` wins over `OPENDDE_HARNESS_TUI_THEME`; both win
 *  over anything measured, including a live OSC 11 reply. */
export function explicitScheme(env: NodeJS.ProcessEnv): ColorScheme | null {
  const lightFlag = (env.OPENDDE_HARNESS_TUI_LIGHT ?? '').trim().toLowerCase()

  if (TRUE_RE.test(lightFlag)) {
    return 'light'
  }

  if (FALSE_RE.test(lightFlag)) {
    return 'dark'
  }

  const themeFlag = (env.OPENDDE_HARNESS_TUI_THEME ?? '').trim().toLowerCase()

  if (themeFlag === 'light' || themeFlag === 'dark') {
    return themeFlag
  }

  return null
}

/**
 * Light or dark, from ordered signals:
 *   1. `OPENDDE_HARNESS_TUI_LIGHT` / `OPENDDE_HARNESS_TUI_THEME` (explicit).
 *   2. `OPENDDE_HARNESS_TUI_BACKGROUND` hex hint (3- or 6-digit).
 *   3. `COLORFGBG` last field — slot 7/15 is light; any other 0-15 slot is
 *      authoritatively dark, so the allow-list below cannot override it.
 *   4. `TERM_PROGRAM` light-default allow-list (empty by default).
 * Anything undecidable stays dark — the default palette is the dark one.
 */
export function detectScheme(
  env: NodeJS.ProcessEnv,
  lightDefaultTermPrograms: ReadonlySet<string> = LIGHT_DEFAULT_TERM_PROGRAMS
): ColorScheme {
  const explicit = explicitScheme(env)

  if (explicit) {
    return explicit
  }

  const hint = parseHex(env.OPENDDE_HARNESS_TUI_BACKGROUND ?? '')

  if (hint) {
    return luminance(hint) >= LUMA_LIGHT_THRESHOLD ? 'light' : 'dark'
  }

  const colorfgbg = (env.COLORFGBG ?? '').trim()

  if (colorfgbg) {
    // Validate as a decimal integer first: `Number('')` is 0, so a malformed
    // `COLORFGBG='15;'` would otherwise look like an authoritative dark slot.
    const lastField = colorfgbg.split(';').at(-1) ?? ''

    if (/^\d+$/.test(lastField)) {
      const bg = Number(lastField)

      if (bg === 7 || bg === 15) {
        return 'light'
      }

      if (bg >= 0 && bg < 16) {
        return 'dark'
      }
    }
  }

  return lightDefaultTermPrograms.has((env.TERM_PROGRAM ?? '').trim()) ? 'light' : 'dark'
}

// ── `tui.theme` ──────────────────────────────────────────────────────
//
// The gateway stores a palette name. Only the two palettes this UI ships can be
// named, plus `default`, which leaves the choice to detection. The value is not
// validated when it is stored (the config schema keeps it a plain string so a
// name written by an older release cannot stop the config loading), so it is
// validated here, and anything else falls back to detection.

export const THEME_NAMES = ['default', 'dark', 'light'] as const

export type ThemeName = (typeof THEME_NAMES)[number]

/** The palette this value names, or nothing when it names none. */
export function resolveThemeName(value: unknown): ThemeName | undefined {
  const name = String(value ?? '')
    .trim()
    .toLowerCase()

  return (THEME_NAMES as readonly string[]).includes(name) ? (name as ThemeName) : undefined
}

/** The scheme a name fixes, or null when it leaves the choice to detection. */
export function schemeForThemeName(name: ThemeName): ColorScheme | null {
  return name === 'default' ? null : name
}

export interface ConfiguredThemeOutcome {
  /** The stored value named no palette. It was ignored; say so. */
  invalid: boolean
  /** The palette changed, so what is already on screen has to be repainted. */
  repaint: boolean
}

/**
 * Apply the stored `tui.theme`.
 *
 * Precedence, highest first: `OPENDDE_HARNESS_TUI_LIGHT` / `_THEME`, which pin
 * the theme at construction; then this; then what the environment and the
 * terminal imply. A name applied here pins the palette too, so the terminal's
 * background reply cannot move a palette the user chose.
 */
export function applyConfiguredTheme(theme: Theme, value: unknown): ConfiguredThemeOutcome {
  const raw = String(value ?? '').trim()

  if (!raw) {
    return { invalid: false, repaint: false }
  }

  const name = resolveThemeName(raw)

  if (!name) {
    return { invalid: true, repaint: false }
  }

  const scheme = schemeForThemeName(name)

  // `default` asks for detection, which is what is already in force; and an
  // environment variable that named a palette outranks the stored one.
  if (!scheme || theme.pinned) {
    return { invalid: false, repaint: false }
  }

  return { invalid: false, repaint: theme.pinScheme(scheme) }
}

/** The scheme implied by a terminal's measured background color. */
export function schemeFromBackground(rgb: RgbColor): ColorScheme {
  return luminance(rgb) >= LUMA_LIGHT_THRESHOLD ? 'light' : 'dark'
}

// ── Theme ────────────────────────────────────────────────────────────

const ANSI_256_RE = /^ansi256\(\s?(\d+)\s?\)$/

type Paint = (text: string) => string

/**
 * The palette in use, as callable styles. Holds its own `Chalk` instance pinned
 * to the resolved tier, so nothing depends on the global chalk singleton's
 * auto-detection or on module import order.
 *
 * The scheme is mutable: an OSC 11 background reply can arrive after boot, and
 * every component holds this same instance, so `setScheme` re-themes the whole
 * app in place (callers still need to invalidate and re-render).
 */
export class Theme {
  readonly tier: ColorTier

  private readonly chalk: ChalkInstance
  private readonly fgPaint = new Map<ThemeToken, Paint>()
  private readonly bgPaint = new Map<ThemeToken, Paint>()
  private rampPaint: Paint[] = []

  private currentScheme: ColorScheme
  private currentColors: ThemeColors
  private pinnedScheme = false

  constructor(scheme: ColorScheme, tier: ColorTier, chalkInstance?: ChalkInstance) {
    this.tier = tier
    this.chalk = chalkInstance ?? new Chalk({ level: tier })
    this.currentScheme = scheme
    this.currentColors = resolvePalette(scheme, tier)
    this.buildPaints()
  }

  get scheme(): ColorScheme {
    return this.currentScheme
  }

  get colors(): ThemeColors {
    return this.currentColors
  }

  /** Whether a choice fixed this palette, as opposed to a measurement. */
  get pinned(): boolean {
    return this.pinnedScheme
  }

  /** Swap the palette in place. Returns whether anything changed.
   *
   *  A pinned palette refuses: this is the path the terminal's background
   *  reply takes, and a measurement must not overrule what the user asked for.
   */
  setScheme(scheme: ColorScheme): boolean {
    return this.pinnedScheme ? false : this.applyScheme(scheme)
  }

  /** Choose the palette and keep it. Returns whether anything changed. */
  pinScheme(scheme: ColorScheme): boolean {
    this.pinnedScheme = true

    return this.applyScheme(scheme)
  }

  private applyScheme(scheme: ColorScheme): boolean {
    if (scheme === this.currentScheme) {
      return false
    }

    this.currentScheme = scheme
    this.currentColors = resolvePalette(scheme, this.tier)
    this.buildPaints()

    return true
  }

  /** pi's strikethrough, for an id a list still carries but no provider serves. */
  strikethrough(text: string): string {
    return this.chalk.strikethrough(text)
  }

  fg(token: ThemeToken, text: string): string {
    return this.fgPaint.get(token)!(text)
  }

  bg(token: ThemeToken, text: string): string {
    return this.bgPaint.get(token)!(text)
  }

  /** How many bands the brand ramp offers. The banner splits its art into
   *  this many, so the gradient spans the art whatever its height. */
  get rampBands(): number {
    return this.rampPaint.length
  }

  /** One band of the brand ramp, 0 = lightest. Out-of-range clamps to the
   *  darkest entry rather than throwing: callers band art by arithmetic. */
  ramp(band: number, text: string): string {
    const paint = this.rampPaint[Math.max(0, Math.min(this.rampPaint.length - 1, band))]

    return paint ? paint(text) : text
  }

  bold(text: string): string {
    return this.chalk.bold(text)
  }

  dim(text: string): string {
    return this.chalk.dim(text)
  }

  italic(text: string): string {
    return this.chalk.italic(text)
  }

  underline(text: string): string {
    return this.chalk.underline(text)
  }

  /** Markdown rendering colors for pi-tui's `Markdown` component. */
  markdownTheme(): MarkdownTheme {
    return {
      // Prose is not identity. The reference leaves headings, bullets and code
      // blocks at the terminal's own foreground, gives links and inline code
      // its periwinkle, and the URL its dim grey; brand violet on a heading or
      // a `span` of code is decoration, which rule 1 of the palette spec puts
      // outside what violet is for.
      heading: text => this.chalk.bold(this.fg('text', text)),
      link: text => this.fg('label', text),
      linkUrl: text => this.fg('dim', text),
      code: text => this.fg('label', text),
      codeBlock: text => this.fg('text', text),
      // pi-tui supplies fence markers here; retain only the language label.
      codeBlockBorder: text => this.fg('border', text.replace(/^`{3,}/, '')),
      quote: text => this.fg('muted', text),
      quoteBorder: text => this.fg('border', text),
      hr: text => this.fg('border', text),
      listBullet: text => this.fg('text', text),
      bold: text => this.chalk.bold(text),
      italic: text => this.chalk.italic(text),
      strikethrough: text => this.chalk.strikethrough(text),
      underline: text => this.chalk.underline(text)
    }
  }

  /** Selection-list colors, shared by the editor's autocomplete and by pickers. */
  selectListTheme(): SelectListTheme {
    return {
      selectedPrefix: text => this.fg('accent', text),
      selectedText: text => this.fg('accent', text),
      description: text => this.fg('muted', text),
      scrollInfo: text => this.fg('muted', text),
      noMatch: text => this.fg('muted', text)
    }
  }

  /** The small editable-enum list, used for the Wire field in the key form. */
  settingsListTheme(): SettingsListTheme {
    return {
      label: (text, selected) => (selected ? this.fg('accent', text) : this.fg('text', text)),
      value: (text, selected) => (selected ? this.fg('accent', text) : this.fg('muted', text)),
      description: text => this.fg('muted', text),
      cursor: '\u2192 ',
      hint: text => this.fg('muted', text)
    }
  }

  editorTheme(): EditorTheme {
    return {
      borderColor: text => this.fg('border', text),
      selectList: this.selectListTheme()
    }
  }

  private buildPaints(): void {
    this.fgPaint.clear()
    this.bgPaint.clear()

    for (const [token, value] of Object.entries(this.currentColors) as [ThemeToken, string][]) {
      this.fgPaint.set(token, this.paintFor(value, 'foreground'))
      this.bgPaint.set(token, this.paintFor(value, 'background'))
    }

    this.rampPaint = resolveBrandRamp(this.currentScheme, this.tier).map(value => this.paintFor(value, 'foreground'))
  }

  /**
   * Turn one palette value into a style function. Values come in four shapes:
   * `#rrggbb`, `ansi256(N)`, `ansi:<chalk color name>`, and `inherit`.
   */
  private paintFor(value: string, type: 'background' | 'foreground'): Paint {
    const c = this.chalk

    // Deliberately unpainted, so the terminal's own foreground or ground
    // stands. Not the same as an unknown value, which also ends at the
    // identity below but is a mistake rather than a choice.
    if (value === INHERIT) {
      return text => text
    }

    if (value.startsWith('#')) {
      return type === 'foreground' ? c.hex(value) : c.bgHex(value)
    }

    const ansi256 = ANSI_256_RE.exec(value)

    if (ansi256) {
      const n = Number(ansi256[1])

      return type === 'foreground' ? c.ansi256(n) : c.bgAnsi256(n)
    }

    if (value.startsWith('ansi:')) {
      const name = value.slice('ansi:'.length)
      const key = type === 'foreground' ? name : 'bg' + name[0]!.toUpperCase() + name.slice(1)
      const paint = (c as unknown as Record<string, Paint | undefined>)[key]

      if (paint) {
        return paint.bind(c) as Paint
      }
    }

    return text => text
  }
}

/**
 * The theme this process should render with, from the environment.
 *
 * A palette the environment named is pinned: neither the stored `tui.theme` nor
 * the terminal's own background reply may move it afterwards.
 */
export function createTheme(env: NodeJS.ProcessEnv = process.env, detectedLevel = new Chalk().level): Theme {
  const theme = new Theme(detectScheme(env), resolveColorTier(env, detectedLevel))
  const explicit = explicitScheme(env)

  if (explicit) {
    theme.pinScheme(explicit)
  }

  return theme
}
