// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

const RICH_RE = /\[(?:bold\s+)?(?:dim\s+)?(#(?:[0-9a-fA-F]{3,8}))\]([\s\S]*?)(\[\/\])/g

export function parseRichMarkup(markup: string): Line[] {
  const lines: Line[] = []

  for (const raw of markup.split('\n')) {
    const trimmed = raw.trimEnd()

    if (!trimmed) {
      lines.push(['', ' '])

      continue
    }

    const matches = [...trimmed.matchAll(RICH_RE)]

    if (!matches.length) {
      lines.push(['', trimmed])

      continue
    }

    let cursor = 0

    for (const m of matches) {
      const before = trimmed.slice(cursor, m.index)

      if (before) {
        lines.push(['', before])
      }

      lines.push([m[1]!, m[2]!])
      cursor = m.index! + m[0].length
    }

    if (cursor < trimmed.length) {
      lines.push(['', trimmed.slice(cursor)])
    }
  }

  return lines
}

const BLOCK_GLYPHS: Record<string, readonly string[]> = {
  A: ['01110', '10001', '10001', '11111', '10001', '10001', '10001'],
  D: ['11110', '10001', '10001', '10001', '10001', '10001', '11110'],
  E: ['11111', '10000', '10000', '11110', '10000', '10000', '11111'],
  G: ['01110', '10001', '10000', '10111', '10001', '10001', '01110'],
  H: ['10001', '10001', '10001', '11111', '10001', '10001', '10001'],
  M: ['10001', '11011', '10101', '10101', '10001', '10001', '10001'],
  N: ['10001', '11001', '10101', '10011', '10001', '10001', '10001'],
  O: ['01110', '10001', '10001', '10001', '10001', '10001', '01110'],
  P: ['11110', '10001', '10001', '11110', '10000', '10000', '10000'],
  R: ['11110', '10001', '10001', '11110', '10100', '10010', '10001'],
  S: ['01111', '10000', '10000', '01110', '00001', '00001', '11110'],
  T: ['11111', '00100', '00100', '00100', '00100', '00100', '00100']
}

export type BlockStyle = 'double' | 'single'

// Cell strings per block style: `double` draws each glyph cell as two columns
// (the classic wide wordmark), `single` as one column for narrow terminals.
const BLOCK_CELLS: Record<BlockStyle, { block: string; empty: string; letterGap: string; wordGap: string }> = {
  double: { block: '██', empty: '  ', letterGap: '  ', wordGap: '      ' },
  single: { block: '█', empty: ' ', letterGap: ' ', wordGap: '   ' }
}

const renderBlockWordmark = (label: string, style: BlockStyle = 'double'): string[] => {
  const cells = BLOCK_CELLS[style]
  const words = label.split(' ')
  const rows = Array.from({ length: 7 }, (_, row) =>
    words
      .map(word =>
        [...word]
          .map(letter =>
            (BLOCK_GLYPHS[letter] ?? BLOCK_GLYPHS.E)![row]!.replaceAll('1', cells.block).replaceAll('0', cells.empty)
          )
          .join(cells.letterGap)
      )
      .join(cells.wordGap)
  )

  // Preserve the established 8-row banner height used by the color ramp and layout.
  return [...rows, '']
}

// The product name, one word per line, so the wordmark fits without the
// ~130-column single-line form ever being needed.
const WORDMARK_WORDS = ['OPENDDE', 'HARNESS'] as const

const renderStackedWordmark = (style: BlockStyle): string[] =>
  WORDMARK_WORDS.flatMap(word => renderBlockWordmark(word, style))

const STACKED_WORDMARK: Record<BlockStyle, string[]> = {
  double: renderStackedWordmark('double'),
  single: renderStackedWordmark('single')
}

const OPENDDE_HARNESS_LOGO_ART = STACKED_WORDMARK.double

const OPENDDE_HARNESS_HERO_ART = [
  '                                      ▄████',
  '                                  ▄███▀ ███',
  '                              ▄███▀     ███',
  '                          ▄███▀          ███',
  '                       ▄██▀              ███',
  '       ████████                     ████████',
  '        ████████                   ████████ ',
  '         ████████                 ████████  ',
  '          ████████               ████████   ',
  '  ▄██▄     ████████             ████████    ',
  ' ██▀        ████████           ████████     ',
  '██▀          ████████         ████████      ',
  '██            ████████       ████████       ',
  '██             ████████     ████████        ',
  '██              ████████████████████         ',
  '██                ████████████████           ',
  ' ██▄                ████████████             ',
  '  ▀██▄                 ██████                ',
  '    ▀███▄               ██████               ',
  '       ▀████▄▄           ██████               ',
  '            ▀████▄▄      ██████               ',
  '                 ▀█████████████▄▄▄▄▄          ',
  '                       ▀▀▀▀▀▀▀██████▀          ',
  ''
].map(row => row.trimEnd())

const linesWidth = (rows: readonly string[]) => rows.reduce((width, row) => Math.max(width, [...row].length), 0)

export const OPENDDE_HARNESS_LOGO_WIDTH = linesWidth(OPENDDE_HARNESS_LOGO_ART)
export const OPENDDE_HARNESS_LOGO_ROWS = OPENDDE_HARNESS_LOGO_ART.length
// The single-width stacked wordmark, for terminals too narrow for the double form.
export const OPENDDE_HARNESS_LOGO_COMPACT_WIDTH = linesWidth(STACKED_WORDMARK.single)
export const OPENDDE_HARNESS_HERO_WIDTH = linesWidth(OPENDDE_HARNESS_HERO_ART)
export const OPENDDE_HARNESS_HERO_ROWS = OPENDDE_HARNESS_HERO_ART.length

// How many vertical colour bands the hero is split into (horizontal gradient).
const HERO_BANDS = 5

// Art downsampling factors, largest art first. The hero and the wordmark are
// always drawn at the same factor so the lockup stays proportional.
export const ART_SCALE_FACTORS = [1, 2, 4] as const
export type ArtScale = (typeof ART_SCALE_FACTORS)[number]

const isFilled = (row: readonly string[], col: number) => {
  const ch = row[col]

  return ch !== undefined && ch !== ' '
}

// Downsample block art by `factor`: every output cell covers a factor-wide,
// factor-tall block of the bitmap (non-space = filled). The upper and lower
// halves of that block become the two halves of one terminal cell, so factor 2
// turns two source rows into one line of half blocks.
export const scaleArt = (rows: readonly string[], factor: number): string[] => {
  if (factor <= 1) {
    return rows.map(row => row.trimEnd())
  }

  const bitmap = rows.map(row => [...row])
  const width = linesWidth(rows)
  const half = Math.max(1, Math.floor(factor / 2))
  const outRows = Math.ceil(rows.length / factor)
  const outCols = Math.ceil(width / factor)
  const anyFilled = (rowStart: number, rowCount: number, colStart: number) => {
    for (let r = rowStart; r < rowStart + rowCount; r++) {
      const row = bitmap[r]

      if (!row) {
        continue
      }

      for (let c = colStart; c < colStart + factor; c++) {
        if (isFilled(row, c)) {
          return true
        }
      }
    }

    return false
  }
  const out: string[] = []

  for (let i = 0; i < outRows; i++) {
    let line = ''

    for (let j = 0; j < outCols; j++) {
      const top = anyFilled(i * factor, half, j * factor)
      const bottom = anyFilled(i * factor + half, factor - half, j * factor)

      line += top && bottom ? '█' : top ? '▀' : bottom ? '▄' : ' '
    }

    out.push(line.trimEnd())
  }

  return out
}

// Size of one art at a scale factor.
export const scaledRows = (rows: number, factor: number) => Math.ceil(rows / factor)
export const scaledCols = (cols: number, factor: number) => Math.ceil(cols / factor)

// Wordmark rows share ramp colours in equal vertical bands, top -> bottom, so
// the whole gradient spans the art whatever its height (the 16-row stacked
// mark uses 4 rows per band with the 4-stop banner palette).
const bandLines = (ramp: readonly string[], rows: readonly string[]): Line[] => {
  const rowsPerBand = Math.max(1, Math.ceil(rows.length / ramp.length))

  return rows.map((text, i) => [ramp[Math.floor(i / rowsPerBand)] ?? ramp[ramp.length - 1]!, text])
}

// Stacked "OPENDDE" / "HARNESS" wordmark in the given block style.
export const openddeHarnessWordmark = (ramp: readonly string[], style: BlockStyle = 'double'): Line[] =>
  bandLines(ramp, STACKED_WORDMARK[style])

// The wordmark downsampled by `factor`, the same pipeline the hero uses, so
// both halves of the lockup shrink together.
export const openddeHarnessWordmarkAt = (ramp: readonly string[], factor: number): Line[] =>
  bandLines(ramp, scaleArt(OPENDDE_HARNESS_LOGO_ART, factor))

// Title (wordmark): the double-width stacked mark, or a custom rich-markup logo.
export const openddeHarnessLogo = (ramp: readonly string[], customLogo?: string): Line[] =>
  customLogo ? parseRichMarkup(customLogo) : openddeHarnessWordmark(ramp, 'double')

// Just the first word, preserving the same glyphs.
const OPENDDE_HARNESS_WORD = renderBlockWordmark('OPENDDE')

// Width of the compact "OPENDDE" word.
export const OPENDDE_HARNESS_WORD_WIDTH = linesWidth(OPENDDE_HARNESS_WORD)

// Just the "OPENDDE" word, with the top -> bottom ramp gradient.
export const openddeHarnessLogoWord = (ramp: readonly string[]): Line[] => bandLines(ramp, OPENDDE_HARNESS_WORD)

// Hero (opendde): a horizontal gradient split into HERO_BANDS column bands,
// coloured right → left so the highlight (ramp[0]) lands on the right and it
// deepens toward the left (ramp[HERO_BANDS-1]). Each row is returned as an
// array of `[color, segment]` pairs rendered inline.
// `factor` > 1 draws the downsampled art (see `scaleHero`); a custom hero
// keeps its own colours at full size and takes the ramp bands once scaled.
export const openddeHarnessHero = (ramp: readonly string[], customHero?: string, factor: number = 1): Line[][] => {
  if (customHero && factor <= 1) {
    return parseRichMarkup(customHero).map(line => [line])
  }

  const art = customHero ? parseRichMarkup(customHero).map(([, text]) => text) : OPENDDE_HARNESS_HERO_ART

  return scaleArt(art, factor).map(row => {
    const chars = [...row]
    const bandWidth = Math.ceil(chars.length / HERO_BANDS)
    const segments: Line[] = []

    for (let band = 0; band < HERO_BANDS; band++) {
      const text = chars.slice(band * bandWidth, (band + 1) * bandWidth).join('')

      if (!text) {
        continue
      }

      const color = ramp[HERO_BANDS - 1 - band] ?? ramp[ramp.length - 1]!
      segments.push([color, text])
    }

    return segments
  })
}

export const artWidth = (lines: Line[]) => lines.reduce((m, [, t]) => Math.max(m, [...t].length), 0)

// Width of a segmented art (hero): max over rows of the summed segment widths.
export const rowsWidth = (rows: Line[][]) =>
  rows.reduce(
    (m, segs) =>
      Math.max(
        m,
        segs.reduce((a, [, t]) => a + [...t].length, 0)
      ),
    0
  )

type Line = [string, string]

export type WordmarkTier = 'text' | BlockStyle

export interface BannerLayout {
  // Hero art drawn beside the wordmark; only when the terminal has room for
  // both plus a gap, and enough rows for the taller hero.
  hero: boolean
  wordmark: WordmarkTier
}

// Rows the app draws around the welcome panel: the transcript's top padding
// (1), the panel's bottom margin (1), and the composer (spacer, input box
// margin, two rules, the input row, status rule: 6).
export const WELCOME_CHROME_ROWS = 8
// The panel's rounded border plus one row of vertical padding each side.
export const PANEL_FRAME_ROWS = 4
// Columns between the hero and the wordmark inside the lockup.
export const LOCKUP_GAP = 4

export interface LockupSize {
  heroCols: number
  heroRows: number
  rows: number
  wordmarkCols: number
  wordmarkRows: number
  cols: number
}

// Footprint of the brand lockup (hero + gap + wordmark) at a scale factor.
export const lockupSize = (factor: number): LockupSize => {
  const heroCols = scaledCols(OPENDDE_HARNESS_HERO_WIDTH, factor)
  const heroRows = scaledRows(OPENDDE_HARNESS_HERO_ROWS, factor)
  const wordmarkCols = scaledCols(OPENDDE_HARNESS_LOGO_WIDTH, factor)
  const wordmarkRows = scaledRows(OPENDDE_HARNESS_LOGO_ROWS, factor)

  return {
    cols: heroCols + LOCKUP_GAP + wordmarkCols,
    heroCols,
    heroRows,
    rows: Math.max(heroRows, wordmarkRows),
    wordmarkCols,
    wordmarkRows
  }
}

// Largest lockup that fits the budget; 0 means "no lockup", i.e. the caller
// falls back to the one-line text brand.
export const resolveLockupScale = (budgetRows: number, budgetCols: number): ArtScale | 0 => {
  for (const factor of ART_SCALE_FACTORS) {
    const size = lockupSize(factor)

    if (size.rows <= budgetRows && size.cols <= budgetCols) {
      return factor
    }
  }

  return 0
}
