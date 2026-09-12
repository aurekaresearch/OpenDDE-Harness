// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import { render } from 'ink-testing-library'
import { createElement } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type * as emojiModule from '../lib/emoji.js'
import type * as syntaxModule from '../lib/syntax.js'

import { Md } from '../components/markdown.js'
import { findStableBoundary, splitStableBlocks, StreamingMd } from '../components/streamingMarkdown.js'
import { LIVE_RENDER_MAX_CHARS } from '../config/limits.js'
import { ensureEmojiPresentation } from '../lib/emoji.js'
import { highlightLine } from '../lib/syntax.js'
import { boundedLiveRenderText } from '../lib/text.js'

// Every markdown parse (cache miss) passes its text through
// ensureEmojiPresentation exactly once, and every highlighted code row calls
// highlightLine once per render: counting them measures the streaming work
// without timing anything.
vi.mock('../lib/emoji.js', async importOriginal => {
  const mod = await importOriginal<typeof emojiModule>()

  return { ...mod, ensureEmojiPresentation: vi.fn(mod.ensureEmojiPresentation) }
})

vi.mock('../lib/syntax.js', async importOriginal => {
  const mod = await importOriginal<typeof syntaxModule>()

  return { ...mod, highlightLine: vi.fn(mod.highlightLine) }
})
// We test the pure boundary logic by rendering the component's ref
// behaviour through repeated calls. Since React isn't being rendered here,
// we reach into the module to test findStableBoundary via its exported
// behaviour — but the pure helper isn't exported. So test the component's
// observable output: pass sequential text values and verify the stable
// prefix never retreats.
//
// Strategy: mount StreamingMd in isolation and observe which <Md>
// instances it renders (by text prop). Without a DOM renderer that's
// heavy, so we validate the helper behaviour by directly invoking the
// fence/boundary logic via a re-exported surface.
import { DEFAULT_THEME } from '../theme.js'

describe('findStableBoundary', () => {
  it('returns -1 when no blank line exists yet', () => {
    expect(findStableBoundary('partial line with no newline yet')).toBe(-1)
  })

  it('returns -1 when only single newlines exist', () => {
    expect(findStableBoundary('line one\nline two\nline three')).toBe(-1)
  })

  it('splits after the last blank line separator', () => {
    // 'first\n\nsecond\n\nthird' → last blank = before 'third'
    const text = 'first paragraph\n\nsecond paragraph\n\nthird'
    const idx = findStableBoundary(text)

    expect(text.slice(0, idx)).toBe('first paragraph\n\nsecond paragraph\n\n')
    expect(text.slice(idx)).toBe('third')
  })

  it('refuses to split inside an open fenced block', () => {
    // Fence opens, contains a blank line inside the code, no close yet.
    const text = '```ts\nfn();\n\nmore code here'

    expect(findStableBoundary(text)).toBe(-1)
  })

  it('splits before an open fenced block but not inside', () => {
    const text = 'intro paragraph\n\n```ts\nfn();\n\nmore code'
    const idx = findStableBoundary(text)

    expect(text.slice(0, idx)).toBe('intro paragraph\n\n')
    expect(text.slice(idx).startsWith('```ts')).toBe(true)
  })

  it('allows splitting after a fenced block closes', () => {
    const text = '```ts\nfn();\n```\n\nnarration continues'
    const idx = findStableBoundary(text)

    expect(text.slice(0, idx)).toBe('```ts\nfn();\n```\n\n')
    expect(text.slice(idx)).toBe('narration continues')
  })

  it('walks backwards through nested fence boundaries safely', () => {
    // Two closed fences + narration + one new open fence. The only legal
    // split is before the open fence, not between the closed ones.
    const text = '```js\na\n```\n\nmid text\n\n```python\nstill open'
    const idx = findStableBoundary(text)

    expect(text.slice(0, idx)).toBe('```js\na\n```\n\nmid text\n\n')
  })

  it('handles empty input', () => {
    expect(findStableBoundary('')).toBe(-1)
  })

  it('refuses to split inside an open $$ math block', () => {
    // Display math has been opened but not closed; the only blank line
    // sits inside the open block, so there's no safe boundary yet.
    const text = '$$\nx + y\n\nmore math'

    expect(findStableBoundary(text)).toBe(-1)
  })

  it('allows splitting after a $$ math block closes', () => {
    const text = '$$\nx + y = z\n$$\n\nnarration continues'
    const idx = findStableBoundary(text)

    expect(text.slice(0, idx)).toBe('$$\nx + y = z\n$$\n\n')
    expect(text.slice(idx)).toBe('narration continues')
  })

  it('splits before an open $$ block but not inside', () => {
    // Mirror of the existing fenced-code test: prose, then an unclosed
    // math block. The only safe boundary is the blank line BEFORE `$$`.
    const text = 'intro paragraph\n\n$$\nx + y\n\nmore'
    const idx = findStableBoundary(text)

    expect(text.slice(0, idx)).toBe('intro paragraph\n\n')
    expect(text.slice(idx).startsWith('$$')).toBe(true)
  })

  it('treats single-line $$x$$ as zero net toggle', () => {
    // `$$x = y$$` opens AND closes on one line, so the stable boundary
    // after it is allowed.
    const text = 'intro\n\n$$x = y$$\n\nnarration'
    const idx = findStableBoundary(text)

    expect(text.slice(0, idx)).toBe('intro\n\n$$x = y$$\n\n')
    expect(text.slice(idx)).toBe('narration')
  })

  it('refuses to split inside an open \\[ math block', () => {
    const text = '\\[\nx + y\n\nmore'

    expect(findStableBoundary(text)).toBe(-1)
  })
})

describe('streaming theme assumption', () => {
  it('theme is exportable (component import sanity check)', () => {
    // Sanity that the theme we pass doesn't change shape. Component import
    // already happens above — this is a smoke test that the module graph
    // for streamingMarkdown wires up without cycles.
    expect(DEFAULT_THEME.color.accent).toBeTruthy()
  })
})

describe('splitStableBlocks', () => {
  it('emits completed blocks and keeps the in-flight remainder', () => {
    expect(splitStableBlocks('a\n\nb\n\n\nc')).toEqual({ blocks: ['a\n\n', 'b\n\n\n'], tail: 'c' })
    expect(splitStableBlocks('```\nx\n\ny\n```\n\nz')).toEqual({ blocks: ['```\nx\n\ny\n```\n\n'], tail: 'z' })
    expect(splitStableBlocks('a\n\n')).toEqual({ blocks: ['a\n\n'], tail: '' })
  })
})

const PARA = (n: number) =>
  `Paragraph ${n} keeps going with **bold**, \`code\` and enough words that it wraps across a few terminal rows.`

const stream = (text: string, size: number) => {
  const deltas: string[] = []

  for (let i = 0; i < text.length; i += size) {
    deltas.push(text.slice(i, i + size))
  }

  return deltas
}

const parseWork = () => vi.mocked(ensureEmojiPresentation).mock.calls.reduce((chars, [text]) => chars + text.length, 0)

const live = (text: string) => createElement(StreamingMd, { compact: false, t: DEFAULT_THEME, text })

describe('StreamingMd streaming cost', () => {
  beforeEach(() => {
    vi.mocked(ensureEmojiPresentation).mockClear()
    vi.mocked(highlightLine).mockClear()
  })

  it('renders a completed stream exactly like a one-shot <Md>', () => {
    const text = [
      '# Title',
      '',
      PARA(1),
      '',
      '- one\n- two',
      '',
      '```ts\nconst a = 1\nconst b = "two"\n```',
      '',
      '| k | v |\n|---|---|\n| a | 1 |',
      '',
      '> quoted',
      '',
      PARA(2)
    ].join('\n')
    const streaming = render(live(''))

    let buf = ''

    for (const delta of stream(text, 7)) {
      buf += delta
      streaming.rerender(live(buf))
    }

    const oneShot = render(createElement(Md, { compact: false, t: DEFAULT_THEME, text }))

    expect(streaming.lastFrame()).toBe(oneShot.lastFrame())
  })

  // CI render time varies; the assertions below bound parsing work independently of wall time.
  it('re-parses only the tail while the live text is trimmed past the live cap', () => {
    const text = Array.from({ length: 320 }, (_, i) => PARA(i)).join('\n\n')
    const deltas = stream(text, 250)

    expect(text.length).toBeGreaterThan(LIVE_RENDER_MAX_CHARS * 1.5)

    const streaming = render(live(''))
    let buf = ''

    for (const delta of deltas) {
      buf += delta
      streaming.rerender(live(boundedLiveRenderText(buf)))
    }

    // Each paragraph is parsed once as a completed block, plus each delta
    // re-parses the in-flight tail and, past the cap, the trimmed label block.
    // Re-parsing the whole live window per delta would cost about
    // deltas * LIVE_RENDER_MAX_CHARS characters (~40x the text length here).
    expect(parseWork()).toBeLessThan(text.length * 4)
    expect(vi.mocked(ensureEmojiPresentation).mock.calls.length).toBeLessThan(deltas.length * 6)
  }, 20_000)

  it('highlights each streamed code line a bounded number of times', () => {
    const lines = Array.from({ length: 120 }, (_, i) => `const value${i} = compute(${i}) + "s"`)
    const streaming = render(live('```ts\n'))
    let buf = '```ts\n'

    for (const line of lines) {
      buf += `${line}\n`
      streaming.rerender(live(buf))
    }

    streaming.rerender(live(`${buf}\`\`\`\n\ndone`))

    // Without per-line memoization every delta re-highlights the whole open
    // fence: about lines^2 / 2 calls.
    expect(vi.mocked(highlightLine).mock.calls.length).toBeLessThan(lines.length * 4)
  })
})
