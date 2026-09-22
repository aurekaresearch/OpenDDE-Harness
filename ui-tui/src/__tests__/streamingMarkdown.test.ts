// Ported from `<scratchpad>/tui-bench/verify_streaming.ts`. The contract is
// that splitting the stream into frozen blocks is invisible: at every prefix
// the component renders exactly what a whole-text `Markdown` would.

import type { MarkdownTheme } from '@earendil-works/pi-tui'

import { Markdown } from '@earendil-works/pi-tui'
import { stripVTControlCharacters as stripAnsi } from 'node:util'
import { describe, expect, it } from 'vitest'

import { StreamingMarkdown } from '../components/streamingMarkdown.js'
import { Theme } from '../theme.js'
import { buildStream, streamText } from './streams.js'

const THEME = new Theme('dark', 3).markdownTheme()
const WIDTH = 120
const PAD = 1

describe('code block presentation', () => {
  it.each(['bash', 'text', ''])('hides fences and keeps indentation and the %s label', language => {
    const text = `\`\`\`${language}\nddeharness tracing\n\`\`\`\n`
    const lines = whole(text).map(line => stripAnsi(line).trimEnd())
    expect(lines.some(line => line.includes('```'))).toBe(false)
    expect(lines).toContain('   ddeharness tracing')
    if (language) {
      expect(lines.some(line => line.trim() === language)).toBe(true)
    }
    expect(mismatchesByCharacter(text)).toEqual([])
    const streaming = new StreamingMarkdown(text, PAD, THEME)
    streaming.finish()
    expect(streaming.render(WIDTH)).toEqual(whole(text))
  })

  it('preserves backticks inside code content', () => {
    const text = '~~~~text\n```literal\n~~~~\n'
    expect(whole(text).map(line => stripAnsi(line).trimEnd())).toContain('   ```literal')
  })
})

function whole(text: string, theme: MarkdownTheme = THEME): string[] {
  return new Markdown(text, PAD, 0, theme).render(WIDTH)
}

/** How many entries the component is holding in caches of its own, counting
 *  nested ones. */
function retainedEntries(component: object): number {
  const count = (value: unknown): number => {
    if (value instanceof Set) {
      return value.size
    }

    if (!(value instanceof Map)) {
      return 0
    }

    return [...value.values()].reduce<number>((total, nested) => total + count(nested), value.size)
  }

  return Object.values(component).reduce<number>((total, value) => total + count(value), 0)
}

/** Feed `text` one character at a time, comparing every prefix. Returns the
 *  prefixes that did not match a whole-text render. */
function mismatchesByCharacter(text: string, theme: MarkdownTheme = THEME): number[] {
  const component = new StreamingMarkdown('', PAD, theme)
  const mismatches: number[] = []

  for (let i = 1; i <= text.length; i++) {
    const prefix = text.slice(0, i)

    component.setText(prefix)

    if (JSON.stringify(component.render(WIDTH)) !== JSON.stringify(whole(prefix, theme))) {
      mismatches.push(i)
    }
  }

  return mismatches
}

const BOUNDARY_CASES: Record<string, string> = {
  bracketMath: '\\[\nx + y\n\nz = 2\n\\]\n\nNext paragraph.\n',
  crlf: 'First.\r\n \t\r\nSecond.\r\n\r\nThird.\r\n',
  indentedCode: '    const x = 1\n\n    const y = 2\n\nNext paragraph.\n',
  looseList: '- first\n\n- second\n\n  continued paragraph\n\nNext paragraph.\n',
  math: '$$\nx + y\n\nz = 2\n$$\n\nNext paragraph.\n',
  openFence: '````ts\nconst x = 1\n\n~~~\n```\nconst y = 2\n````\n\nNext paragraph.\n',
  paragraphs: 'First paragraph.\n\nSecond paragraph.\n\nThird paragraph.',
  quote: '> first\n>\n> second\n\n> continued\n\nNext paragraph.\n',
  setext: 'A heading\n---\n\nNext paragraph.\n',
  table: '| a | b |\n|---|---|\n| c | d |\n\nNext paragraph.\n',
  tildeFence: '~~~ts\nconst x = 1\n\n```\nconst y = 2\n~~~\n\nNext paragraph.\n'
}

describe('StreamingMarkdown boundaries', () => {
  for (const [name, text] of Object.entries(BOUNDARY_CASES)) {
    it(`matches a whole-text Markdown at every prefix: ${name}`, () => {
      expect(mismatchesByCharacter(text)).toEqual([])
    })
  }

  // A link definition resolves document-wide, in both directions, so blocks
  // that could depend on one are not independent and are not split.
  const NONLOCAL_CASES: Record<string, string> = {
    definitionAfterUse: '[label][ref]\n\nAnother paragraph.\n\n[ref]: https://example.com\n',
    definitionBeforeUse: '[ref]: https://example.com\n\nAnother paragraph.\n\n[label][ref]\n',
    footnote: 'A claim[^1].\n\nAnother paragraph.\n\n[^1]: the note\n',
    // An HTML comment swallows the blank line that would otherwise close a
    // block, so a boundary declared inside one is not a boundary.
    htmlComment: '<!--\nfirst\n\nsecond\n-->\n\nAfter\n',
    htmlCommentInline: 'Text <!-- a note\n\nstill the note --> and on.\n\nAfter\n',
    indentedDefinition: '   [ref]: https://example.com\n\nText.\n\n[label][ref]\n'
  }

  for (const [name, text] of Object.entries(NONLOCAL_CASES)) {
    it(`matches a whole-text Markdown at every prefix: ${name}`, () => {
      expect(mismatchesByCharacter(text)).toEqual([])
    })
  }

  it('stops splitting from the delta that opens an HTML comment', () => {
    const component = new StreamingMarkdown('', PAD, THEME)

    component.setText('First paragraph.\n\nSecond paragraph.\n\n')
    expect(component.children.length).toBeGreaterThan(1)

    component.setText('First paragraph.\n\nSecond paragraph.\n\n<!--\nnote\n')

    expect(component.children.length).toBe(1)
  })

  it('stops splitting from the delta that defines a reference', () => {
    const component = new StreamingMarkdown('', PAD, THEME)

    component.setText('First paragraph.\n\nSecond paragraph.\n\n')
    expect(component.children.length).toBeGreaterThan(1)

    component.setText('First paragraph.\n\nSecond paragraph.\n\n[ref]: https://example.com\n\n[a][ref]\n')

    // One `Markdown` again: the already-frozen blocks were given back so the
    // definition can reach the use.
    expect(component.children.length).toBe(1)
    expect(component.render(WIDTH)).toEqual(
      whole('First paragraph.\n\nSecond paragraph.\n\n[ref]: https://example.com\n\n[a][ref]\n')
    )
  })

  it('keeps splitting a reply that only uses a reference nothing defines', () => {
    const component = new StreamingMarkdown('', PAD, THEME)

    component.setText('[label][ref] is literal here.\n\nSecond paragraph.\n\nThird.\n')

    expect(component.children.length).toBeGreaterThan(1)
  })

  it('holds a loose list open until a non-list line completes', () => {
    const component = new StreamingMarkdown('', PAD, THEME)

    component.setText('- one\n\n- two\n\n')
    expect(component.children.length).toBe(1)

    component.setText('- one\n\n- two\n\nNext.\n')
    expect(component.children.length).toBe(2)
  })

  it('never rewrites a block it has already frozen', () => {
    const component = new StreamingMarkdown('', PAD, THEME)

    component.setText('- one\n\n- two\n\nNext.\n')

    const frozen = component.children[0] as Markdown
    const lines = frozen.render(WIDTH)

    frozen.setText = () => {
      throw new Error('a completed block was mutated')
    }

    component.setText('- one\n\n- two\n\nNext.\nMore.')

    expect(component.children[0]).toBe(frozen)
    expect(frozen.render(WIDTH)).toEqual(lines)
  })
})

describe('StreamingMarkdown streams', () => {
  for (const kind of ['default', 'fence', 'long'] as const) {
    it(`matches a whole-text Markdown throughout the ${kind} benchmark stream`, () => {
      const deltas = buildStream(kind)
      const component = new StreamingMarkdown('', PAD, THEME)

      let text = ''
      let checkpoints = 0

      for (let i = 0; i < deltas.length; i++) {
        text += deltas[i]
        component.setText(text)

        if (i % 128 === 127 || i === deltas.length - 1) {
          checkpoints++
          expect(component.render(WIDTH)).toEqual(whole(text))
        }
      }

      expect(checkpoints).toBeGreaterThan(8)

      // `fence` is one unbroken code block: there is no boundary to split at,
      // so it degrades to a single block. The other two split.
      expect(component.children.length > 1).toBe(kind !== 'fence')
    })
  }
})

describe('StreamingMarkdown finish', () => {
  it('freezes without re-parsing anything', () => {
    const text = streamText('default')
    const component = new StreamingMarkdown(text, PAD, THEME)
    const before = component.render(WIDTH)
    const children = [...component.children]

    expect(children.length).toBeGreaterThan(1)

    for (const child of children) {
      ;(child as Markdown).setText = () => {
        throw new Error('finish() re-parsed a block')
      }
    }

    component.finish()
    component.finish()

    expect(component.children).toEqual(children)
    expect(component.render(WIDTH)).toEqual(before)
    expect(before).toEqual(whole(text))
  })

  it('rebuilds when the text is replaced after finishing', () => {
    const component = new StreamingMarkdown('First.\n\nSecond.\n', PAD, THEME)

    component.finish()
    component.setText('Replacement.\n\nNext.\n')

    expect(component.render(WIDTH)).toEqual(whole('Replacement.\n\nNext.\n'))

    component.finish()
    component.setText('Reused after finish.')

    expect(component.render(WIDTH)).toEqual(whole('Reused after finish.'))
  })

  it('rebuilds when the new text is not an extension of the old', () => {
    const component = new StreamingMarkdown('One.\n\nTwo.\n', PAD, THEME)

    component.setText('Different.\n\nText.\n')

    expect(component.render(WIDTH)).toEqual(whole('Different.\n\nText.\n'))
  })
})

describe('StreamingMarkdown code highlighting', () => {
  /** A highlighter whose answer for a line depends on the lines above it: a
   *  block comment opened on one line runs until another closes it. Nothing
   *  can decide that from a single line, which is why the theme's callback is
   *  handed whole blocks, as pi hands them to it. */
  function blockAware(): MarkdownTheme {
    return {
      ...THEME,
      highlightCode: (code: string) => {
        let inComment = false

        return code.split('\n').map(line => {
          const opens = line.includes('/*')
          const closes = line.includes('*/')
          const marked = inComment || opens ? `«${line}»` : line

          inComment = (inComment || opens) && !closes

          return marked
        })
      }
    }
  }

  it('hands the highlighter whole blocks, so its state spans their lines', () => {
    const theme = blockAware()

    expect(mismatchesByCharacter('```ts\n/* one\ntwo\nthree */\nfour\n```\n\nAfter.\n', theme)).toEqual([])
  })

  it('passes the callback the theme supplied through rather than wrapping it', () => {
    const seen: string[] = []
    const theme: MarkdownTheme = {
      ...THEME,
      highlightCode: (code: string, lang?: string) => {
        seen.push(`${lang}:${code}`)

        return code.split('\n')
      }
    }
    const component = new StreamingMarkdown('```ts\nconst x = 1\nconst y = 2\n```\n', PAD, theme)

    component.render(WIDTH)

    expect(seen).toContain('ts:const x = 1\nconst y = 2')
  })

  it('keeps no per-prefix cache while a long line is written', () => {
    // Memoizing each growing prefix of the line being typed retained
    // n(n+1)/2 characters and never released them, not even on finish().
    const theme = blockAware()
    const component = new StreamingMarkdown('', PAD, theme)
    const line = 'x'.repeat(2_000)

    for (let i = 1; i <= line.length; i++) {
      component.setText(`\`\`\`ts\n${line.slice(0, i)}`)
      component.render(WIDTH)
    }

    expect(retainedEntries(component)).toBe(0)

    component.finish()

    expect(retainedEntries(component)).toBe(0)
  })
})
