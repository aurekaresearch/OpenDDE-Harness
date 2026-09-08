// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

// StreamingMd — incremental markdown renderer for in-flight assistant text.
//
// Naive approach (render <Md text={full}/>) re-tokenizes the entire message
// on every stream delta. This splits `text` at every stable top-level block
// boundary (blank line outside a fenced code / math span) and renders one
// <Md> per completed block plus one for the in-flight tail. <Md> is memoized
// on its exact text and caches its parse by text, so a completed block costs
// nothing on later deltas: React bails out of its subtree and the parser never
// sees it again. Only the tail re-parses — O(tail) per delta, not O(total).
//
// Blocks are keyed by their own text, not their index, so this stays true when
// the live text is trimmed from the front (boundedLiveRenderText moves the
// start on every delta once a reply crosses the live cap): the surviving
// blocks keep their fibers and only the label plus the cut first block change.
//
// Layout: the <Md> subtrees MUST render stacked (column). The parent
// container in messageLine.tsx is a default `flexDirection: 'row'` Box
// (Ink's default), so a bare Fragment of <Md> siblings would lay them out
// side-by-side — the "jumbled columns while streaming" rendering bug.

import { Box } from '@hermes/ink'
import { memo } from 'react'

import type { Theme } from '../theme.js'

import { Md } from './markdown.js'

type FenceState = { codeOpen: boolean; mathOpen: boolean; mathOpener: '$$' | '\\[' | null }

// Track ``` / ~~~ AND `$$` / `\[…\]` fence toggles line by line. Inside an
// open fence a blank line is not a block boundary; splitting there would
// orphan the fence and let the tail re-render as broken markdown. Math fences
// only toggle when the code fence is closed so snippets like
// ` ```\n$$x$$\n``` ` (math example inside a code block) don't double-count.
// A `$$x$$` line that opens AND closes on its own produces zero net toggles;
// that's `len >= 4` plus the trailing `$$`.
//
// NB: this is INTENTIONALLY more conservative than `markdown.tsx`'s parser,
// which falls back to paragraph rendering when an `$$` opener has no matching
// closer. The renderer can do that safely because it always sees the full
// text. A block committed here is frozen, so prematurely deciding "this `$$`
// is just prose" would render a paragraph that becomes wrong the instant the
// closer streams in. Treating any unmatched opener as still-open keeps the
// boundary parked behind it until the closer arrives (or the stream ends and
// the non-streaming <Md> takes over with its fallback).
const stepFence = (state: FenceState, line: string): void => {
  if (/^(?:`{3,}|~{3,})/.test(line)) {
    state.codeOpen = !state.codeOpen
  } else if (!state.codeOpen) {
    if (!state.mathOpen && /^\$\$/.test(line)) {
      if (!(line.length >= 4 && /\$\$$/.test(line))) {
        state.mathOpen = true
        state.mathOpener = '$$'
      }
    } else if (!state.mathOpen && /^\\\[/.test(line)) {
      if (!/\\\]$/.test(line)) {
        state.mathOpen = true
        state.mathOpener = '\\['
      }
    } else if (state.mathOpen && state.mathOpener === '$$' && /\$\$$/.test(line)) {
      state.mathOpen = false
      state.mathOpener = null
    } else if (state.mathOpen && state.mathOpener === '\\[' && /\\\]$/.test(line)) {
      state.mathOpen = false
      state.mathOpener = null
    }
  }
}

// Split `text` into completed top-level blocks — each ends at a blank line
// ("\n\n" plus any further newlines) outside fenced code / math — and the
// in-flight remainder. A block depends only on the text before it, so it
// never changes once emitted while the stream appends.
export const splitStableBlocks = (text: string): { blocks: string[]; tail: string } => {
  const blocks: string[] = []
  const state: FenceState = { codeOpen: false, mathOpen: false, mathOpener: null }
  let blockStart = 0
  let i = 0

  while (i < text.length) {
    const nl = text.indexOf('\n', i)

    if (nl < 0) {
      break
    }

    stepFence(state, text.slice(i, nl).trim())
    i = nl + 1

    if (text.charCodeAt(i) === 10 && !state.codeOpen && !state.mathOpen) {
      while (text.charCodeAt(i) === 10) {
        i++
      }

      blocks.push(text.slice(blockStart, i))
      blockStart = i
    }
  }

  return { blocks, tail: text.slice(blockStart) }
}

// Index just past the last stable block boundary, or -1 if none exists yet.
export const findStableBoundary = (text: string) => {
  const { tail } = splitStableBlocks(text)

  return tail.length === text.length ? -1 : text.length - tail.length
}

export const StreamingMd = memo(function StreamingMd({ compact, t, text }: StreamingMdProps) {
  const { blocks, tail } = splitStableBlocks(text)

  if (!blocks.length) {
    return <Md compact={compact} t={t} text={tail} />
  }

  const seen = new Map<string, number>()

  return (
    <Box flexDirection="column">
      {blocks.map(block => {
        const dup = seen.get(block) ?? 0
        seen.set(block, dup + 1)

        return <Md compact={compact} key={`${dup}:${block}`} t={t} text={block} />
      })}
      {tail ? <Md compact={compact} key="tail" t={t} text={tail} /> : null}
    </Box>
  )
})

interface StreamingMdProps {
  compact?: boolean
  t: Theme
  text: string
}
