// SPDX-License-Identifier: MIT
// Portions Copyright (c) original ink contributors (vadimdemedes/ink, MIT).
// Portions Copyright (c) 2025 Nous Research (hermes-agent / hermes-ink, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-{hermes-agent,ink}.txt.

import { describe, expect, it } from 'vitest'

import { INITIAL_STATE, type KeyParseState, parseMultipleKeypresses } from '../parse-keypress.js'

import { InputEvent } from './input-event.js'

// Feed one chunk, then flush until the parser gives up its buffer — the App
// watchdog path for a sequence whose tail never arrives.
const parseWithFlushes = (chunk: string) => {
  let state: KeyParseState = INITIAL_STATE
  let [keys, next] = parseMultipleKeypresses(state, chunk)
  state = next

  while (state.incomplete) {
    ;[keys, next] = parseMultipleKeypresses(state, null)
    state = next
  }

  return keys.map(k => new InputEvent(k as never))
}

describe('InputEvent text extraction', () => {
  it('never turns a truncated SGR mouse report into input text', () => {
    const [event] = parseWithFlushes('\x1b[<64;100;')

    expect(event?.input).toBe('')
    expect(event?.key.escape).toBe(false)
  })

  it('never turns a truncated OSC reply into input text', () => {
    const [event] = parseWithFlushes('\x1b]11;rgb:1111/2222/')

    expect(event?.input).toBe('')
  })

  it('keeps plain and meta text input intact', () => {
    expect(parseWithFlushes('abc')[0]?.input).toBe('abc')
    expect(parseWithFlushes('\x1bx')[0]?.input).toBe('x')
    expect(parseWithFlushes('\x1b[')[0]?.input).toBe('[')
    expect(parseWithFlushes('[<64;100;23M')[0]?.input).toBe('')
  })
})
