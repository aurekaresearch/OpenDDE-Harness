import { describe, expect, it } from 'vitest'

import type { CtrlCState } from '../app/useInputHandlers.js'

import { decideCtrlC } from '../app/useInputHandlers.js'

const ctrlC = (over: Partial<CtrlCState> = {}): CtrlCState => ({
  busyWithSession: false,
  escapeArmed: false,
  hasPendingInput: false,
  turnActive: false,
  ...over
})

describe('decideCtrlC', () => {
  it('quits only from an idle UI with an empty composer', () => {
    expect(decideCtrlC(ctrlC())).toBe('quit')
  })

  it('clears a pending line instead of quitting', () => {
    expect(decideCtrlC(ctrlC({ hasPendingInput: true }))).toBe('clear-input')
  })

  it('cancels the turn on the first press while one is in flight', () => {
    expect(decideCtrlC(ctrlC({ busyWithSession: true, turnActive: true }))).toBe('cancel-turn')
  })

  it('force-resets on the second press when the cancel produced no terminal event', () => {
    expect(decideCtrlC(ctrlC({ busyWithSession: true, escapeArmed: true, turnActive: true }))).toBe('force-reset')
  })

  it('falls back to the legacy interrupt when busy without a typed turn', () => {
    expect(decideCtrlC(ctrlC({ busyWithSession: true }))).toBe('interrupt-legacy')
  })

  it('never quits while busy, whatever the composer holds', () => {
    for (const hasPendingInput of [false, true]) {
      for (const turnActive of [false, true]) {
        expect(decideCtrlC(ctrlC({ busyWithSession: true, hasPendingInput, turnActive }))).not.toBe('quit')
      }
    }
  })

  it('ignores escapeArmed unless a turn is actually active', () => {
    expect(decideCtrlC(ctrlC({ escapeArmed: true }))).toBe('quit')
    expect(decideCtrlC(ctrlC({ busyWithSession: true, escapeArmed: true }))).toBe('interrupt-legacy')
  })
})
