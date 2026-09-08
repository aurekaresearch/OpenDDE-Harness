import { describe, expect, it } from 'vitest'

import { shouldPassThroughToGlobalHandler } from '../components/textInput.js'

const key = (overrides: Record<string, unknown> = {}) => ({ ctrl: false, meta: false, ...overrides }) as any

describe('shouldPassThroughToGlobalHandler', () => {
  it('does not swallow ordinary typing keys', () => {
    expect(shouldPassThroughToGlobalHandler('h', key())).toBe(false)
    expect(shouldPassThroughToGlobalHandler('o', key())).toBe(false)
    expect(shouldPassThroughToGlobalHandler('b', key({ ctrl: true }))).toBe(false)
  })

  it('always passes through global control keys', () => {
    expect(shouldPassThroughToGlobalHandler('c', key({ ctrl: true }))).toBe(true)
    expect(shouldPassThroughToGlobalHandler('x', key({ ctrl: true }))).toBe(true)
    expect(shouldPassThroughToGlobalHandler('', key({ escape: true }))).toBe(true)
    expect(shouldPassThroughToGlobalHandler('', key({ tab: true }))).toBe(true)
    expect(shouldPassThroughToGlobalHandler('', key({ pageUp: true }))).toBe(true)
    expect(shouldPassThroughToGlobalHandler('', key({ pageDown: true }))).toBe(true)
  })
})
