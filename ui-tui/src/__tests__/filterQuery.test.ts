// SPDX-License-Identifier: MIT
// Copyright (c) 2026 EverMind.
// See NOTICES.md.

import { describe, expect, it } from 'vitest'

import { matchesQuery, matchingIndices } from '../domain/filterQuery.js'

describe('matchesQuery', () => {
  it('keeps every row when nothing is typed', () => {
    expect(matchesQuery('anything', '')).toBe(true)
    expect(matchesQuery('anything', '   ')).toBe(true)
  })

  it('matches case-insensitively on any part of the row', () => {
    expect(matchesQuery('DeepSeek-V4-Flash', 'flash')).toBe(true)
    expect(matchesQuery('DeepSeek-V4-Flash', 'FLASH')).toBe(true)
    expect(matchesQuery('DeepSeek-V4-Flash', 'seek-v4')).toBe(true)
    expect(matchesQuery('DeepSeek-V4-Flash', 'terra')).toBe(false)
  })

  it('asks for every word, in any order', () => {
    expect(matchesQuery('custom · DeepSeek-V4-Flash', 'flash custom')).toBe(true)
    expect(matchesQuery('custom · DeepSeek-V4-Flash', 'flash openai')).toBe(false)
  })

  it('stays a substring match, so ids that differ by a digit stay apart', () => {
    // A relay lists hundreds of near-identical ids; a fuzzy match would put
    // gpt-5.6 under a query for 5.2 and the wrong model would be one Enter away.
    expect(matchesQuery('gpt-5.6-terra', '5.2')).toBe(false)
    expect(matchesQuery('gpt-5.6-terra', 'gt6')).toBe(false)
  })
})

describe('matchingIndices', () => {
  it('reports positions in the original list, in order', () => {
    const rows = ['claude-sonnet-4-6', 'gpt-5.6-terra', 'deepseek-v4-flash']

    expect(matchingIndices(rows, '')).toEqual([0, 1, 2])
    expect(matchingIndices(rows, 'e')).toEqual([0, 1, 2])
    expect(matchingIndices(rows, 'v4')).toEqual([2])
    expect(matchingIndices(rows, 'nothing')).toEqual([])
  })
})
