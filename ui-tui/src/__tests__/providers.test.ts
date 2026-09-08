// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import { describe, expect, it } from 'vitest'

import { bareModelId, providerDisplayNames } from '../domain/providers.js'

describe('providerDisplayNames', () => {
  it('returns bare names when all are unique', () => {
    expect(
      providerDisplayNames([
        { name: 'Anthropic', slug: 'anthropic' },
        { name: 'OpenAI', slug: 'openai' }
      ])
    ).toEqual(['Anthropic', 'OpenAI'])
  })

  it('appends slug to every collision so the disambiguation is symmetric', () => {
    expect(
      providerDisplayNames([
        { name: 'Kimi For Coding', slug: 'kimi-coding' },
        { name: 'Kimi For Coding', slug: 'kimi-coding-cn' }
      ])
    ).toEqual(['Kimi For Coding (kimi-coding)', 'Kimi For Coding (kimi-coding-cn)'])
  })

  it('only disambiguates the colliding group', () => {
    expect(
      providerDisplayNames([
        { name: 'Anthropic', slug: 'anthropic' },
        { name: 'Foo', slug: 'foo-a' },
        { name: 'Foo', slug: 'foo-b' }
      ])
    ).toEqual(['Anthropic', 'Foo (foo-a)', 'Foo (foo-b)'])
  })

  it('falls back to plain name if slug is empty', () => {
    expect(
      providerDisplayNames([
        { name: 'Foo', slug: '' },
        { name: 'Foo', slug: '' }
      ])
    ).toEqual(['Foo', 'Foo'])
  })

  it('skips disambiguation when slug equals name', () => {
    expect(
      providerDisplayNames([
        { name: 'foo', slug: 'foo' },
        { name: 'foo', slug: 'foo' }
      ])
    ).toEqual(['foo', 'foo'])
  })

  it('handles empty input', () => {
    expect(providerDisplayNames([])).toEqual([])
  })

  it('preserves order', () => {
    const input = [
      { name: 'Z', slug: 'z' },
      { name: 'A', slug: 'a1' },
      { name: 'A', slug: 'a2' }
    ]

    expect(providerDisplayNames(input)).toEqual(['Z', 'A (a1)', 'A (a2)'])
  })
})

describe('bareModelId', () => {
  it('drops the prefix only for a provider the user pointed at an endpoint', () => {
    const custom = { auth_type: 'endpoint', slug: 'custom' }
    const vendor = { auth_type: 'key', slug: 'deepseek' }

    expect(bareModelId(custom, 'custom/gpt-4o')).toBe('gpt-4o')
    expect(bareModelId(custom, 'gpt-4o')).toBe('gpt-4o')
    expect(bareModelId(vendor, 'deepseek/deepseek-v4-flash')).toBe('deepseek/deepseek-v4-flash')
    expect(bareModelId(undefined, 'custom/gpt-4o')).toBe('custom/gpt-4o')
  })
})
