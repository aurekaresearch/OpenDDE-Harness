// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import { beforeEach, describe, expect, it, vi } from 'vitest'

import { createSlashHandler } from '../app/createSlashHandler.js'
import { getOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'

describe('createSlashHandler', () => {
  beforeEach(() => {
    resetOverlayState()
    resetUiState()
  })

  it('opens the resume picker locally', () => {
    const ctx = buildCtx()

    expect(createSlashHandler(ctx)('/resume')).toBe(true)
    expect(getOverlayState().picker).toBe(true)
  })

  it('handles /redraw locally without slash worker fallback', () => {
    const ctx = buildCtx()

    expect(createSlashHandler(ctx)('/redraw')).toBe(true)
    expect(ctx.gateway.gw.request).not.toHaveBeenCalled()
    expect(ctx.transcript.sys).toHaveBeenCalledWith('ui redrawn')
  })

  it('points /onboard and /setup at the terminal wizard without calling the gateway', () => {
    const ctx = buildCtx()
    const handler = createSlashHandler(ctx)

    expect(handler('/onboard')).toBe(true)
    expect(handler('/setup --reset')).toBe(true)
    expect(ctx.gateway.gw.request).not.toHaveBeenCalled()
    expect(ctx.gateway.rpc).not.toHaveBeenCalled()
    expect(ctx.transcript.sys).toHaveBeenCalledTimes(2)
    expect(ctx.transcript.sys).toHaveBeenCalledWith(
      'Exit the TUI and run `ddeharness onboard` in a terminal to change configuration.'
    )
  })

  it('runs /doctor through the CLI slash worker', async () => {
    patchUiState({ sid: 'sid-abc' })
    const request = vi.fn(() => Promise.resolve({ output: 'all good' }))
    const ctx = buildCtx({ gateway: { ...buildGateway(), gw: { getLogTail: vi.fn(() => ''), request } } })

    expect(createSlashHandler(ctx)('/doctor --fix')).toBe(true)
    expect(request).toHaveBeenCalledWith('slash.exec', { command: 'doctor --fix', session_id: 'sid-abc' })
    await vi.waitFor(() => {
      expect(ctx.transcript.sys).toHaveBeenCalledWith('all good')
    })
  })

  it('exits locally for /quit', () => {
    const ctx = buildCtx()

    expect(createSlashHandler(ctx)('/quit')).toBe(true)
    expect(ctx.session.die).toHaveBeenCalledTimes(1)
    expect(ctx.gateway.gw.request).not.toHaveBeenCalled()
  })

  it('routes /status to live session.status instead of slash worker', async () => {
    patchUiState({ sid: 'sid-abc' })
    const rpc = vi.fn(() => Promise.resolve({ output: 'OpenDDE Harness TUI Status' }))
    const ctx = buildCtx({ gateway: { ...buildGateway(), rpc } })

    expect(createSlashHandler(ctx)('/status')).toBe(true)
    expect(rpc).toHaveBeenCalledWith('session.status', { session_id: 'sid-abc' })
    expect(ctx.gateway.gw.request).not.toHaveBeenCalled()
    await vi.waitFor(() => {
      expect(ctx.transcript.page).toHaveBeenCalledWith('OpenDDE Harness TUI Status', 'Status')
    })
  })

  it('sends a bare /model switch with no provider param', async () => {
    patchUiState({ sid: 'sid-abc' })

    const ctx = buildCtx({
      gateway: {
        ...buildGateway(),
        rpc: vi.fn(() => Promise.resolve({ applied: true, previous: null, value: 'x-model' }))
      }
    })

    expect(createSlashHandler(ctx)('/model x-model')).toBe(true)
    expect(ctx.gateway.rpc).toHaveBeenCalledWith('config.set', {
      key: 'model',
      session_id: 'sid-abc',
      value: 'x-model'
    })
  })

  it('parses a --provider suffix into a structured provider param', async () => {
    patchUiState({ sid: 'sid-abc' })

    const ctx = buildCtx({
      gateway: {
        ...buildGateway(),
        rpc: vi.fn(() => Promise.resolve({ applied: true, previous: null, value: 'claude-sonnet-4.6' }))
      }
    })

    expect(createSlashHandler(ctx)('/model claude-sonnet-4.6 --provider openrouter')).toBe(true)
    expect(ctx.gateway.rpc).toHaveBeenCalledWith('config.set', {
      key: 'model',
      provider: 'openrouter',
      session_id: 'sid-abc',
      value: 'claude-sonnet-4.6'
    })
  })

  it('applies /reasoning hide to the thinking section immediately', async () => {
    patchUiState({ sections: { thinking: 'expanded' }, showReasoning: true, sid: 'sid-abc' })

    const ctx = buildCtx({
      gateway: {
        ...buildGateway(),
        rpc: vi.fn(() => Promise.resolve({ value: 'hide' }))
      }
    })

    expect(createSlashHandler(ctx)('/reasoning hide')).toBe(true)

    await vi.waitFor(() => {
      expect(getUiState().showReasoning).toBe(false)
      expect(getUiState().sections.thinking).toBe('hidden')
    })
    expect(ctx.gateway.rpc).toHaveBeenCalledWith('config.set', {
      key: 'reasoning',
      session_id: 'sid-abc',
      value: 'hide'
    })
  })

  it('applies /reasoning show to the thinking section immediately', async () => {
    patchUiState({ sections: { thinking: 'hidden' }, showReasoning: false, sid: 'sid-abc' })

    const ctx = buildCtx({
      gateway: {
        ...buildGateway(),
        rpc: vi.fn(() => Promise.resolve({ value: 'show' }))
      }
    })

    expect(createSlashHandler(ctx)('/reasoning show')).toBe(true)

    await vi.waitFor(() => {
      expect(getUiState().showReasoning).toBe(true)
      expect(getUiState().sections.thinking).toBe('expanded')
    })
  })

  it('opens the skills hub locally for bare /skills', () => {
    const ctx = buildCtx()

    expect(createSlashHandler(ctx)('/skills')).toBe(true)
    expect(getOverlayState().skillsHub).toBe(true)
    expect(ctx.gateway.rpc).not.toHaveBeenCalled()
    expect(ctx.gateway.gw.request).not.toHaveBeenCalled()
  })

  it('routes /skills install <name> to skills.manage without opening overlay', () => {
    const ctx = buildCtx()

    expect(createSlashHandler(ctx)('/skills install foo')).toBe(true)
    expect(getOverlayState().skillsHub).toBe(false)
    expect(ctx.gateway.rpc).toHaveBeenCalledWith('skills.manage', {
      action: 'install',
      query: 'foo'
    })
  })

  it('routes /skills inspect <name> to skills.manage', () => {
    const ctx = buildCtx()

    createSlashHandler(ctx)('/skills inspect my-skill')
    expect(ctx.gateway.rpc).toHaveBeenCalledWith('skills.manage', {
      action: 'inspect',
      query: 'my-skill'
    })
  })

  it('routes /skills search <query> to skills.manage', () => {
    const ctx = buildCtx()

    createSlashHandler(ctx)('/skills search vibe')
    expect(ctx.gateway.rpc).toHaveBeenCalledWith('skills.manage', {
      action: 'search',
      query: 'vibe'
    })
  })

  it('routes /skills browse [page] to skills.manage with a numeric page', () => {
    const ctx = buildCtx()

    createSlashHandler(ctx)('/skills browse 3')
    expect(ctx.gateway.rpc).toHaveBeenCalledWith('skills.manage', {
      action: 'browse',
      page: 3
    })
  })

  it('delegates non-native /skills subcommands to slash.exec', () => {
    const ctx = buildCtx()

    createSlashHandler(ctx)('/skills check')
    expect(ctx.gateway.rpc).not.toHaveBeenCalled()
    expect(ctx.gateway.gw.request).toHaveBeenCalledWith('slash.exec', {
      command: 'skills check',
      session_id: null
    })
  })

  it('passes /new <title> through to the session lifecycle', () => {
    const ctx = buildCtx()

    createSlashHandler(ctx)('/new sprint planning')
    getOverlayState().confirm?.onConfirm()

    expect(ctx.session.newSession).toHaveBeenCalledWith('New session started', 'sprint planning')
    expect(ctx.gateway.rpc).not.toHaveBeenCalled()
  })

  it('cycles details mode and persists it', async () => {
    const ctx = buildCtx()

    expect(getUiState().detailsMode).toBe('collapsed')
    expect(createSlashHandler(ctx)('/details toggle')).toBe(true)
    expect(getUiState().detailsMode).toBe('expanded')
    expect(getUiState().detailsModeCommandOverride).toBe(true)
    expect(getUiState().sections).toEqual({
      thinking: 'expanded',
      tools: 'expanded',
      subagents: 'expanded',
      activity: 'expanded'
    })
    expect(ctx.gateway.rpc).toHaveBeenCalledWith('config.set', {
      key: 'details_mode',
      value: 'expanded'
    })
    expect(ctx.transcript.sys).toHaveBeenCalledWith('details: expanded')
  })

  it('sets a per-section override and persists it under details_mode.<section>', () => {
    const ctx = buildCtx()

    expect(createSlashHandler(ctx)('/details activity hidden')).toBe(true)
    expect(getUiState().sections.activity).toBe('hidden')
    expect(ctx.gateway.rpc).toHaveBeenCalledWith('config.set', {
      key: 'details_mode.activity',
      value: 'hidden'
    })
    expect(ctx.transcript.sys).toHaveBeenCalledWith('details activity: hidden')
  })

  it('clears a per-section override on /details <section> reset', () => {
    const ctx = buildCtx()
    createSlashHandler(ctx)('/details tools expanded')
    expect(getUiState().sections.tools).toBe('expanded')

    createSlashHandler(ctx)('/details tools reset')
    expect(getUiState().sections.tools).toBeUndefined()
    expect(ctx.gateway.rpc).toHaveBeenLastCalledWith('config.set', {
      key: 'details_mode.tools',
      value: ''
    })
    expect(ctx.transcript.sys).toHaveBeenCalledWith('details tools: reset')
  })

  it('rejects unknown section modes with a usage hint', () => {
    const ctx = buildCtx()
    createSlashHandler(ctx)('/details tools blink')
    expect(getUiState().sections.tools).toBeUndefined()
    expect(ctx.transcript.sys).toHaveBeenCalledWith('usage: /details <section> [hidden|collapsed|expanded|reset]')
  })

  it.each([
    ['/reload-mcp', 'reload.mcp', { session_id: null }],
    ['/fast status', 'config.get', { key: 'fast', session_id: null }],
    ['/busy status', 'config.get', { key: 'busy' }],
    ['/indicator', 'config.get', { key: 'indicator' }]
  ])('routes %s through native RPC (no slash worker)', (command, method, params) => {
    const rpc = vi.fn(() => Promise.resolve({}))
    const ctx = buildCtx({ gateway: { ...buildGateway(), rpc } })

    expect(createSlashHandler(ctx)(command)).toBe(true)
    expect(rpc).toHaveBeenCalledWith(method, params)
    expect(ctx.gateway.gw.request).not.toHaveBeenCalled()
  })

  it('hot-swaps the live indicator when /indicator <style> succeeds', async () => {
    const rpc = vi.fn(() => Promise.resolve({ value: 'emoji' }))
    const ctx = buildCtx({ gateway: { ...buildGateway(), rpc } })

    expect(createSlashHandler(ctx)('/indicator emoji')).toBe(true)
    expect(rpc).toHaveBeenCalledWith('config.set', { key: 'indicator', value: 'emoji' })
    await vi.waitFor(() => expect(getUiState().indicatorStyle).toBe('emoji'))
  })

  it('rejects unknown indicator styles before hitting the gateway', () => {
    const rpc = vi.fn(() => Promise.resolve({}))
    const ctx = buildCtx({ gateway: { ...buildGateway(), rpc } })

    expect(createSlashHandler(ctx)('/indicator sparkle')).toBe(true)
    expect(rpc).not.toHaveBeenCalled()
    expect(ctx.transcript.sys).toHaveBeenCalledWith('usage: /indicator [ascii|emoji|kaomoji|unicode]')
  })

  it('drops stale slash.exec output after a newer slash', async () => {
    let resolveLate: (v: { output?: string }) => void
    let slashExecCalls = 0

    const ctx = buildCtx({
      gateway: {
        gw: {
          getLogTail: vi.fn(() => ''),
          request: vi.fn((method: string) => {
            if (method === 'slash.exec') {
              slashExecCalls += 1

              if (slashExecCalls === 1) {
                return new Promise<{ output?: string }>(res => {
                  resolveLate = res
                })
              }

              return Promise.resolve({ output: 'fresh' })
            }

            return Promise.resolve({})
          })
        },
        rpc: vi.fn(() => Promise.resolve({}))
      }
    })

    const h = createSlashHandler(ctx)
    expect(h('/slow')).toBe(true)
    expect(h('/later')).toBe(true)
    resolveLate!({ output: 'too late' })
    await vi.waitFor(() => {
      expect(ctx.transcript.sys).toHaveBeenCalled()
    })

    expect(ctx.transcript.sys).not.toHaveBeenCalledWith('too late')
  })

  it('dispatches command.dispatch with typed alias', async () => {
    const ctx = buildCtx({
      gateway: {
        gw: {
          getLogTail: vi.fn(() => ''),
          request: vi.fn((method: string) => {
            if (method === 'slash.exec') {
              return Promise.reject(new Error('no'))
            }

            if (method === 'command.dispatch') {
              return Promise.resolve({ type: 'alias', target: 'help' })
            }

            return Promise.resolve({})
          })
        },
        rpc: vi.fn(() => Promise.resolve({}))
      }
    })

    const h = createSlashHandler(ctx)
    expect(h('/zzz')).toBe(true)
    await vi.waitFor(() => {
      expect(ctx.transcript.panel).toHaveBeenCalledWith(expect.any(String), expect.any(Array))
    })
  })

  it('resolves unique local aliases through the catalog', () => {
    const ctx = buildCtx({
      local: {
        catalog: {
          canon: {
            '/h': '/help',
            '/help': '/help'
          }
        }
      }
    })

    expect(createSlashHandler(ctx)('/h')).toBe(true)
    expect(ctx.transcript.panel).toHaveBeenCalledWith(expect.any(String), expect.any(Array))
  })

  it('lets exact catalog commands win over longer prefix matches', async () => {
    const ctx = buildCtx({
      local: {
        catalog: {
          canon: {
            '/profile': '/profile',
            '/plugins': '/plugins'
          }
        }
      }
    })

    expect(createSlashHandler(ctx)('/profile')).toBe(true)
    await vi.waitFor(() => {
      expect(ctx.gateway.gw.request).toHaveBeenCalledWith('slash.exec', {
        command: 'profile',
        session_id: null
      })
    })
    expect(ctx.transcript.sys).not.toHaveBeenCalledWith(expect.stringContaining('ambiguous command'))
  })

  it('keeps ambiguous prefix handling when there is no exact catalog match', () => {
    const ctx = buildCtx({
      local: {
        catalog: {
          canon: {
            '/status': '/status',
            '/statusbar': '/statusbar'
          }
        }
      }
    })

    expect(createSlashHandler(ctx)('/stat')).toBe(true)
    expect(ctx.transcript.sys).toHaveBeenCalledWith('ambiguous command: /status, /statusbar')
    expect(ctx.gateway.gw.request).not.toHaveBeenCalled()
  })

  it('falls through to command.dispatch for skill commands and sends the message', async () => {
    const skillMessage = 'Use this skill to do X.\n\n## Steps\n1. First step'

    const ctx = buildCtx({
      gateway: {
        gw: {
          getLogTail: vi.fn(() => ''),
          request: vi.fn((method: string) => {
            if (method === 'slash.exec') {
              return Promise.reject(new Error('skill command: use command.dispatch'))
            }

            if (method === 'command.dispatch') {
              return Promise.resolve({ type: 'skill', message: skillMessage, name: 'opendde-agent-dev' })
            }

            return Promise.resolve({})
          })
        },
        rpc: vi.fn(() => Promise.resolve({}))
      }
    })

    const h = createSlashHandler(ctx)
    expect(h('/opendde-agent-dev')).toBe(true)
    await vi.waitFor(() => {
      expect(ctx.transcript.sys).toHaveBeenCalledWith('⚡ loading skill: opendde-agent-dev')
    })
    expect(ctx.transcript.send).toHaveBeenCalledWith(skillMessage)
  })

  it('/history pages the current TUI transcript (user + assistant)', () => {
    const ctx = buildCtx({
      local: {
        ...buildLocal(),
        getHistoryItems: vi.fn(() => [
          { role: 'user', text: 'hello' },
          { role: 'system', text: 'ignore me' },
          { role: 'assistant', text: 'hi there' },
          { role: 'user', text: 'test' }
        ])
      }
    })

    createSlashHandler(ctx)('/history')
    expect(ctx.transcript.page).toHaveBeenCalledTimes(1)

    const [body, title] = ctx.transcript.page.mock.calls[0]!

    expect(title).toBe('History')
    expect(body).toContain('[You #1]')
    expect(body).toContain('hello')
    expect(body).toContain('[OpenDDE Harness #2]')
    expect(body).toContain('hi there')
    expect(body).toContain('[You #3]')
    expect(body).not.toContain('ignore me')
    expect(ctx.gateway.gw.request).not.toHaveBeenCalled()
  })

  it('/history reports empty state without paging', () => {
    const ctx = buildCtx()

    createSlashHandler(ctx)('/history')
    expect(ctx.transcript.page).not.toHaveBeenCalled()
    expect(ctx.transcript.sys).toHaveBeenCalledWith('no conversation yet')
  })

  it('/title <name> uses session.title RPC and bypasses slash.exec', async () => {
    patchUiState({ sid: 'sid-abc' })
    const rpc = vi.fn(() => Promise.resolve({ pending: false, title: 'my title' }))
    const ctx = buildCtx({ gateway: { ...buildGateway(), rpc } })

    createSlashHandler(ctx)('/title my title')

    expect(rpc).toHaveBeenCalledWith('session.title', { session_id: 'sid-abc', title: 'my title' })
    expect(ctx.gateway.gw.request).not.toHaveBeenCalled()
    await vi.waitFor(() => {
      expect(ctx.transcript.sys).toHaveBeenCalledWith('session title set: my title')
    })
  })

  it('/title with no args fetches and displays the current title', async () => {
    patchUiState({ sid: 'sid-abc' })
    const rpc = vi.fn(() => Promise.resolve({ title: 'demo title' }))
    const ctx = buildCtx({ gateway: { ...buildGateway(), rpc } })

    createSlashHandler(ctx)('/title')

    expect(rpc).toHaveBeenCalledWith('session.title', { session_id: 'sid-abc' })
    expect(ctx.gateway.gw.request).not.toHaveBeenCalled()
    await vi.waitFor(() => {
      expect(ctx.transcript.sys).toHaveBeenCalledWith('title: demo title')
    })
  })

  it('/fork keeps the transcript and appends a lineage confirmation with both bare ids', async () => {
    patchUiState({ sid: 'tui:20260616_074815_3d2dca' })
    const rpc = vi.fn(() =>
      Promise.resolve({
        session_id: 'tui:20260616_075351_e17cd7',
        title: 'Greeting (fork)',
        message_count: 3
      })
    )
    const ctx = buildCtx({ gateway: { ...buildGateway(), rpc } })

    createSlashHandler(ctx)('/fork')

    expect(rpc).toHaveBeenCalledWith('session.branch', {
      name: '',
      session_id: 'tui:20260616_074815_3d2dca'
    })
    await vi.waitFor(() => {
      // transcript is NOT cleared (history preserved)
      expect(ctx.transcript.setHistoryItems).not.toHaveBeenCalled()
      const lines = (ctx.transcript.sys as ReturnType<typeof vi.fn>).mock.calls.map(c => String(c[0])).join('\n')
      expect(lines).toContain('Forked')
      expect(lines).toContain('Greeting (fork)')
      expect(lines).toContain('3 messages carried')
      // both bare ids present, channel prefix stripped
      expect(lines).toContain('20260616_074815_3d2dca')
      expect(lines).toContain('20260616_075351_e17cd7')
      expect(lines).not.toContain('tui:')
    })
  })

  it('/fork singularizes the carried-message count', async () => {
    patchUiState({ sid: 'tui:20260616_074815_3d2dca' })
    const rpc = vi.fn(() => Promise.resolve({ session_id: 'tui:child', title: 't (fork)', message_count: 1 }))
    const ctx = buildCtx({ gateway: { ...buildGateway(), rpc } })

    createSlashHandler(ctx)('/fork')

    await vi.waitFor(() => {
      const lines = (ctx.transcript.sys as ReturnType<typeof vi.fn>).mock.calls.map(c => String(c[0])).join('\n')
      expect(lines).toContain('1 message carried')
      expect(lines).not.toContain('1 messages carried')
    })
  })

  it('/fork shows a (none) placeholder when the parent id is empty', async () => {
    // Guards the defensive prevSid-empty branch: the backend never returns a
    // truthy session_id for an empty source, so this co-occurrence is
    // unreachable in production, but the renderer must not print a blank parent.
    const rpc = vi.fn(() => Promise.resolve({ session_id: 'tui:child', title: 't (fork)', message_count: 0 }))
    const ctx = buildCtx({ gateway: { ...buildGateway(), rpc } })

    createSlashHandler(ctx)('/fork')

    await vi.waitFor(() => {
      const lines = (ctx.transcript.sys as ReturnType<typeof vi.fn>).mock.calls.map(c => String(c[0])).join('\n')
      expect(lines).toContain('parent  (none)')
    })
  })
})

const buildCtx = (overrides: Partial<Ctx> = {}): Ctx => ({
  ...overrides,
  slashFlightRef: overrides.slashFlightRef ?? { current: 0 },
  composer: { ...buildComposer(), ...overrides.composer },
  gateway: { ...buildGateway(), ...overrides.gateway },
  local: { ...buildLocal(), ...overrides.local },
  session: { ...buildSession(), ...overrides.session },
  transcript: { ...buildTranscript(), ...overrides.transcript }
})

const buildComposer = () => ({
  enqueue: vi.fn(),
  hasSelection: false,
  paste: vi.fn(),
  queueRef: { current: [] as string[] },
  selection: { copySelection: vi.fn(async () => '') },
  setInput: vi.fn()
})

const buildGateway = () => ({
  gw: {
    getLogTail: vi.fn(() => ''),
    request: vi.fn(() => Promise.resolve({}))
  },
  rpc: vi.fn(() => Promise.resolve({}))
})

const buildLocal = () => ({
  catalog: null,
  getHistoryItems: vi.fn(() => []),
  getLastUserMsg: vi.fn(() => ''),
  maybeWarn: vi.fn(),
  setCatalog: vi.fn()
})

const buildSession = () => ({
  closeSession: vi.fn(() => Promise.resolve(null)),
  deleteSessionWithFallback: vi.fn(() => Promise.resolve(true)),
  die: vi.fn(),
  guardBusySessionSwitch: vi.fn(() => false),
  newSession: vi.fn(),
  resetVisibleHistory: vi.fn(),
  resumeById: vi.fn(),
  setSessionStartedAt: vi.fn()
})

const buildTranscript = () => ({
  page: vi.fn(),
  panel: vi.fn(),
  send: vi.fn(),
  setHistoryItems: vi.fn(),
  sys: vi.fn(),
  trimLastExchange: vi.fn(items => items)
})

interface Ctx {
  slashFlightRef: { current: number }
  composer: ReturnType<typeof buildComposer>
  gateway: ReturnType<typeof buildGateway>
  local: ReturnType<typeof buildLocal>
  session: ReturnType<typeof buildSession>
  transcript: ReturnType<typeof buildTranscript>
}
