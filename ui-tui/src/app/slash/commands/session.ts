// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import type {
  ConfigGetValueResponse,
  ModelOverlayResponse,
  ConfigSetResponse,
  SessionBranchResponse,
  SessionExportResponse
} from '../../../gatewayTypes.js'
import type { SlashCommand } from '../types.js'

import { DEFAULT_INDICATOR_STYLE, INDICATOR_STYLES, type IndicatorStyle } from '../../interfaces.js'
import { patchOverlayState } from '../../overlayStore.js'
import { patchUiState } from '../../uiStore.js'

// v1 model switch is global-scope only. The picker passes `<model> --provider
// <slug>`; a bare `/model <name>` carries no provider. Parse both into the
// structured config.set params {key:'model', value, provider?}.
const parseModelArg = (arg: string): { provider?: string; value: string } => {
  const m = arg.trim().match(/^(.*?)\s+--provider\s+(\S+)\s*$/)

  if (m) {
    return { provider: m[2], value: m[1]!.trim() }
  }

  return { value: arg.trim() }
}

// The wire contract is the full `<channel>:<chat_id>` key; users naturally
// type the bare chat_id they see in filenames, so default those to tui:.
const normalizeSessionId = (id: string) => (id.includes(':') ? id : `tui:${id}`)

export const sessionCommands: SlashCommand[] = [
  {
    help: 'change or show model',
    name: 'model',
    run: (arg, ctx) => {
      if (ctx.session.guardBusySessionSwitch('change models')) {
        return
      }

      if (!arg.trim()) {
        return patchOverlayState({ modelPicker: true })
      }

      const { provider, value } = parseModelArg(arg)

      ctx.gateway
        .rpc<ConfigSetResponse>('config.set', {
          key: 'model',
          session_id: ctx.sid,
          value,
          ...(provider ? { provider } : {})
        })
        .then(
          ctx.guarded<ConfigSetResponse>(r => {
            if (!r.value) {
              return ctx.transcript.sys('error: invalid response: model switch')
            }

            ctx.transcript.sys(`model → ${r.value}`)
            ctx.local.maybeWarn(r)

            patchUiState(state => ({
              ...state,
              info: state.info ? { ...state.info, model: r.value! } : { model: r.value!, skills: {}, tools: {} }
            }))
          })
        )
    }
  },

  {
    help: 'browse/manage sessions: list, new [title], resume [id], delete [id|current]',
    name: 'sessions',
    run: (arg, ctx) => {
      const parts = arg.trim().split(/\s+/)
      const sub = parts[0]?.toLowerCase() ?? ''
      const rest = parts.slice(1).join(' ').trim()

      if (!sub || sub === 'list') {
        if (ctx.session.guardBusySessionSwitch('switch sessions')) {
          return
        }

        return patchOverlayState({ picker: true })
      }

      if (sub === 'new') {
        if (ctx.session.guardBusySessionSwitch('switch sessions')) {
          return
        }

        ctx.session.newSession('New session started', rest || undefined)

        return
      }

      if (sub === 'resume') {
        if (ctx.session.guardBusySessionSwitch('switch sessions')) {
          return
        }

        if (rest) {
          ctx.session.resumeById(rest)
        } else {
          patchOverlayState({ picker: true })
        }

        return
      }

      if (sub === 'delete') {
        const targetId = rest === 'current' || !rest ? ctx.sid : normalizeSessionId(rest)

        if (!targetId) {
          return ctx.transcript.sys('no active session to delete')
        }

        // Deleting the active session implies a session switch (fallback to
        // most-recent / fresh), so it needs the same busy guard as /resume.
        const isActive = targetId === ctx.sid

        if (isActive && ctx.session.guardBusySessionSwitch('delete the active session')) {
          return
        }

        void ctx.session
          .deleteSessionWithFallback(targetId)
          .then(removed => {
            // Active deletes announce themselves by switching the session;
            // non-active deletes have no visible effect without this note.
            if (!isActive) {
              ctx.transcript.sys(removed ? `deleted session: ${targetId}` : `no such session: ${targetId}`)
            }
          })
          .catch((e: unknown) => {
            const msg = e instanceof Error ? e.message : String(e)
            ctx.transcript.sys(`error: ${msg}`)
          })

        return
      }

      ctx.transcript.sys('usage: /sessions [list|new [title]|resume [id]|delete [id|current]]')
    }
  },

  {
    help: 'switch the agent personality for this session',
    name: 'personality',
    supported: false,
    run: (arg, ctx) => {
      if (!arg) {
        return
      }

      ctx.gateway.rpc<ConfigSetResponse>('config.set', { key: 'personality', session_id: ctx.sid, value: arg }).then(
        ctx.guarded<ConfigSetResponse>(r => {
          if (r.history_reset) {
            ctx.session.resetVisibleHistory(r.info ?? null)
          }

          ctx.transcript.sys(`personality: ${r.value || 'default'}${r.history_reset ? ' · transcript cleared' : ''}`)
          ctx.local.maybeWarn(r)
        })
      )
    }
  },

  {
    aliases: ['fork'],
    help: 'branch the session',
    name: 'branch',
    run: (arg, ctx) => {
      const prevSid = ctx.sid

      ctx.gateway.rpc<SessionBranchResponse>('session.branch', { name: arg, session_id: ctx.sid }).then(
        ctx.guarded<SessionBranchResponse>(r => {
          if (!r.session_id) {
            return
          }

          void ctx.session.closeSession(prevSid)
          patchUiState({ sid: r.session_id })
          ctx.session.setSessionStartedAt(Date.now())
          // Keep the on-screen history: the forked child is a full copy of the
          // displayed session, so what's already shown is correct for the child.
          const bareSessionId = (k: string) => (k.includes(':') ? k.slice(k.indexOf(':') + 1) : k)
          const messageCount = r.message_count ?? 0
          const title = r.title || '(untitled)'
          ctx.transcript.sys(`⑂ Forked "${title}" · ${messageCount} message${messageCount === 1 ? '' : 's'} carried`)
          ctx.transcript.sys(`   parent  ${prevSid ? bareSessionId(prevSid) : '(none)'}`)
          ctx.transcript.sys(`   forked  ${bareSessionId(r.session_id)}`)
        })
      )
    }
  },

  {
    help: 'export the session transcript to a markdown file',
    name: 'export',
    usage: '/export [id]',
    run: (arg, ctx) => {
      // Pass the raw id so the backend resolves it cross-channel (a bare id
      // must not be forced to tui: here, or cross-channel resolution breaks).
      const target = arg.trim() || ctx.sid

      if (!target) {
        return ctx.transcript.sys('no active session to export')
      }

      ctx.gateway.rpc<SessionExportResponse>('session.export', { session_id: target }).then(
        ctx.guarded<SessionExportResponse>(r => {
          if (r.exported && r.path) {
            return ctx.transcript.sys(`✓ exported to ${r.path}`)
          }

          if (r.reason === 'ambiguous') {
            return ctx.transcript.sys(`ambiguous session id — candidates: ${(r.candidates ?? []).join(', ')}`)
          }

          if (r.reason === 'write_failed') {
            return ctx.transcript.sys('error: failed to write export file')
          }

          return ctx.transcript.sys(`no such session: ${arg.trim() || target}`)
        })
      )
    }
  },

  {
    help: 'switch theme skin (fires skin.changed)',
    name: 'skin',
    supported: false,
    run: (arg, ctx) => {
      if (!arg) {
        return ctx.gateway
          .rpc<ConfigGetValueResponse>('config.get', { key: 'skin' })
          .then(ctx.guarded<ConfigGetValueResponse>(r => ctx.transcript.sys(`skin: ${r.value || 'default'}`)))
      }

      ctx.gateway
        .rpc<ConfigSetResponse>('config.set', { key: 'skin', value: arg })
        .then(ctx.guarded<ConfigSetResponse>(r => r.value && ctx.transcript.sys(`skin → ${r.value}`)))
    }
  },

  {
    help: 'pick the busy indicator: kaomoji (default), emoji, unicode (braille), or ascii',
    name: 'indicator',
    supported: false,
    usage: `/indicator [${INDICATOR_STYLES.join('|')}]`,
    run: (arg, ctx) => {
      const value = arg.trim().toLowerCase()

      if (!value) {
        return ctx.gateway
          .rpc<ConfigGetValueResponse>('config.get', { key: 'indicator' })
          .then(
            ctx.guarded<ConfigGetValueResponse>(r =>
              ctx.transcript.sys(`indicator: ${r.value || DEFAULT_INDICATOR_STYLE}`)
            )
          )
      }

      if (!(INDICATOR_STYLES as readonly string[]).includes(value)) {
        return ctx.transcript.sys(`usage: /indicator [${INDICATOR_STYLES.join('|')}]`)
      }

      ctx.gateway.rpc<ConfigSetResponse>('config.set', { key: 'indicator', value }).then(
        ctx.guarded<ConfigSetResponse>(r => {
          if (!r.value) {
            return
          }

          // Hot-swap the running TUI immediately so the next render
          // uses the new style without waiting for the 5s mtime poll
          // to re-apply config.full.
          patchUiState({ indicatorStyle: value as IndicatorStyle })
          ctx.transcript.sys(`indicator → ${r.value}`)
        })
      )
    }
  },

  {
    help: 'toggle yolo mode (per-session approvals)',
    name: 'yolo',
    supported: false,
    run: (_arg, ctx) => {
      ctx.gateway
        .rpc<ConfigSetResponse>('config.set', { key: 'yolo', session_id: ctx.sid })
        .then(ctx.guarded<ConfigSetResponse>(r => ctx.transcript.sys(`yolo ${r.value === '1' ? 'on' : 'off'}`)))
    }
  },

  {
    aliases: ['reasoning'],
    help: "set the current model's thinking level (off|minimal|low|medium|high|xhigh|max|default)",
    name: 'thinking',
    run: (arg, ctx) => {
      const level = arg.trim().toLowerCase()

      if (!level) {
        const current = ctx.ui.info?.reasoning_effort || 'default'

        return ctx.transcript.sys(
          `thinking: ${current} · usage: /thinking off|minimal|low|medium|high|xhigh|max|default`
        )
      }

      ctx.gateway
        .rpc<ModelOverlayResponse>('model.overlay', { field: 'reasoning_effort', session_id: ctx.sid, value: level })
        .then(
          ctx.guarded<ModelOverlayResponse>(r => {
            const effort = typeof r.value === 'string' ? r.value : undefined

            patchUiState(state => ({
              ...state,
              info: state.info ? { ...state.info, reasoning_effort: effort } : state.info
            }))
            ctx.transcript.sys(`thinking: ${effort ?? 'default'} for ${r.model}`)
          })
        )
    },
    usage: '/thinking <level>'
  },

  {
    help: "declare the current model's context window in tokens (e.g. 128k), or default",
    name: 'context',
    run: (arg, ctx) => {
      const value = arg.trim().toLowerCase()

      if (!value) {
        const max = ctx.ui.info?.usage?.context_max

        return ctx.transcript.sys(`context: ${max ? `${max} tokens` : 'unknown'} · usage: /context <tokens|default>`)
      }

      ctx.gateway
        .rpc<ModelOverlayResponse>('model.overlay', { field: 'context_window_tokens', session_id: ctx.sid, value })
        .then(
          ctx.guarded<ModelOverlayResponse>(r => {
            const max = r.context_window_tokens ?? 0

            patchUiState(state => ({
              ...state,
              info: state.info?.usage
                ? {
                    ...state.info,
                    usage: { ...state.info.usage, context_max: max, context_source: max ? 'overlay' : 'unknown' }
                  }
                : state.info
            }))
            ctx.transcript.sys(`context: ${max ? `${max} tokens` : 'unknown'} for ${r.model}`)
          })
        )
    },
    usage: '/context <tokens>'
  },

  {
    help: 'toggle fast mode [normal|fast|status|on|off|toggle]',
    name: 'fast',
    supported: false,
    run: (arg, ctx) => {
      const mode = arg.trim().toLowerCase()
      const valid = new Set(['', 'status', 'normal', 'fast', 'on', 'off', 'toggle'])

      if (!valid.has(mode)) {
        return ctx.transcript.sys('usage: /fast [normal|fast|status|on|off|toggle]')
      }

      if (!mode || mode === 'status') {
        return ctx.gateway
          .rpc<ConfigGetValueResponse>('config.get', { key: 'fast', session_id: ctx.sid })
          .then(
            ctx.guarded<ConfigGetValueResponse>(r =>
              ctx.transcript.sys(`fast mode: ${r.value === 'fast' ? 'fast' : 'normal'}`)
            )
          )
          .catch(ctx.guardedErr)
      }

      ctx.gateway
        .rpc<ConfigSetResponse>('config.set', { key: 'fast', session_id: ctx.sid, value: mode })
        .then(
          ctx.guarded<ConfigSetResponse>(r => {
            const next = r.value === 'fast' ? 'fast' : 'normal'
            ctx.transcript.sys(`fast mode: ${next}`)
            patchUiState(state => ({
              ...state,
              info: state.info
                ? {
                    ...state.info,
                    fast: next === 'fast',
                    service_tier: next === 'fast' ? 'priority' : ''
                  }
                : state.info
            }))
          })
        )
        .catch(ctx.guardedErr)
    }
  },

  {
    help: 'control busy enter mode [queue|interrupt|status]',
    name: 'busy',
    supported: false,
    run: (arg, ctx) => {
      const mode = arg.trim().toLowerCase()
      const valid = new Set(['', 'status', 'queue', 'interrupt'])

      if (!valid.has(mode)) {
        return ctx.transcript.sys('usage: /busy [queue|interrupt|status]')
      }

      if (!mode || mode === 'status') {
        return ctx.gateway
          .rpc<ConfigGetValueResponse>('config.get', { key: 'busy' })
          .then(
            ctx.guarded<ConfigGetValueResponse>(r => {
              const current = r.value || 'interrupt'
              ctx.transcript.sys(`busy input mode: ${current}`)
            })
          )
          .catch(ctx.guardedErr)
      }

      ctx.gateway
        .rpc<ConfigSetResponse>('config.set', { key: 'busy', value: mode })
        .then(
          ctx.guarded<ConfigSetResponse>(r => {
            const next = r.value || mode
            ctx.transcript.sys(`busy input mode: ${next}`)
          })
        )
        .catch(ctx.guardedErr)
    }
  },

  {
    help: 'cycle verbose tool-output mode (updates live agent)',
    name: 'verbose',
    supported: false,
    run: (arg, ctx) => {
      ctx.gateway
        .rpc<ConfigSetResponse>('config.set', { key: 'verbose', session_id: ctx.sid, value: arg || 'cycle' })
        .then(ctx.guarded<ConfigSetResponse>(r => r.value && ctx.transcript.sys(`verbose: ${r.value}`)))
    }
  }
]
