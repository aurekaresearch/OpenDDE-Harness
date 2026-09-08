// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import type { DelegationPauseResponse, ReloadMcpResponse, SlashExecResponse } from '../../../gatewayTypes.js'
import type { PanelSection } from '../../../types.js'
import type { SlashCommand } from '../types.js'

import { applyDelegationStatus, getDelegationState } from '../../delegationStore.js'
import { patchOverlayState } from '../../overlayStore.js'
import {
  focusProteinDesignTask,
  formatProteinDesignTask,
  proteinDesignTaskList,
  readProteinDesignTaskLog,
  refreshProteinDesignTasks,
  resolveProteinDesignTask
} from '../../proteinDesignTaskStore.js'
import { getSpawnHistory, setDiffPair, type SpawnSnapshot } from '../../spawnHistoryStore.js'

interface SkillInfo {
  category?: string
  description?: string
  name?: string
  path?: string
}

interface SkillsListResponse {
  skills?: Record<string, string[]>
}

interface SkillsInspectResponse {
  info?: SkillInfo
}

interface SkillsSearchResponse {
  results?: { description?: string; name: string }[]
}

interface SkillsInstallResponse {
  installed?: boolean
  name?: string
}

interface SkillsBrowseItem {
  description?: string
  name: string
  source?: string
  trust?: string
}

interface SkillsBrowseResponse {
  items?: SkillsBrowseItem[]
  page?: number
  total?: number
  total_pages?: number
}

export const opsCommands: SlashCommand[] = [
  {
    help: 'list background protein-design tasks',
    name: 'tasks',
    run: (_arg, ctx) => {
      void refreshProteinDesignTasks()
        .then(() => {
          const tasks = proteinDesignTaskList().slice(0, 20)

          if (!tasks.length) {
            ctx.transcript.sys('no protein-design tasks found')
            return
          }

          const lines = tasks.map(task => {
            const best = task.bestObjective == null ? '—' : task.bestObjective.toFixed(4)
            return `${task.status.padEnd(9)} ${task.target.padEnd(16)} ${task.cycle}/${task.totalCycles || '?'}  ${task.phase || 'setup'}  best ${best}  ${task.taskId.slice(0, 8)}`
          })
          ctx.transcript.page(lines.join('\n'), 'Protein Design Tasks')
        })
        .catch(ctx.guardedErr)
    }
  },
  {
    help: 'inspect, focus, or read logs for a protein-design task',
    name: 'task',
    usage: '/task <id> | /task follow <id> | /task logs <id>',
    run: (arg, ctx) => {
      const parts = arg.trim().split(/\s+/).filter(Boolean)

      void refreshProteinDesignTasks()
        .then(async () => {
          const action = parts[0] === 'follow' || parts[0] === 'logs' ? parts.shift() : 'show'
          const query = parts[0] || ''
          const task = resolveProteinDesignTask(query)

          if (!task) {
            ctx.transcript.sys(
              query ? `unknown or ambiguous task: ${query}` : 'usage: /task <id> | /task follow <id> | /task logs <id>'
            )
            return
          }

          if (action === 'follow') {
            focusProteinDesignTask(task.taskId)
            ctx.transcript.sys(`following ${task.target} · ${task.taskId.slice(0, 8)} in the background task bar`)
            return
          }

          if (action === 'logs') {
            const body = await readProteinDesignTaskLog(task.taskId)
            ctx.transcript.page(body, `${task.target} · ${task.taskId.slice(0, 8)} · worker log`)
            return
          }

          ctx.transcript.page(formatProteinDesignTask(task), `${task.target} · ${task.taskId.slice(0, 8)}`)
        })
        .catch(ctx.guardedErr)
    }
  },
  {
    help: 'open the tracing dashboard (LLM/tool/memory spans)',
    name: 'tracing',
    run: (_arg, ctx) => {
      ctx.gateway
        .rpc<SlashExecResponse>('slash.exec', { command: 'tracing', session_id: ctx.sid })
        .then(
          ctx.guarded<SlashExecResponse>(r => {
            const body = r?.output || 'tracing dashboard'
            ctx.transcript.sys(r?.warning ? `warning: ${r.warning}\n${body}` : body)
          })
        )
        .catch(ctx.guardedErr)
    }
  },
  {
    aliases: ['reload_mcp'],
    help: 'reload MCP servers in the live session (warns about prompt cache invalidation)',
    name: 'reload-mcp',
    supported: false,
    run: (arg, ctx) => {
      // Parse arg: `now` / `always` skip the confirmation gate.
      // `always` additionally persists approvals.mcp_reload_confirm=false.
      const a = (arg || '').trim().toLowerCase()

      const params: { session_id: string | null; confirm?: boolean; always?: boolean } = {
        session_id: ctx.sid
      }

      if (a === 'now' || a === 'approve' || a === 'once' || a === 'yes') {
        params.confirm = true
      } else if (a === 'always') {
        params.confirm = true
        params.always = true
      }

      ctx.gateway
        .rpc<ReloadMcpResponse>('reload.mcp', params)
        .then(
          ctx.guarded<ReloadMcpResponse>(r => {
            if (r.status === 'confirm_required') {
              ctx.transcript.sys(r.message || '/reload-mcp requires confirmation')

              return
            }

            if (r.status === 'reloaded') {
              ctx.transcript.sys(
                params.always
                  ? 'MCP servers reloaded · future /reload-mcp will run without confirmation'
                  : 'MCP servers reloaded'
              )

              return
            }

            ctx.transcript.sys('reload complete')
          })
        )
        .catch(ctx.guardedErr)
    }
  },

  {
    aliases: ['tasks'],
    help: 'open the spawn-tree dashboard (live audit + kill/pause controls)',
    name: 'agents',
    supported: false,
    run: (arg, ctx) => {
      const sub = arg.trim().toLowerCase()

      // Stay compatible with the gateway `/agents [pause|resume|status]` CLI —
      // explicit subcommands skip the overlay and act directly so scripts and
      // multi-step flows can drive it without entering interactive mode.
      if (sub === 'pause' || sub === 'resume' || sub === 'unpause') {
        const paused = sub === 'pause'
        ctx.gateway.gw
          .request<DelegationPauseResponse>('delegation.pause', { paused })
          .then(r => {
            applyDelegationStatus({ paused: r?.paused })
            ctx.transcript.sys(`delegation · ${r?.paused ? 'paused' : 'resumed'}`)
          })
          .catch(ctx.guardedErr)

        return
      }

      if (sub === 'status') {
        const d = getDelegationState()
        ctx.transcript.sys(
          `delegation · ${d.paused ? 'paused' : 'active'} · caps d${d.maxSpawnDepth ?? '?'}/${d.maxConcurrentChildren ?? '?'}`
        )

        return
      }

      patchOverlayState({ agents: true, agentsInitialHistoryIndex: 0 })
    }
  },

  {
    help: 'replay a completed spawn tree · `/replay [N|last]`',
    name: 'replay',
    supported: false,
    run: (arg, ctx) => {
      const history = getSpawnHistory()
      const raw = arg.trim()
      const lower = raw.toLowerCase()

      if (!history.length) {
        return ctx.transcript.sys('no completed spawn trees this session')
      }

      let index = 1

      if (raw && lower !== 'last') {
        const parsed = parseInt(raw, 10)

        if (Number.isNaN(parsed) || parsed < 1 || parsed > history.length) {
          return ctx.transcript.sys(`replay: index out of range 1..${history.length}`)
        }

        index = parsed
      }

      patchOverlayState({ agents: true, agentsInitialHistoryIndex: index })
    }
  },

  {
    help: 'diff two completed spawn trees · `/replay-diff <baseline> <candidate>` (history indexes)',
    name: 'replay-diff',
    supported: false,
    run: (arg, ctx) => {
      const parts = arg.trim().split(/\s+/).filter(Boolean)

      if (parts.length !== 2) {
        return ctx.transcript.sys('usage: /replay-diff <a> <b>  (e.g. /replay-diff 1 2 for last two)')
      }

      const [a, b] = parts
      const history = getSpawnHistory()

      const resolve = (token: string): null | SpawnSnapshot => {
        const n = parseInt(token!, 10)

        if (Number.isFinite(n) && n >= 1 && n <= history.length) {
          return history[n - 1] ?? null
        }

        return null
      }

      const baseline = resolve(a!)
      const candidate = resolve(b!)

      if (!baseline || !candidate) {
        return ctx.transcript.sys(`replay-diff: could not resolve indices · history has ${history.length} entries`)
      }

      setDiffPair({ baseline, candidate })
      patchOverlayState({ agents: true, agentsInitialHistoryIndex: 0 })
    }
  },

  {
    help: 'browse, inspect, install skills',
    name: 'skills',
    supported: false,
    run: (arg, ctx, cmd) => {
      const text = arg.trim()

      if (!text) {
        return patchOverlayState({ skillsHub: true })
      }

      const [sub, ...rest] = text.split(/\s+/)
      const query = rest.join(' ').trim()
      const { rpc } = ctx.gateway
      const { panel, sys } = ctx.transcript

      const runViaSlashWorker = () => {
        ctx.gateway.gw
          .request<SlashExecResponse>('slash.exec', { command: cmd.slice(1), session_id: ctx.sid })
          .then(r => {
            if (ctx.stale()) {
              return
            }

            const body = r?.output || '/skills: no output'
            const formatted = r?.warning ? `warning: ${r.warning}\n${body}` : body
            const long = formatted.length > 180 || formatted.split('\n').filter(Boolean).length > 2

            long ? ctx.transcript.page(formatted, 'Skills') : ctx.transcript.sys(formatted)
          })
          .catch(ctx.guardedErr)
      }

      if (sub === 'list') {
        rpc<SkillsListResponse>('skills.manage', { action: 'list' })
          .then(
            ctx.guarded<SkillsListResponse>(r => {
              const cats = Object.entries(r.skills ?? {}).sort()

              if (!cats.length) {
                return sys('no skills available')
              }

              panel(
                'Skills',
                cats.map<PanelSection>(([title, items]) => ({ items, title }))
              )
            })
          )
          .catch(ctx.guardedErr)

        return
      }

      if (sub === 'inspect') {
        if (!query) {
          return sys('usage: /skills inspect <name>')
        }

        rpc<SkillsInspectResponse>('skills.manage', { action: 'inspect', query })
          .then(
            ctx.guarded<SkillsInspectResponse>(r => {
              const info = r.info ?? {}

              if (!info.name) {
                return sys(`unknown skill: ${query}`)
              }

              const rows: [string, string][] = [
                ['Name', String(info.name)],
                ['Category', String(info.category ?? '')],
                ['Path', String(info.path ?? '')]
              ]

              const sections: PanelSection[] = [{ rows }]

              if (info.description) {
                sections.push({ text: String(info.description) })
              }

              panel('Skill', sections)
            })
          )
          .catch(ctx.guardedErr)

        return
      }

      if (sub === 'search') {
        if (!query) {
          return sys('usage: /skills search <query>')
        }

        rpc<SkillsSearchResponse>('skills.manage', { action: 'search', query })
          .then(
            ctx.guarded<SkillsSearchResponse>(r => {
              const results = r.results ?? []

              if (!results.length) {
                return sys(`no results for: ${query}`)
              }

              panel(`Search: ${query}`, [{ rows: results.map(s => [s.name, s.description ?? '']) }])
            })
          )
          .catch(ctx.guardedErr)

        return
      }

      if (sub === 'install') {
        if (!query) {
          return sys('usage: /skills install <name or url>')
        }

        sys(`installing ${query}…`)

        rpc<SkillsInstallResponse>('skills.manage', { action: 'install', query })
          .then(
            ctx.guarded<SkillsInstallResponse>(r =>
              sys(r.installed ? `installed ${r.name ?? query}` : 'install failed')
            )
          )
          .catch(ctx.guardedErr)

        return
      }

      if (sub === 'browse') {
        const pageNum = query ? parseInt(query, 10) : 1

        if (Number.isNaN(pageNum) || pageNum < 1) {
          return sys('usage: /skills browse [page]  (page must be a positive number)')
        }

        sys('fetching community skills (scans 6 sources, may take ~15s)…')

        rpc<SkillsBrowseResponse>('skills.manage', { action: 'browse', page: pageNum })
          .then(
            ctx.guarded<SkillsBrowseResponse>(r => {
              const items = r.items ?? []

              if (!items.length) {
                return sys(`no skills on page ${pageNum}${r.total ? ` (total ${r.total})` : ''}`)
              }

              const rows: [string, string][] = items.map(s => [
                s.trust ? `${s.name} · ${s.trust}` : s.name,
                String(s.description ?? '').slice(0, 160)
              ])

              const footer: string[] = []

              if (r.page && r.total_pages) {
                footer.push(`page ${r.page} of ${r.total_pages}`)
              }

              if (r.total) {
                footer.push(`${r.total} skills total`)
              }

              if (r.page && r.total_pages && r.page < r.total_pages) {
                footer.push(`/skills browse ${r.page + 1} for more`)
              }

              panel(`Browse Skills${pageNum > 1 ? ` — p${pageNum}` : ''}`, [
                { rows },
                ...(footer.length ? [{ text: footer.join(' · ') }] : [])
              ])
            })
          )
          .catch(ctx.guardedErr)

        return
      }

      runViaSlashWorker()
    }
  },

  {
    help: 'list or configure tools',
    name: 'tools',
    supported: false,
    run: (_arg, ctx, cmd) => {
      ctx.gateway.gw
        .request<SlashExecResponse>('slash.exec', { command: cmd.slice(1), session_id: ctx.sid })
        .then(r => {
          if (ctx.stale()) {
            return
          }

          const body = r?.output || '/tools: no output'
          const text = r?.warning ? `warning: ${r.warning}\n${body}` : body
          const long = text.length > 180 || text.split('\n').filter(Boolean).length > 2

          long ? ctx.transcript.page(text, 'Tools') : ctx.transcript.sys(text)
        })
        .catch(ctx.guardedErr)
    }
  }
]
