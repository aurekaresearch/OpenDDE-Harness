import type { SlashExecResponse } from '../../../gatewayTypes.js'
import type { SlashCommand } from '../types.js'

export const setupCommands: SlashCommand[] = [
  {
    aliases: ['onboard'],
    help: 'configuration is changed with ddeharness onboard in a terminal',
    name: 'setup',
    run: (_arg, ctx) =>
      ctx.transcript.sys('Exit the TUI and run `ddeharness onboard` in a terminal to change configuration.')
  },
  {
    help: 'check configuration and compute resources; --fix prepares missing resources',
    name: 'doctor',
    run: (_arg, ctx, cmd) => {
      ctx.gateway.gw
        .request<SlashExecResponse>('slash.exec', { command: cmd.slice(1), session_id: ctx.sid })
        .then(r => {
          if (ctx.stale()) {
            return
          }

          const body = r?.output || '/doctor: no output'
          const text = r?.warning ? `warning: ${r.warning}\n${body}` : body
          const long = text.length > 180 || text.split('\n').filter(Boolean).length > 2

          if (long) {
            ctx.transcript.page(text, 'Doctor')
          } else {
            ctx.transcript.sys(text)
          }
        })
        .catch(ctx.guardedErr)
    }
  }
]
