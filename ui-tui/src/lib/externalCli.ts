import { spawn } from 'node:child_process'

export interface LaunchResult {
  code: null | number
  error?: string
}

// Set by `ddeharness tui` to the entry that started this session. The commands run
// here write credentials, and PATH can name a different install than the one
// this TUI belongs to -- which would write them where it will not look.
const resolveOpenDDEHarnessBin = () => process.env.OPENDDE_HARNESS_BIN?.trim() || 'ddeharness'

export const launchOpenDDEHarnessCommand = (args: string[]): Promise<LaunchResult> =>
  new Promise(resolve => {
    const child = spawn(resolveOpenDDEHarnessBin(), args, { stdio: 'inherit' })

    child.on('error', err => resolve({ code: null, error: err.message }))
    child.on('exit', code => resolve({ code }))
  })
