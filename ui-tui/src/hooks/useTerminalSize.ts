// SPDX-License-Identifier: MIT
// Copyright (c) 2026 EverMind.
// See NOTICES.md.

import { useStdout } from '@hermes/ink'
import { useCallback, useMemo, useSyncExternalStore } from 'react'

const DEFAULT_COLUMNS = 80
const DEFAULT_ROWS = 24

export interface TerminalSize {
  columns: number
  rows: number
}

// Current terminal size, re-rendering the caller on `resize`. Reading
// `stdout.columns` during render alone goes stale: React only re-runs a
// component when its own state or props change, not when the tty does.
export function useTerminalSize(): TerminalSize {
  const { stdout } = useStdout()

  const subscribe = useCallback(
    (onChange: () => void) => {
      if (!stdout) {
        return () => {}
      }

      stdout.on('resize', onChange)

      return () => {
        stdout.off('resize', onChange)
      }
    },
    [stdout]
  )

  // Snapshot as a primitive so unchanged sizes compare equal and skip renders.
  // The first render reads the size synchronously; when the stream has none
  // yet, fall back to process.stdout, then to the COLUMNS/LINES env the launcher
  // passes through, before the defaults.
  const getSnapshot = useCallback(
    () =>
      `${stdout?.columns || process.stdout.columns || Number(process.env.COLUMNS) || DEFAULT_COLUMNS}x${
        stdout?.rows || process.stdout.rows || Number(process.env.LINES) || DEFAULT_ROWS
      }`,
    [stdout]
  )

  const key = useSyncExternalStore(subscribe, getSnapshot, getSnapshot)

  return useMemo(() => {
    const [columns, rows] = key.split('x').map(Number)

    return { columns: columns!, rows: rows! }
  }, [key])
}
