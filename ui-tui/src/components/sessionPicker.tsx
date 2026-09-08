// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import { Box, Text, useInput, useStdout } from '@hermes/ink'
import { useEffect, useState } from 'react'

import type { GatewayClient } from '../gatewayClientStub.js'
import type { SessionDeleteResponse, SessionListItem, SessionListResponse } from '../gatewayTypes.js'
import type { Theme } from '../theme.js'

import { overlayPaneRows } from '../lib/overlayMetrics.js'
import { asRpcResult, rpcErrorMessage } from '../lib/rpc.js'
import { OverlayHint, useOverlayKeys, windowOffset } from './overlayControls.js'

const VISIBLE = 15
const MIN_WIDTH = 60
const MARKER_WIDTH = 2
const MIN_PREVIEW_WIDTH = 8

const age = (ts: number) => {
  const d = (Date.now() / 1000 - ts) / 86400

  if (d < 1) {
    return 'today'
  }

  if (d < 2) {
    return 'yesterday'
  }

  return `${Math.floor(d)}d ago`
}

export function SessionPicker({ activeSid, gw, onCancel, onDeleteActive, onSelect, t }: SessionPickerProps) {
  const [items, setItems] = useState<SessionListItem[]>([])
  const [err, setErr] = useState('')
  const [sel, setSel] = useState(0)
  const [loading, setLoading] = useState(true)
  // When non-null, the user pressed `d` on this index and we're waiting for
  // a second `d`/`D` to confirm deletion.  Any other key cancels the prompt.
  const [confirmDelete, setConfirmDelete] = useState<null | number>(null)
  const [deleting, setDeleting] = useState(false)

  const { stdout } = useStdout()
  // The pane the picker renders in spans the terminal minus its margins,
  // border and padding; the row count leaves the pane's chrome on screen.
  const width = Math.max(MIN_WIDTH, (stdout?.columns ?? 80) - 6)
  const visible = Math.min(VISIBLE, overlayPaneRows(stdout?.rows ?? 24))
  // Every cell has a fixed width and the four add up to the row, so no cell
  // is ever flex-shrunk to a fractional width: a shrunk text wraps onto a
  // second line and the whole row grows to two lines.
  const idWidth = Math.max(...items.map(s => s.id.length), 0) + 7
  const metaWidth =
    Math.max(...items.map(s => `(${s.message_count} msgs, ${age(s.started_at)}, ${s.source || 'tui'})`.length), 0) + 2
  const previewWidth = Math.max(MIN_PREVIEW_WIDTH, width - MARKER_WIDTH - idWidth - metaWidth)

  useOverlayKeys({ onClose: onCancel })

  useEffect(() => {
    gw.request<SessionListResponse>('session.list', { limit: 200 })
      .then(raw => {
        const r = asRpcResult<SessionListResponse>(raw)

        if (!r) {
          setErr('invalid response: session.list')
          setLoading(false)

          return
        }

        setItems(r.sessions ?? [])
        setErr('')
        setLoading(false)
      })
      .catch((e: unknown) => {
        setErr(rpcErrorMessage(e))
        setLoading(false)
      })
  }, [gw])

  const performDelete = (index: number) => {
    const target = items[index]

    if (!target || deleting) {
      return
    }

    setDeleting(true)

    if (onDeleteActive && activeSid === target.id) {
      // The fallback switches session and closes the picker, so this
      // component unmounts — skip the local list bookkeeping entirely.
      onDeleteActive(target.id).catch((e: unknown) => {
        setErr(rpcErrorMessage(e))
        setDeleting(false)
      })

      return
    }

    gw.request<SessionDeleteResponse>('session.delete', { session_id: target.id })
      .then(raw => {
        const r = asRpcResult<SessionDeleteResponse>(raw)

        if (!r) {
          setErr('invalid response: session.delete')
          setDeleting(false)

          return
        }

        if (r.deleted === null) {
          setErr(`no such session: ${target.id}`)
          setDeleting(false)

          return
        }

        if (r.deleted !== target.id) {
          setErr('invalid response: session.delete')
          setDeleting(false)

          return
        }

        setItems(prev => {
          const next = prev.filter((_, i) => i !== index)
          setSel(s => Math.max(0, Math.min(s, next.length - 1)))

          return next
        })
        setErr('')
        setDeleting(false)
      })
      .catch((e: unknown) => {
        setErr(rpcErrorMessage(e))
        setDeleting(false)
      })
  }

  useInput((ch, key) => {
    if (deleting) {
      return
    }

    if (confirmDelete !== null) {
      if (ch?.toLowerCase() === 'd') {
        const idx = confirmDelete
        setConfirmDelete(null)
        performDelete(idx)
      } else {
        setConfirmDelete(null)
      }

      return
    }

    if (key.upArrow && sel > 0) {
      setSel(s => s - 1)
    }

    if (key.downArrow && sel < items.length - 1) {
      setSel(s => s + 1)
    }

    if (key.return && items[sel]) {
      onSelect(items[sel]!.id)

      return
    }

    if (ch?.toLowerCase() === 'd' && items[sel]) {
      setConfirmDelete(sel)

      return
    }

    const n = parseInt(ch)

    if (n >= 1 && n <= Math.min(9, items.length)) {
      onSelect(items[n - 1]!.id)
    }
  })

  if (loading) {
    return <Text color={t.color.muted}>loading sessions…</Text>
  }

  if (err && !items.length) {
    return (
      <Box flexDirection="column">
        <Text color={t.color.label}>error: {err}</Text>
        <OverlayHint t={t}>Esc/q cancel</OverlayHint>
      </Box>
    )
  }

  if (!items.length) {
    return (
      <Box flexDirection="column">
        <Text color={t.color.muted}>no previous sessions</Text>
        <OverlayHint t={t}>Esc/q cancel</OverlayHint>
      </Box>
    )
  }

  const offset = windowOffset(items.length, sel, visible)

  return (
    <Box flexDirection="column" width={width}>
      <Text bold color={t.color.accent}>
        Resume Session
      </Text>

      {offset > 0 && <Text color={t.color.muted}> ↑ {offset} more</Text>}

      {items.slice(offset, offset + visible).map((s, vi) => {
        const i = offset + vi
        const selected = sel === i
        const pendingDelete = confirmDelete === i

        return (
          <Box key={s.id}>
            <Box flexShrink={0} width={MARKER_WIDTH}>
              <Text bold={selected} color={selected ? t.color.accent : t.color.muted} inverse={selected}>
                {selected ? '▸ ' : '  '}
              </Text>
            </Box>

            <Box flexShrink={0} width={idWidth}>
              <Text
                bold={selected}
                color={selected ? t.color.accent : t.color.muted}
                inverse={selected}
                wrap="truncate-end"
              >
                {String(i + 1).padStart(2)}. [{s.id}]
              </Text>
            </Box>

            <Box flexShrink={0} width={metaWidth}>
              <Text
                bold={selected}
                color={selected ? t.color.accent : t.color.muted}
                inverse={selected}
                wrap="truncate-end"
              >
                ({s.message_count} msgs, {age(s.started_at)}, {s.source || 'tui'})
              </Text>
            </Box>

            <Box flexShrink={0} width={previewWidth}>
              <Text
                bold={selected}
                color={pendingDelete ? t.color.label : selected ? t.color.accent : t.color.muted}
                inverse={selected}
                wrap="truncate-end"
              >
                {pendingDelete ? 'press d again to delete' : s.title || s.preview || '(untitled)'}
              </Text>
            </Box>
          </Box>
        )
      })}

      {offset + visible < items.length && <Text color={t.color.muted}> ↓ {items.length - offset - visible} more</Text>}
      {err && <Text color={t.color.label}>error: {err}</Text>}
      {deleting ? (
        <OverlayHint t={t}>deleting…</OverlayHint>
      ) : (
        <OverlayHint t={t}>↑/↓ select · Enter resume · 1-9 quick · d delete · Esc/q cancel</OverlayHint>
      )}
    </Box>
  )
}

interface SessionPickerProps {
  activeSid?: null | string
  gw: GatewayClient
  onCancel: () => void
  onDeleteActive?: (id: string) => Promise<unknown>
  onSelect: (id: string) => void
  t: Theme
}
