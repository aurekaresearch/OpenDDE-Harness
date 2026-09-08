// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import { Box, Text } from '@hermes/ink'
import { useStore } from '@nanostores/react'

import type { AppOverlaysProps } from '../app/interfaces.js'

import { useGateway } from '../app/gatewayContext.js'
import { $overlayState, patchOverlayState } from '../app/overlayStore.js'
import { $uiSessionId, $uiTheme } from '../app/uiStore.js'
import { launchOpenDDEHarnessCommand } from '../lib/externalCli.js'
import { suspendForHandoff } from '../lib/handoff.js'
import { FloatBox } from './appChrome.js'
import { ModelPicker } from './modelPicker.js'
import { OverlayHint } from './overlayControls.js'
import { ApprovalPrompt, ClarifyPrompt, ConfirmPrompt } from './prompts.js'
import { SessionPicker } from './sessionPicker.js'
import { SkillsHub } from './skillsHub.js'

const COMPLETION_WINDOW = 16

export function PromptZone({
  cols,
  onApprovalChoice,
  onClarifyAnswer,
  onConfirmAnswer
}: Pick<AppOverlaysProps, 'cols' | 'onApprovalChoice' | 'onClarifyAnswer' | 'onConfirmAnswer'>) {
  const overlay = useStore($overlayState)
  const theme = useStore($uiTheme)

  if (overlay.approval) {
    return (
      <Box flexDirection="column" flexShrink={0} paddingX={1} paddingY={1}>
        <ApprovalPrompt onChoice={onApprovalChoice} req={overlay.approval} t={theme} />
      </Box>
    )
  }

  if (overlay.confirm) {
    const req = overlay.confirm
    const isRpc = Boolean(req.requestId)

    const onConfirm = () => {
      if (isRpc) {
        onConfirmAnswer(true)

        return
      }

      patchOverlayState({ confirm: null })
      req.onConfirm?.()
    }

    const onCancel = () => {
      if (isRpc) {
        onConfirmAnswer(false)

        return
      }

      patchOverlayState({ confirm: null })
    }

    return (
      <Box flexDirection="column" flexShrink={0} paddingX={1} paddingY={1}>
        <ConfirmPrompt onCancel={onCancel} onConfirm={onConfirm} req={req} t={theme} />
      </Box>
    )
  }

  if (overlay.clarify) {
    return (
      <Box flexDirection="column" flexShrink={0} paddingX={1} paddingY={1}>
        <ClarifyPrompt
          key={overlay.clarify.requestId}
          cols={cols}
          onAnswer={onClarifyAnswer}
          onCancel={() => onClarifyAnswer('')}
          req={overlay.clarify}
          t={theme}
        />
      </Box>
    )
  }

  return null
}

export function OverlayPane({
  onModelSelect,
  onPickerDeleteActive,
  onPickerSelect,
  pagerPageSize
}: Pick<AppOverlaysProps, 'onModelSelect' | 'onPickerDeleteActive' | 'onPickerSelect' | 'pagerPageSize'>) {
  const { gw } = useGateway()
  const overlay = useStore($overlayState)
  const sid = useStore($uiSessionId)
  const theme = useStore($uiTheme)

  return (
    <Box flexDirection="column" flexGrow={1} paddingTop={1} paddingX={1}>
      <Box borderColor={theme.color.border} borderStyle="round" flexDirection="column" paddingX={1}>
        {overlay.picker ? (
          <SessionPicker
            activeSid={sid}
            gw={gw}
            onCancel={() => patchOverlayState({ picker: false })}
            onDeleteActive={onPickerDeleteActive}
            onSelect={onPickerSelect}
            t={theme}
          />
        ) : overlay.modelPicker ? (
          <ModelPicker
            gw={gw}
            launcher={launchOpenDDEHarnessCommand}
            onCancel={() => patchOverlayState({ modelPicker: false })}
            onSelect={onModelSelect}
            sessionId={sid}
            suspend={suspendForHandoff}
            t={theme}
          />
        ) : overlay.skillsHub ? (
          <SkillsHub gw={gw} onClose={() => patchOverlayState({ skillsHub: false })} t={theme} />
        ) : overlay.pager ? (
          <Box flexDirection="column" paddingY={1}>
            {overlay.pager.title && (
              <Box justifyContent="center" marginBottom={1}>
                <Text bold color={theme.color.primary}>
                  {overlay.pager.title}
                </Text>
              </Box>
            )}

            {overlay.pager.lines.slice(overlay.pager.offset, overlay.pager.offset + pagerPageSize).map((line, i) => (
              <Text key={i}>{line}</Text>
            ))}

            <Box marginTop={1}>
              <OverlayHint t={theme}>
                {overlay.pager.offset + pagerPageSize < overlay.pager.lines.length
                  ? `↑↓/jk line · Enter/Space/PgDn page · b/PgUp back · g/G top/bottom · Esc/q close (${Math.min(overlay.pager.offset + pagerPageSize, overlay.pager.lines.length)}/${overlay.pager.lines.length})`
                  : `end · ↑↓/jk · b/PgUp back · g top · Esc/q close (${overlay.pager.lines.length} lines)`}
              </OverlayHint>
            </Box>
          </Box>
        ) : null}
      </Box>
    </Box>
  )
}

export function FloatingOverlays({
  cols,
  compIdx,
  completions
}: Pick<AppOverlaysProps, 'cols' | 'compIdx' | 'completions'>) {
  const theme = useStore($uiTheme)

  if (!completions.length) {
    return null
  }

  // Fixed viewport centered on compIdx — previously the slice end was
  // compIdx + 8 so the dropdown grew from 8 rows to 16 as the user scrolled
  // down, bouncing the height on every keystroke.
  const viewportSize = Math.min(COMPLETION_WINDOW, completions.length)

  const start = Math.max(0, Math.min(compIdx - Math.floor(COMPLETION_WINDOW / 2), completions.length - viewportSize))

  return (
    <Box alignItems="flex-start" bottom="100%" flexDirection="column" left={0} position="absolute" right={0}>
      <FloatBox color={theme.color.border}>
        <Box flexDirection="column" width={Math.max(28, cols - 6)}>
          {completions.slice(start, start + viewportSize).map((item, i) => {
            const active = start + i === compIdx

            return (
              <Box
                backgroundColor={active ? theme.color.completionCurrentBg : theme.color.completionBg}
                flexDirection="row"
                key={`${start + i}:${item.text}:${item.display}:${item.meta ?? ''}`}
                width="100%"
              >
                <Text bold color={theme.color.label}>
                  {' '}
                  {item.display}
                </Text>
                {item.meta ? (
                  <Text
                    backgroundColor={active ? theme.color.completionMetaCurrentBg : theme.color.completionMetaBg}
                    color={theme.color.muted}
                  >
                    {' '}
                    {item.meta}
                  </Text>
                ) : null}
              </Box>
            )
          })}
        </Box>
      </FloatBox>
    </Box>
  )
}
