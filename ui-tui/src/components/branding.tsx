// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import { Box, Text } from '@hermes/ink'
import { useEffect, useState } from 'react'
import unicodeSpinners from 'unicode-animations'

import type { PanelSection, SessionInfo } from '../types.js'

import {
  type ArtScale,
  LOCKUP_GAP,
  lockupSize,
  openddeHarnessHero,
  openddeHarnessWordmarkAt,
  resolveLockupScale,
  WELCOME_CHROME_ROWS
} from '../banner.js'
import { useTerminalSize } from '../hooks/useTerminalSize.js'
import { planSessionPanel } from '../lib/panelLayout.js'
import { flat } from '../lib/text.js'
import { DEFAULT_THEME, type Theme } from '../theme.js'

const LOADER_TICK_MS = 120

function InlineLoader({ label, t }: { label: string; t: Theme }) {
  const [tick, setTick] = useState(0)
  const spinner = unicodeSpinners.braille
  const frame = spinner.frames[tick % spinner.frames.length] ?? '⠋'

  useEffect(() => {
    const id = setInterval(() => setTick(n => n + 1), Math.max(LOADER_TICK_MS, spinner.interval))

    return () => clearInterval(id)
  }, [spinner.interval])

  return (
    <Text color={t.color.muted} wrap="truncate">
      <Text color={t.color.accent}>{frame}</Text> {label}
    </Text>
  )
}

const STARTUP_MESSAGES = [
  'loading antibody design tools…',
  'connecting to the compute service…',
  'preparing design skills…'
]
const STARTUP_LABEL_MS = 900

// The line the session panel shows in its section area while the backend
// builds the agent loop, before the session.info handshake fills the sections.
function StartupLine({ t }: { t: Theme }) {
  const [step, setStep] = useState(0)

  useEffect(() => {
    const id = setInterval(() => setStep(n => n + 1), STARTUP_LABEL_MS)

    return () => clearInterval(id)
  }, [])

  const label = STARTUP_MESSAGES[Math.min(step, STARTUP_MESSAGES.length - 1)] ?? STARTUP_MESSAGES[0]

  return <InlineLoader label={label} t={t} />
}

export function ArtLines({ lines }: { lines: [string, string][] }) {
  return (
    <>
      {lines.map(([c, text], i) => (
        // `truncate` so wide banner art clips at the box edge instead of
        // wrapping each row into an unreadable scatter on narrow terminals.
        <Text color={c} key={i} wrap="truncate">
          {text}
        </Text>
      ))}
    </>
  )
}

// Like ArtLines but each row is an array of `[color, segment]` pairs rendered
// inline — used for horizontally-graded art (the opendde hero).
export function ArtRows({ rows }: { rows: [string, string][][] }) {
  return (
    <>
      {rows.map((segs, i) => (
        <Text key={i} wrap="truncate">
          {segs.map(([c, text], j) => (
            <Text color={c} key={j}>
              {text}
            </Text>
          ))}
        </Text>
      ))}
    </>
  )
}

const PROVIDER_LABELS: Record<string, string> = {
  anthropic: 'Anthropic',
  openai: 'OpenAI',
  openrouter: 'OpenRouter',
  qwen: 'Qwen',
  google: 'Google',
  mistral: 'Mistral'
}

// formatProvider — Resolve a user-facing provider label.
//
// Why the slug → model_id fallback: user config may set `provider="auto"`
// (LiteLLM auto-routing dispatch mode), which isn't a real provider name.
// In that case parse the model_id prefix (e.g. "openrouter/qwen/..." →
// "openrouter") to find the real provider.
//
// Why the LUT: capitalize-only would yield "Openai" / "Openrouter" — visually
// wrong. PROVIDER_LABELS keeps canonical casing for known providers; unknown
// providers fall back to plain capitalize.
export function formatProvider(slug?: string, modelId?: string): string {
  let effective = slug ?? ''
  if (!effective || effective === 'auto') {
    // Only treat model_id as carrying provider info when it has a `/` prefix
    // (e.g. "openrouter/qwen/qwen3.6-plus"). A bare "sonnet" is a model name,
    // not a provider — fall through to '—'.
    const id = modelId ?? ''
    effective = id.includes('/') ? (id.split('/')[0] ?? '') : ''
  }
  if (!effective) {
    return '—'
  }
  const key = effective.toLowerCase()
  return PROVIDER_LABELS[key] ?? effective.charAt(0).toUpperCase() + effective.slice(1)
}

// The brand lockup: the hero on the left, the stacked OPENDDE / HARNESS
// wordmark on the right, both downsampled by the same factor so they stay
// proportional, four columns apart and centred against each other.
export function BrandLockup({ factor, t }: { factor: ArtScale; t: Theme }) {
  const yellow = t.yellow
  // Banner draws the .100/.300/.500/.600 stops (skip .50 so the top band isn't
  // near-white); the wordmark maps one ramp entry per vertical band.
  const palette = [yellow[1]!, yellow[2]!, yellow[3]!, yellow[4]!]
  const heroRamp = [t.color.primary, t.color.primary, t.color.primary, t.color.primary]
  const size = lockupSize(factor)

  return (
    <Box alignItems="center" flexDirection="row">
      <Box flexDirection="column" flexShrink={0} width={size.heroCols}>
        <ArtRows rows={openddeHarnessHero(heroRamp, t.bannerHero || undefined, factor)} />
      </Box>

      <Box flexDirection="column" flexShrink={0} marginLeft={LOCKUP_GAP}>
        <ArtLines lines={openddeHarnessWordmarkAt(palette, factor)} />
      </Box>
    </Box>
  )
}

function BrandTextLine({ label, t }: { label: string; t: Theme }) {
  return (
    <Text bold wrap="truncate">
      <Text color={t.color.accent}>{t.brand.icon}</Text>
      <Text color={t.color.primary}>{`  ${label}`}</Text>
    </Text>
  )
}

export function Branding({ t }: { t?: Theme } = {}) {
  const theme = t ?? DEFAULT_THEME
  const { columns, rows } = useTerminalSize()
  const factor = resolveLockupScale(rows - 2, columns - 2)

  return (
    <Box flexDirection="column" marginBottom={1}>
      {factor === 0 ? (
        <BrandTextLine label={theme.brand.name} t={theme} />
      ) : (
        <>
          <BrandLockup factor={factor} t={theme} />
          {factor === 4 && (
            <Text bold color={theme.color.primary} wrap="truncate">
              {theme.brand.name}
            </Text>
          )}
        </>
      )}
    </Box>
  )
}

// The demo gallery imports `Banner`; keep the name working.
export const Banner = Branding

// ── Collapsible helpers ──────────────────────────────────────────────

function CollapseToggle({
  count,
  open,
  suffix,
  t,
  title,
  onToggle
}: {
  count?: number
  open: boolean
  suffix?: string
  t: Theme
  title: string
  onToggle: () => void
}) {
  // One Text so a narrow column truncates the header instead of wrapping its
  // pieces onto separate rows.
  return (
    <Box onClick={onToggle}>
      <Text wrap="truncate">
        <Text color={t.color.accent}>{open ? '▾ ' : '▸ '}</Text>
        <Text bold color={t.color.accent}>
          {title}
        </Text>
        {typeof count === 'number' ? <Text color={t.color.muted}> ({count})</Text> : null}
        {suffix ? <Text color={t.color.muted}> {suffix}</Text> : null}
      </Text>
    </Box>
  )
}

// ── SessionPanel ─────────────────────────────────────────────────────

const SKILLS_MAX = 8
const TOOLSETS_MAX = 8

// The plan is built from the terminal size and this nominal section model,
// never from `info`, so the lockup and the panel's height are settled on the
// first frame and do not move when the sections fill in. The backend groups
// every tool into one bucket and sends no MCP servers, so one tools row is
// reserved and the MCP section is budgeted as absent; content past the
// reservation grows the section area instead of moving anything above it.
const TOOLS_BODY_ROWS = 1
// Widest footer meta worth an inline slot: version, model, provider, session.
const FOOTER_META_RESERVE_COLS = 80

export function SessionPanel({ info, maxCols, sid, t }: SessionPanelProps) {
  // Width the panel actually has. The full terminal width overshoots whenever
  // the panel is embedded in a narrower container (e.g. the demo gallery's
  // sidebar), so callers can pass `maxCols`; otherwise assume the terminal.
  const size = useTerminalSize()
  const cols = maxCols ?? size.columns
  const available = size.rows - WELCOME_CHROME_ROWS
  const strip = (s: string) => (s.endsWith('_tools') ? s.slice(0, -6) : s)

  const version = info
    ? `${info.version ? `v${info.version}` : ''}${info.release_date ? ` (${info.release_date})` : ''}`.trim()
    : ''
  const nameLabel = version ? `${t.brand.name} ${version}` : t.brand.name

  // ── Local collapse state for each section ──
  // `null` is the default state; an explicit toggle wins over the plan, so a
  // resize never undoes what the user chose.
  const [toolsToggled, setToolsToggled] = useState<boolean | null>(null)
  const [skillsToggled, setSkillsToggled] = useState<boolean | null>(null)
  const [systemToggled, setSystemToggled] = useState<boolean | null>(null)
  const [mcpToggled, setMcpToggled] = useState<boolean | null>(null)

  const skillEntries = Object.entries(info?.skills ?? {}).sort()
  const skillsTotal = info ? flat(info.skills).length : 0
  const skillsCatCount = skillEntries.length
  const toolEntries = Object.entries(info?.tools ?? {}).sort()
  const toolsTotal = info ? flat(info.tools).length : 0
  const mcpServers = info?.mcp_servers ?? []
  const sysPromptLen = (info?.system_prompt ?? '').length

  // Footer meta (version · model · provider · endpoint · session). Kept beside
  // `/help` only when the reserved width fits the column; otherwise the footer
  // becomes a column so the whole meta line drops below `/help` instead of
  // wrapping mid-string. The endpoint segment appears only for a provider that
  // has several, since a single-endpoint one has no label worth a slot. The
  // version rides here whenever the lockup leaves the product name no line of
  // its own.
  const meta = info
    ? `${info.model.split('/').pop()} · ${formatProvider(info.provider, info.model_id)}${info.endpoint ? ` · ${info.endpoint}` : ''}${sid ? ` · ${sid}` : ''}`
    : ''

  // Bodies count toward the budget only in their default state; a body the
  // user opened by hand is theirs to scroll, not the plan's to trade away.
  const plan = planSessionPanel({
    available,
    cols,
    footerMetaLength: FOOTER_META_RESERVE_COLS,
    sections: {
      mcp: { body: 0, open: mcpToggled, present: false },
      skills: { body: 0, open: skillsToggled, present: true },
      system: { body: 0, open: systemToggled, present: true },
      tools: { body: toolsToggled === null ? TOOLS_BODY_ROWS : 0, open: toolsToggled, present: true }
    }
  })
  const w = plan.width
  const gap = plan.gap
  const lineBudget = Math.max(12, w - 2)
  const footerMeta = plan.versionInFooter && version ? `${version} · ${meta}` : meta

  const truncLine = (pfx: string, items: string[]) => {
    let line = ''
    let shown = 0

    for (const item of [...items].sort()) {
      const next = line ? `${line}, ${item}` : item

      if (pfx.length + next.length > lineBudget) {
        return line ? `${line}, …+${items.length - shown}` : `${item}, …`
      }

      line = next
      shown++
    }

    return line
  }

  // ── Collapsible skills section ──
  const skillsBody = () => {
    if (info?.lazy && skillEntries.length === 0) {
      return <InlineLoader label="scanning skills" t={t} />
    }

    const shown = skillEntries.slice(0, SKILLS_MAX)
    const overflow = skillEntries.length - SKILLS_MAX

    return (
      <>
        {shown.map(([k, vs]) => (
          <Text key={k} wrap="truncate">
            <Text color={t.color.muted}>{strip(k)}: </Text>
            <Text color={t.color.text}>{truncLine(strip(k) + ': ', vs)}</Text>
          </Text>
        ))}
        {overflow > 0 && <Text color={t.color.muted}>(and {overflow} more categories…)</Text>}
      </>
    )
  }

  // ── Collapsible tools section ──
  const toolsBody = () => {
    const shown = toolEntries.slice(0, TOOLSETS_MAX)
    const overflow = toolEntries.length - TOOLSETS_MAX

    return (
      <>
        {shown.map(([k, vs]) => (
          <Text key={k} wrap="truncate">
            <Text color={t.color.muted}>{strip(k)}: </Text>
            <Text color={t.color.text}>{truncLine(strip(k) + ': ', vs)}</Text>
          </Text>
        ))}
        {overflow > 0 && <Text color={t.color.muted}>(and {overflow} more toolsets…)</Text>}
      </>
    )
  }

  // ── Collapsible MCP section ──
  const mcpBody = () => (
    <>
      {mcpServers.map(s => (
        <Text key={s.name} wrap="truncate">
          <Text color={t.color.muted}>{`  ${s.name} `}</Text>
          <Text color={t.color.muted}>{`[${s.transport}]`}</Text>
          <Text color={t.color.muted}>: </Text>
          {s.connected ? (
            <Text color={t.color.text}>
              {s.tools} tool{s.tools === 1 ? '' : 's'}
            </Text>
          ) : (
            <Text color={t.color.error}>failed</Text>
          )}
        </Text>
      ))}
    </>
  )

  // ── System prompt body ──
  const systemBody = () => {
    if (sysPromptLen === 0) {
      return <Text color={t.color.muted}>No system prompt loaded.</Text>
    }

    return <Text color={t.color.muted}>{info?.system_prompt}</Text>
  }

  return (
    <Box
      borderColor={t.color.border}
      borderStyle="round"
      flexDirection="column"
      marginBottom={1}
      paddingX={2}
      paddingY={plan.paddingY}
      width={cols}
    >
      {/* The brand lockup is the top of the panel, full width; every section
          sits below it, never beside. */}
      {plan.lockup === 0 ? (
        <BrandTextLine label={nameLabel} t={t} />
      ) : (
        <Box flexDirection="column">
          <BrandLockup factor={plan.lockup} t={t} />
          {plan.nameLine && (
            <Text bold color={t.color.primary} wrap="truncate">
              {nameLabel}
            </Text>
          )}
        </Box>
      )}

      {/* The section area keeps the rows the plan reserved whether it holds
          the startup line or the sections, so the divider and footer below
          it do not move when the info arrives. */}
      <Box flexDirection="column" minHeight={plan.sectionRows}>
        {info === null ? (
          <Box marginTop={gap}>
            <StartupLine t={t} />
          </Box>
        ) : (
          <>
            {plan.show.tools && (
              <Box flexDirection="column" marginTop={gap}>
                <CollapseToggle
                  count={toolsTotal}
                  onToggle={() => setToolsToggled(!plan.open.tools)}
                  open={plan.open.tools}
                  t={t}
                  title="Available Tools"
                />
                {plan.open.tools && toolsBody()}
              </Box>
            )}

            {plan.show.skills && (
              <Box flexDirection="column" marginTop={gap}>
                <CollapseToggle
                  count={skillsTotal}
                  onToggle={() => setSkillsToggled(!plan.open.skills)}
                  open={plan.open.skills}
                  suffix={
                    skillsCatCount > 0 ? `in ${skillsCatCount} categor${skillsCatCount === 1 ? 'y' : 'ies'}` : undefined
                  }
                  t={t}
                  title="Available Skills"
                />
                {plan.open.skills && skillsBody()}
              </Box>
            )}

            {plan.show.system && sysPromptLen > 0 && (
              <Box flexDirection="column" marginTop={gap}>
                <CollapseToggle
                  onToggle={() => setSystemToggled(!plan.open.system)}
                  open={plan.open.system}
                  suffix={`— ${sysPromptLen.toLocaleString()} chars`}
                  t={t}
                  title="System Prompt"
                />
                {plan.open.system && systemBody()}
              </Box>
            )}

            {plan.show.mcp && mcpServers.length > 0 && (
              <Box flexDirection="column" marginTop={gap}>
                <CollapseToggle
                  count={mcpServers.length}
                  onToggle={() => setMcpToggled(!plan.open.mcp)}
                  open={plan.open.mcp}
                  suffix="connected"
                  t={t}
                  title="MCP Servers"
                />
                {plan.open.mcp && mcpBody()}
              </Box>
            )}
          </>
        )}
      </Box>

      {plan.divider && (
        <Box marginTop={gap}>
          <Text color={t.color.border} wrap="truncate">
            {'─'.repeat(Math.max(1, w))}
          </Text>
        </Box>
      )}

      {/* Footer: /help on the left, version · model · session on the right.
          Explicit height so the empty meta of the loading frame still holds
          its row. */}
      <Box
        flexDirection={plan.footerInline ? 'row' : 'column'}
        height={plan.footerInline ? 1 : 2}
        justifyContent={plan.footerInline ? 'space-between' : 'flex-start'}
        width={w}
      >
        <Text color={t.color.muted}>
          <Text color={t.color.accent}>/help</Text> for commands
        </Text>

        <Text color={t.color.muted} wrap="truncate">
          {footerMeta}
        </Text>
      </Box>

      {plan.cwd && (
        <Box marginTop={gap}>
          <Text color={t.color.muted} wrap="truncate-end">
            {info?.cwd || process.cwd()}
          </Text>
        </Box>
      )}
    </Box>
  )
}

export function Panel({ sections, t, title }: PanelProps) {
  return (
    <Box borderColor={t.color.border} borderStyle="round" flexDirection="column" paddingX={2} paddingY={1}>
      <Box justifyContent="center" marginBottom={1}>
        <Text bold color={t.color.primary}>
          {title}
        </Text>
      </Box>

      {sections.map((sec, si) => (
        <Box flexDirection="column" key={si} marginTop={si > 0 ? 1 : 0}>
          {sec.title && (
            <Text bold color={t.color.accent}>
              {sec.title}
            </Text>
          )}

          {sec.rows?.map(([k, v], ri) => (
            <Text key={ri} wrap="truncate">
              <Text color={t.color.muted}>{k.padEnd(20)}</Text>
              <Text color={t.color.text}>{v}</Text>
            </Text>
          ))}

          {sec.items?.map((item, ii) => (
            <Text color={t.color.text} key={ii} wrap="truncate">
              {item}
            </Text>
          ))}

          {sec.text && <Text color={t.color.muted}>{sec.text}</Text>}
        </Box>
      ))}
    </Box>
  )
}

interface PanelProps {
  sections: PanelSection[]
  t: Theme
  title: string
}

interface SessionPanelProps {
  // `null` before the session.info handshake: the same frame and lockup, with
  // the startup line where the sections will go.
  info: SessionInfo | null
  // Container width to lay out against; defaults to the full terminal. Pass it
  // when embedding the panel in a narrower region (e.g. the demo gallery).
  maxCols?: number
  sid?: string | null
  t: Theme
}
