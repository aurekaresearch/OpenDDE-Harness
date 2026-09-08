// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

/** Platform-aware keybinding helpers.
 *
 * On macOS the "action" modifier is Cmd. Modern terminals that support kitty
 * keyboard protocol report Cmd as `key.super`; legacy terminals often surface it
 * as `key.meta`. Some macOS terminals also translate Cmd+Left/Right/Backspace
 * into readline-style Ctrl+A/Ctrl+E/Ctrl+U before the app sees them.
 * On other platforms the action modifier is Ctrl.
 * Ctrl+C stays the interrupt key on macOS. On non-mac terminals it can also
 * copy an active TUI selection, matching common terminal selection behavior.
 */

export const isMac = process.platform === 'darwin'

/** True when the platform action-modifier is pressed (Cmd on macOS, Ctrl elsewhere). */
export const isActionMod = (key: { ctrl: boolean; meta: boolean; super?: boolean }): boolean =>
  isMac ? key.meta || key.super === true : key.ctrl

/**
 * Accept raw Ctrl+<letter> as an action shortcut on macOS, where `isActionMod`
 * otherwise means Cmd. Two motivations:
 *   - Some macOS terminals rewrite Cmd navigation/deletion into readline control
 *     keys (Cmd+Left → Ctrl+A, Cmd+Right → Ctrl+E, Cmd+Backspace → Ctrl+U).
 *   - Ctrl+K (kill-to-end) and Ctrl+W (delete-word-back) are standard readline
 *     bindings that users expect to work regardless of platform, even though
 *     no terminal rewrites Cmd into them.
 */
export const isMacActionFallback = (
  key: { ctrl: boolean; meta: boolean; super?: boolean },
  ch: string,
  target: 'a' | 'e' | 'u' | 'k' | 'w'
): boolean => isMac && key.ctrl && !key.meta && key.super !== true && ch.toLowerCase() === target

/** Match action-modifier + a single character (case-insensitive). */
export const isAction = (key: { ctrl: boolean; meta: boolean; super?: boolean }, ch: string, target: string): boolean =>
  isActionMod(key) && ch.toLowerCase() === target

export const isRemoteShell = (env: NodeJS.ProcessEnv = process.env): boolean =>
  Boolean(env.SSH_CONNECTION || env.SSH_CLIENT || env.SSH_TTY)

/**
 * Full-screen mouse tracking prevents the outer terminal from owning a drag
 * selection. Copy stable selections automatically on macOS and across SSH,
 * where OSC 52 bridges the remote TUI to the user's local clipboard.
 */
export const shouldCopySelectionOnSelect = (
  platform: NodeJS.Platform = process.platform,
  env: NodeJS.ProcessEnv = process.env
): boolean => platform === 'darwin' || isRemoteShell(env)

export const isCopyShortcut = (
  key: { ctrl: boolean; meta: boolean; super?: boolean },
  ch: string,
  env: NodeJS.ProcessEnv = process.env
): boolean =>
  ch.toLowerCase() === 'c' &&
  (isAction(key, ch, 'c') ||
    (isRemoteShell(env) && (key.meta || key.super === true)) ||
    // VS Code/Cursor/Windsurf terminal setup forwards Cmd+C as a CSI-u
    // sequence with the super bit plus a benign ctrl bit. Accept that shape
    // even though raw Ctrl+C should remain interrupt on local macOS.
    (isMac && key.ctrl && (key.meta || key.super === true)))
