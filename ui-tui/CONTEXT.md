# TUI

The terminal front-end (`ui-tui/`, React/Ink). Renders the chat transcript and overlays;
talks to the Runtime only via TUI-RPC. One active session per client process.

## Language

**Overlay**:
A modal layer over the chat view, tracked in `overlayStore` and driven by keyboard. Kinds
split into flow-scoped (Confirm, Approval, Clarify, Pager), which a finished turn clears,
and user-toggled (Agents, Model Picker, Picker, Skills Hub), which survive it; the FPS
counter is a separate component, not an overlay-store kind.

**MessageLine**:
The UI element rendering one transcript row in the chat view.
_Avoid_: "chat stream" for the UI — chat stream is the data feed it renders

**Status Bar**:
The status rule at the top or bottom of the layout, rendered by the `StatusRule` component;
placement is set by `StatusBarMode` (`top` | `bottom` | `off`).
_Avoid_: "StatusRulePane" — the exported component is `StatusRule`, there is no "Pane".

**Agents Overlay**:
The overlay showing the subagent tree (`SubagentNode` hierarchy with subtree
token/cost aggregates); opened with `/agents`, including for past turns by history index.

**Confirm Overlay**:
The countdown overlay a destructive Confirm Round-Trip presents; the answer resolves
the paused turn.

**Theme**:
The named color/glyph token set all components draw from.

**Brand Lockup**:
The welcome screen's hero art plus the OPENDDE HARNESS wordmark, drawn from `banner.ts`.
`resolveLockupScale` picks the largest scale factor that fits the terminal's row/column
budget, so the lockup shrinks with the window instead of wrapping.

**Current Session**:
The session the TUI is bound to — switching session means rebinding the client to a
different Runtime session key.
