# Agent Instructions

Workspace conventions for this antibody design project. Edit freely; the product identity and scope come from the runtime and do not need to be repeated here.

## Working in this workspace
- Keep design YAML files under `designs/`, one directory per target, and copy a bundled example before adapting it.
- Record the reasoning behind every non-default choice (scaffold, fixed residues, loss weights, placement) in the YAML comments so a later run can be reproduced.
- Prefer the smallest change that answers the question: adjust a running task before starting a new one when the target and scaffold are unchanged.
- Name tasks after the target and the scaffold, for example `crlf2-vhh-01`.

## Project notes
(Add target-specific constraints, preferred frameworks, and internal naming rules here.)
