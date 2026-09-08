# Tool Usage Notes

Tool signatures arrive through function calling. This file documents constraints that are not visible in the signatures.

## Protein design tools
- `protein_design_context` returns compute readiness, folding defaults, example paths, and the GPU inventory. Call it once per design conversation.
- `protein_design_start` launches a detached task that keeps running after the terminal closes. It requires the user's explicit approval in the same conversation.
- `protein_design_status`, `protein_design_candidates`, `protein_design_adjust`, and `protein_design_stop` operate on a task id. Adjustments apply at the next cycle boundary.
- `protein_design_search_target_msa` is slow (minutes); mention that before calling it.

## exec
- Commands time out (default 60 s); destructive commands (rm -rf, dd, shutdown, ...) are blocked; output is truncated at 10,000 characters.
- Long computations belong in design tasks, not in exec.

## read_file
- Images are returned as pictures and downscaled when large; crop the region you need instead of reading a whole screenshot.
- An image is visible only for the turn that read it; read it again if you need another look.
