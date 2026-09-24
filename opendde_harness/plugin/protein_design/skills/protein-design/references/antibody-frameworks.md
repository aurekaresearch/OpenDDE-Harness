# Reference antibody frameworks

Read this reference only when a user wants an antibody-design configuration but has not supplied a starting binder scaffold. These are convenient variable-region starting points, not target-specific binders and not evidence that a framework is suitable for a particular antigen.

## Selection rules

- First confirm whether the design is a single-chain VHH or paired VH/VL variable regions. The paired entries below are called “Fab” for convenience, but contain only VH and VL sequences; constant domains are not included.
- Offer the two matching choices and let the user select one in the Binder / antibody confirmation round. Do not silently choose a default.
- Use the CDR-masked sequence in a de novo CDR bootstrap. The unmasked source sequence is included for provenance and inspection only.
- Keep every listed `fixed_residues` position immutable. The ranges are one-based and inclusive, matching the protein-design YAML schema.
- Do not transfer a source antibody's biological specificity or target annotation to a new task.

## Paired VH/VL choices

### Fab-1: adalimumab-derived variable framework

Source VH: `EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSAITWNSGHIDYADSVEGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS`

Source VL: `DIQMTQSPSSLSASVGDRVTITCRASQGIRNYLAWYQQKPGKAPKLLIYAASTLQSGVPSRFSGSGSGTDFTLTISSLQPEDVATYYCQRYNRAPYTFGQGTKVEIK`

```yaml
initial_binders:
  - name: fab_adalimumab_framework
    chains:
      H:
        sequence: EVQLVESGGGLVQPGRSLRLSCAASXXXXXXXXXXWVRQAPGKGLEWVSXXXXXXXXXXYADSVEGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKXXXXXXXXXXXXWGQGTLVTVSS
        cdr_regions: "26:35,50:59,99:110"
        fixed_residues: "1:25,36:49,60:98,111:121"
        chain_type: VH
      L:
        sequence: DIQMTQSPSSLSASVGDRVTITCXXXXXXXXXXXWYQQKPGKAPKLLIYXXXXXXXGVPSRFSGSGSGTDFTLTISSLQPEDVATYYCXXXXXXXXXFGQGTKVEIK
        cdr_regions: "24:34,50:56,89:97"
        fixed_residues: "1:23,35:49,57:88,98:107"
        chain_type: VL
```

### Fab-2: belimumab-derived variable framework

Source VH: `QVQLQQSGAEVKKPGSSVRVSCKASGGTFNNNAINWVRQAPGQGLEWMGGIIPMFGTAKYSQNFQGRVAITADESTGTASMELSSLRSEDTAVYYCARSRDLLLFPHHALSPWGRGTMVTVSS`

Source VL: `SSELTQDPAVSVALGQTVRVTCQGDSLRSYYASWYQQKPGQAPVLVIYGKNNRPSGIPDRFSGSSSGNTASLTITGAQAEDEADYYCSSRDSSGNHWVFGGGTELTVL`

```yaml
initial_binders:
  - name: fab_belimumab_framework
    chains:
      H:
        sequence: QVQLQQSGAEVKKPGSSVRVSCKASXXXXXXXXXXWVRQAPGQGLEWMGXXXXXXXXXXYSQNFQGRVAITADESTGTASMELSSLRSEDTAVYYCARXXXXXXXXXXXXXXWGRGTMVTVSS
        cdr_regions: "26:35,50:59,99:112"
        fixed_residues: "1:25,36:49,60:98,113:123"
        chain_type: VH
      L:
        sequence: SSELTQDPAVSVALGQTVRVTCXXXXXXXXXXXWYQQKPGQAPVLVIYXXXXXXXGIPDRFSGSSSGNTASLTITGAQAEDEADYYCXXXXXXXXXXXFGGGTELTVL
        cdr_regions: "23:33,49:55,88:98"
        fixed_residues: "1:22,34:48,56:87,99:108"
        chain_type: VL
```

## VHH choices

### VHH-1: TPP-3444-derived framework

Source VHH: `EVQLVESGGGLVQPGGSLRLSCAASGRAHSDYAMAWFRQAPGQEREFVAGIGWSGGDTLYADSVRGRFTNSRDNSKNTLYLQMNSLRAEDTAVYYCAARQGQYIYSSMRSDSYDYWGQGTLVTVSS`

```yaml
initial_binders:
  - name: vhh_tpp3444_framework
    chains:
      D:
        sequence: EVQLVESGGGLVQPGGSLRLSCAASXXXXXXXXXXWFRQAPGQEREFVAXXXXXXXXXXYADSVRGRFTNSRDNSKNTLYLQMNSLRAEDTAVYYCAAXXXXXXXXXXXXXXXXXWGQGTLVTVSS
        cdr_regions: "26:35,50:59,99:115"
        fixed_residues: "1:25,36:49,60:98,116:126"
        chain_type: VHH
```

### VHH-2: caplacizumab-derived framework

Source VHH: `EVQLVESGGGLVQPGGSLRLSCAASGRTFSYNPMGWFRQAPGKGRELVAAISRTGGSTYYPDSVEGRFTISRDNAKRMVYLQMNSLRAEDTAVYYCAAAGVRAEDGRVRTLPSEYTFWGQGTQVTVSS`

```yaml
initial_binders:
  - name: vhh_caplacizumab_framework
    chains:
      D:
        sequence: EVQLVESGGGLVQPGGSLRLSCAASXXXXXXXXXXWFRQAPGKGRELVAXXXXXXXXXXYPDSVEGRFTISRDNAKRMVYLQMNSLRAEDTAVYYCAAXXXXXXXXXXXXXXXXXXXWGQGTQVTVSS
        cdr_regions: "26:35,50:59,99:117"
        fixed_residues: "1:25,36:49,60:98,118:128"
        chain_type: VHH
```

## Provenance

The sequences and region annotations in this reference were selected from the user-provided framework tables. Their original one-based inclusive coordinates are retained as YAML ranges. Preserve the source names so downstream records retain scaffold provenance.
