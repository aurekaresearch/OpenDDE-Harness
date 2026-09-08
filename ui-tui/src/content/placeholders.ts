import { pick } from '../lib/text.js'

export const PLACEHOLDERS = [
  'Design a VHH against human CRLF2 and keep the framework fixed',
  'Graft these CDRs onto a humanized VH/VL scaffold',
  'Design a nanobody that blocks the receptor binding site',
  'Humanize this llama VHH but keep the binding loops',
  'Design three binders against the same epitope so I can compare them',
  'Make a shorter CDR-H3 variant of the best candidate',
  'Which epitope on PD-L1 should the binder target?',
  'Is CDR-H3 too long for a stable VHH here?',
  'Will this mutation break the disulfide?',
  'What makes this framework more stable than the parent?',
  'Does the paratope overlap the receptor binding site?',
  'How confident is the model about this loop?',
  'Would a shorter linker help the bispecific fold?',
  'Start the CRLF2 quickstart and show me the plan first',
  'Pause my design task after this cycle',
  'How is my running design task doing?',
  'Run another design cycle with tighter filters',
  'Stop the run but keep the candidates so far',
  'Compare the top candidates from my last run',
  'Why did these candidates fail the developability filter?',
  'Show the contacts between the best candidate and the epitope',
  'Rank the candidates by predicted affinity',
  'Which candidates share the same binding mode?',
  'Summarize what changed between my last two runs',
  'Show the sequence differences from the parent antibody',
  'Is the compute service ready?',
  'Refold the best candidate on four GPUs',
  'How many GPUs are free right now?',
  'Which structure prediction model am I using?',
  'Show me the tools and skills available here'
]

export const PLACEHOLDER = pick(PLACEHOLDERS)
