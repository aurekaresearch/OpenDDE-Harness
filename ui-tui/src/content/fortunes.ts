const FORTUNES = [
  'the candidate you almost filtered out is the binder',
  'one fixed framework residue saves a week of rescue work',
  'graft the CDR you trust, not the one you like',
  'the liability you are ignoring is already in your target notes',
  'shorter CDR3, calmer developability',
  'today favors tighter filters over bigger libraries',
  'the right epitope is already in the structure you have',
  'a long fold is not a failed fold',
  'the refold will work once you stop rushing the buffer',
  'rank on the assay you trust, not the score you like',
  'the developability filter is about to save your future self',
  'your instincts are correctly suspicious of that one loop'
]

const LEGENDARY = [
  'legendary drop: nanomolar binder on the first run',
  'legendary drop: every candidate clears developability',
  'legendary drop: the model folds it exactly as designed'
]

const hash = (s: string) => [...s].reduce((h, c) => Math.imul(h ^ c.charCodeAt(0), 16777619), 2166136261) >>> 0

const fromScore = (n: number) => {
  const rare = n % 20 === 0
  const bag = rare ? LEGENDARY : FORTUNES

  return `${rare ? '🌟' : '🔮'} ${bag[n % bag.length]}`
}

export const randomFortune = () => fromScore(Math.floor(Math.random() * 0x7fffffff))
export const dailyFortune = (seed: null | string) => fromScore(hash(`${seed || 'anon'}|${new Date().toDateString()}`))
