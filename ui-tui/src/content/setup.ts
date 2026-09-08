import type { PanelSection } from '../types.js'

export const SETUP_REQUIRED_TITLE = 'Setup Required'

export const buildSetupRequiredSections = (): PanelSection[] => [
  {
    text: 'OpenDDE Harness needs a model provider before it can start a design session.'
  },
  {
    rows: [
      ['/model', 'configure provider + model in-place'],
      ['Ctrl+C', 'exit and run `ddeharness onboard` in a terminal']
    ],
    title: 'Actions'
  }
]
