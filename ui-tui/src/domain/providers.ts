export const providerDisplayNames = (providers: readonly { name: string; slug: string }[]): string[] => {
  const counts = new Map<string, number>()

  for (const p of providers) {
    counts.set(p.name, (counts.get(p.name) ?? 0) + 1)
  }

  return providers.map(p =>
    (counts.get(p.name) ?? 0) > 1 && p.slug && p.slug !== p.name ? `${p.name} (${p.slug})` : p.name
  )
}

/** The vendor's own id, for a provider whose endpoint the user supplied.
 *
 * Only that provider: elsewhere the prefix names which vendor serves the model,
 * and a row that hides it stops matching the id that gets stored. Storage keeps
 * the prefix either way -- this is what the row reads, not what is written.
 */
export const bareModelId = (provider: { auth_type?: string; slug: string } | undefined, model: string): string => {
  if (provider?.auth_type !== 'endpoint') {
    return model
  }

  for (const prefix of [`${provider.slug}/`, `${provider.slug.replace(/_/gu, '-')}/`]) {
    if (model.startsWith(prefix)) {
      return model.slice(prefix.length)
    }
  }

  return model
}
