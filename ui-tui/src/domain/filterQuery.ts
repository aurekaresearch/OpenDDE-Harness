/** Case-insensitive keyword match for the picker lists.

    Every whitespace-separated token has to appear somewhere in the row's
    searchable text, in any order: "deep flash" finds "DeepSeek-V4-Flash",
    and "custom deep" finds it under a relay whose slug is in that text.
    Substring rather than fuzzy on purpose -- a relay lists hundreds of ids
    that differ by a digit, and a fuzzy match puts the wrong one first. */
export const matchesQuery = (text: string, query: string): boolean => {
  const tokens = query.trim().toLowerCase().split(/\s+/).filter(Boolean)

  if (tokens.length === 0) {
    return true
  }

  const hay = text.toLowerCase()

  return tokens.every(token => hay.includes(token))
}

/** The indices of the rows a query keeps, in list order. */
export const matchingIndices = (haystacks: readonly string[], query: string): number[] =>
  haystacks.reduce<number[]>((keep, hay, i) => {
    if (matchesQuery(hay, query)) {
      keep.push(i)
    }

    return keep
  }, [])
