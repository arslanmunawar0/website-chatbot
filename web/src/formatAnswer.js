const escapeHtml = (text) =>
  text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')

function formatInline(line, sourcesByNumber) {
  return escapeHtml(line)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\[(\d+)\]/g, (match, number) => {
      const source = sourcesByNumber.get(Number(number))
      if (!source) return match
      return `<a class="cite" href="${escapeHtml(source.url)}" target="_blank" rel="noopener" title="${escapeHtml(source.title)}">${number}</a>`
    })
}

/**
 * Render an LLM answer (light Markdown: paragraphs, bullet lists, **bold**) as safe HTML,
 * turning citation markers like [1] into links to the matching source page.
 */
export function formatAnswer(text, sources = []) {
  const sourcesByNumber = new Map(sources.map((s) => [s.number, s]))
  const html = []
  let listItems = []

  const flushList = () => {
    if (listItems.length) html.push(`<ul>${listItems.join('')}</ul>`)
    listItems = []
  }

  for (const rawLine of text.split('\n')) {
    const line = rawLine.trim()
    const bullet = line.match(/^(?:[-*•]|\d+\.)\s+(.*)$/)
    if (bullet) {
      listItems.push(`<li>${formatInline(bullet[1], sourcesByNumber)}</li>`)
      continue
    }
    flushList()
    if (line) html.push(`<p>${formatInline(line, sourcesByNumber)}</p>`)
  }
  flushList()
  return html.join('')
}
