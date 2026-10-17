const escapeHtml = (text) =>
  text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')

function formatInline(line, sourcesByNumber) {
  const codeSpans = []
  const withPlaceholders = line.replace(/`([^`]+)`/g, (_, code) => {
    codeSpans.push(`<code>${escapeHtml(code)}</code>`)
    return `\u0000${codeSpans.length - 1}\u0000`
  })

  return escapeHtml(withPlaceholders)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\[(\d+)\]/g, (match, number) => {
      const source = sourcesByNumber.get(Number(number))
      if (!source) return match
      return `<a class="cite" href="${escapeHtml(source.url)}" target="_blank" rel="noopener" title="${escapeHtml(source.title)}">${number}</a>`
    })
    .replace(/\u0000(\d+)\u0000/g, (_, index) => codeSpans[Number(index)])
}

/**
 * Render an LLM answer (light Markdown: paragraphs, nested bullet lists, **bold**, `code`
 * and fenced code blocks) as safe HTML, turning citation markers like [1] into source links.
 */
export function formatAnswer(text, sources = []) {
  const sourcesByNumber = new Map(sources.map((s) => [s.number, s]))
  const html = []
  let listItems = []
  let codeLines = null

  const flushList = () => {
    if (listItems.length) html.push(`<ul>${listItems.join('')}</ul>`)
    listItems = []
  }

  for (const rawLine of text.split('\n')) {
    if (rawLine.trim().startsWith('```')) {
      if (codeLines === null) {
        flushList()
        codeLines = []
      } else {
        html.push(`<pre><code>${escapeHtml(codeLines.join('\n'))}</code></pre>`)
        codeLines = null
      }
      continue
    }
    if (codeLines !== null) {
      codeLines.push(rawLine)
      continue
    }

    const line = rawLine.trim()
    const bullet = rawLine.match(/^(\s*)(?:[-*•]|\d+\.)\s+(.*)$/)
    if (bullet) {
      const nested = bullet[1].replace(/\t/g, '  ').length >= 2
      listItems.push(
        `<li${nested ? ' class="nested"' : ''}>${formatInline(bullet[2], sourcesByNumber)}</li>`,
      )
      continue
    }
    flushList()
    if (line) html.push(`<p>${formatInline(line, sourcesByNumber)}</p>`)
  }

  if (codeLines !== null) html.push(`<pre><code>${escapeHtml(codeLines.join('\n'))}</code></pre>`)
  flushList()
  return html.join('')
}

export function siteLabel(name) {
  return name
    .split('_')
    .filter(Boolean)
    .map((word) => word[0].toUpperCase() + word.slice(1))
    .join(' ')
}
