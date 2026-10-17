const BASE_URL = import.meta.env.VITE_API_URL ?? ''

async function request(path, options = {}) {
  const response = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    const detail = typeof body.detail === 'string' ? body.detail : response.statusText
    throw new Error(detail || `Request failed (${response.status})`)
  }
  return response.json()
}

export async function fetchSites() {
  const { sites } = await request('/api/sites')
  return sites
}

export function askQuestion({ site, question, history }) {
  return request('/api/chat', {
    method: 'POST',
    body: JSON.stringify({ site, question, history }),
  })
}
