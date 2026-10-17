<script setup>
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import { askQuestion, fetchSites } from './api'
import { formatAnswer } from './formatAnswer'

const MAX_HISTORY_MESSAGES = 10

const sites = ref([])
const site = ref('')
const messages = ref([])
const draft = ref('')
const loading = ref(false)
const error = ref('')
const scroller = ref(null)

const canSend = computed(() => site.value && draft.value.trim() && !loading.value)

const suggestions = [
  'What does this website offer?',
  'How can I contact support?',
  'What are the opening hours or delivery times?',
]

onMounted(async () => {
  try {
    sites.value = await fetchSites()
    site.value = sites.value[0] ?? ''
  } catch (e) {
    error.value = `Could not reach the chatbot API: ${e.message}`
  }
})

watch(site, () => {
  messages.value = []
  error.value = ''
})

async function scrollToBottom() {
  await nextTick()
  scroller.value?.scrollTo({ top: scroller.value.scrollHeight, behavior: 'smooth' })
}

async function send(text = draft.value) {
  const question = text.trim()
  if (!question || !site.value || loading.value) return

  const history = messages.value
    .slice(-MAX_HISTORY_MESSAGES)
    .map(({ role, content }) => ({ role, content }))

  messages.value.push({ role: 'user', content: question })
  draft.value = ''
  error.value = ''
  loading.value = true
  scrollToBottom()

  try {
    const { answer, sources } = await askQuestion({ site: site.value, question, history })
    messages.value.push({ role: 'assistant', content: answer, sources })
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
    scrollToBottom()
  }
}

function onKeydown(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    send()
  }
}
</script>

<template>
  <div class="page">
    <main class="chat">
      <header class="chat-header">
        <div class="brand">
          <span class="brand-mark">AI</span>
          <div>
            <h1>Website Assistant</h1>
            <p>Answers from the site's own pages, with sources</p>
          </div>
        </div>
        <label class="site-picker">
          <span>Website</span>
          <select v-model="site" :disabled="!sites.length">
            <option v-for="name in sites" :key="name" :value="name">{{ name }}</option>
          </select>
        </label>
      </header>

      <section ref="scroller" class="messages" aria-live="polite">
        <div v-if="!messages.length" class="empty">
          <h2>Ask anything about <strong>{{ site || 'a website' }}</strong></h2>
          <p>The assistant searches the crawled pages and cites where each answer comes from.</p>
          <div class="suggestions">
            <button
              v-for="s in suggestions"
              :key="s"
              type="button"
              :disabled="!site"
              @click="send(s)"
            >
              {{ s }}
            </button>
          </div>
        </div>

        <article
          v-for="(message, index) in messages"
          :key="index"
          class="message"
          :class="message.role"
        >
          <p v-if="message.role === 'user'" class="bubble">{{ message.content }}</p>
          <template v-else>
            <div class="bubble" v-html="formatAnswer(message.content, message.sources)" />
            <ol v-if="message.sources?.length" class="sources">
              <li v-for="source in message.sources" :key="source.number">
                <a :href="source.url" target="_blank" rel="noopener">
                  <span class="source-number">{{ source.number }}</span>
                  {{ source.title }}
                </a>
              </li>
            </ol>
          </template>
        </article>

        <div v-if="loading" class="message assistant">
          <div class="bubble typing" aria-label="Assistant is typing">
            <span /><span /><span />
          </div>
        </div>

        <p v-if="error" class="error" role="alert">{{ error }}</p>
      </section>

      <form class="composer" @submit.prevent="send()">
        <textarea
          v-model="draft"
          rows="1"
          placeholder="Type your question…"
          :disabled="!site"
          @keydown="onKeydown"
        />
        <button type="submit" :disabled="!canSend">Send</button>
      </form>
    </main>
  </div>
</template>
