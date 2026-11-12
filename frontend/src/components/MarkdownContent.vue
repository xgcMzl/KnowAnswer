<script setup lang="ts">
import { computed } from 'vue'
import DOMPurify from 'dompurify'
import { marked } from 'marked'

const props = defineProps<{ content: string }>()

marked.setOptions({ gfm: true, breaks: true })

const html = computed(() => {
  const parsed = marked.parse(props.content || '') as string
  return DOMPurify.sanitize(parsed)
})
</script>

<template>
  <div class="markdown-body" v-html="html"></div>
</template>
