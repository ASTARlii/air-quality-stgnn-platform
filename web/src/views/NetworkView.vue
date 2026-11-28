<template>
  <div class="card">
    <h3>关联网络 · {{ filter.pollutant.toUpperCase() }}</h3>
    <div class="toolbar">
      <button :disabled="building" @click="recompute">重算网络</button>
      <span v-if="status" class="hint">{{ statusText }}</span>
    </div>
    <div ref="chartEl" class="chart"></div>
    <div v-if="error" class="hint err">{{ error }}</div>
  </div>
</template>

<script setup>
import { onMounted, onBeforeUnmount, ref, watch, computed } from 'vue'
import * as echarts from 'echarts'
import { useFilterStore, NETWORK_WINDOW } from '../stores/filter'
import { ensureNetwork, recomputeNetwork, getTask } from '../api/client'

const filter = useFilterStore()
const chartEl = ref(null)
const status = ref('')
const building = ref(false)
const error = ref('')
let chart = null

const STATUS_TEXT = {
  pending: '网络构建排队中…',
  running: '正在计算转移熵与滞后相关（约 1-4 分钟）…',
  success: '构建完成',
  failed: '构建失败',
}
const statusText = computed(() => STATUS_TEXT[status.value] || status.value)

function render(data) {
  const nPass = data.metrics?.n_fdr_pass ?? 0
  chart.setOption({
    title: {
      text: `关联网络 · ${data.pollutant.toUpperCase()} · 边 ${data.edges.length}（FDR 通过 ${nPass}）`,
      subtext: `${data.from} ~ ${data.to} · T=${data.metrics?.T}h`,
      left: 'center',
    },
    tooltip: {
      formatter: p => p.dataType === 'edge'
        ? `${p.data.source} → ${p.data.target}<br/>权重 ${p.data.value.toFixed(3)} · 滞后 ${p.data.lag}h`
        : p.data.name,
    },
    series: [{
      type: 'graph',
      layout: 'force',
      roam: true,
      force: { repulsion: 220, edgeLength: [50, 160] },
      label: { show: true, fontSize: 9 },
      data: data.nodes.map(n => ({
        id: n.id, name: `${n.name}\n${n.id}`, symbolSize: 14,
      })),
      links: data.edges.map(e => ({
        source: e.src, target: e.dst, value: e.weight, lag: e.lag,
        lineStyle: {
          width: Math.max(0.4, e.weight * 3),
          opacity: e.fdr_pass ? 0.85 : 0.35,
          color: e.fdr_pass ? '#5470c6' : '#bbb',
        },
      })),
      lineStyle: { curveness: 0.1 },
    }],
  }, true)
}

async function load() {
  building.value = true
  error.value = ''
  status.value = 'pending'
  try {
    const data = await ensureNetwork(
      { pollutant: filter.pollutant, ...NETWORK_WINDOW },
      s => { status.value = s },
    )
    status.value = 'success'
    render(data)
  } catch (e) {
    error.value = `网络加载失败：${e.message}`
    status.value = 'failed'
  } finally {
    building.value = false
  }
}

async function recompute() {
  building.value = true
  error.value = ''
  try {
    const r = await recomputeNetwork({
      pollutant: filter.pollutant,
      from_time: NETWORK_WINDOW.from,
      to_time: NETWORK_WINDOW.to,
      surrogates: NETWORK_WINDOW.surrogates,
    })
    status.value = r.status
    for (let i = 0; i < 600; i++) {
      await new Promise(res => setTimeout(res, 1000))
      const t = await getTask(r.task_key)
      status.value = t.status
      if (t.status === 'success') break
      if (t.status === 'failed') throw new Error(t.error || '构建失败')
    }
    const data = await ensureNetwork(
      { pollutant: filter.pollutant, ...NETWORK_WINDOW },
      s => { status.value = s },
    )
    render(data)
  } catch (e) {
    error.value = `重算失败：${e.message}`
    status.value = 'failed'
  } finally {
    building.value = false
  }
}

onMounted(() => {
  chart = echarts.init(chartEl.value)
  load()
  window.addEventListener('resize', () => chart && chart.resize())
})
onBeforeUnmount(() => { chart && chart.dispose() })
watch(() => filter.pollutant, load)
</script>
