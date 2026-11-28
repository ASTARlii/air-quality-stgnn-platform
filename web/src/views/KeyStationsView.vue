<template>
  <div class="card">
    <h3>关键站点 Top-K · {{ filter.pollutant.toUpperCase() }}</h3>
    <div class="toolbar">
      <label>Top <input v-model.number="top" type="number" min="1" max="24" style="width:64px" /></label>
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
import { ensureNetwork, getKeyStations } from '../api/client'

const filter = useFilterStore()
const top = ref(10)
const chartEl = ref(null)
const status = ref('')
const error = ref('')
let chart = null

const STATUS_TEXT = {
  pending: '网络构建排队中…',
  running: '正在构建关联网络（约 1-4 分钟）…',
  success: '就绪',
  failed: '构建失败',
}
const statusText = computed(() => STATUS_TEXT[status.value] || status.value)

async function load() {
  error.value = ''
  status.value = 'pending'
  try {
    // 中心性依赖关联网络，先确保网络已构建
    await ensureNetwork(
      { pollutant: filter.pollutant, ...NETWORK_WINDOW },
      s => { status.value = s },
    )
    const data = await getKeyStations({
      top: top.value, pollutant: filter.pollutant, ...NETWORK_WINDOW,
    })
    status.value = 'success'
    chart.setOption({
      title: { text: `关键站点 Top ${top.value}（综合中心性）`, left: 'center' },
      tooltip: {
        formatter: p => {
          const it = data.items[p.dataIndex]
          return `${it.name}（${it.station_id}）<br/>综合 ${it.score}<br/>度 ${it.degree} · PageRank ${it.pagerank}`
        },
      },
      grid: { left: 110, right: 60, top: 50, bottom: 40 },
      xAxis: { type: 'value', name: '中心性得分' },
      yAxis: { type: 'category', data: data.items.map(i => i.name).reverse() },
      series: [{
        type: 'bar',
        data: data.items.map(i => i.score).reverse(),
        label: { show: true, position: 'right', formatter: p => p.value.toFixed(3) },
        itemStyle: { color: '#5470c6' },
      }],
    }, true)
  } catch (e) {
    error.value = `加载失败：${e.message}`
    status.value = 'failed'
  }
}

onMounted(() => {
  chart = echarts.init(chartEl.value)
  load()
  window.addEventListener('resize', () => chart && chart.resize())
})
onBeforeUnmount(() => { chart && chart.dispose() })
watch([top, () => filter.pollutant], load)
</script>
