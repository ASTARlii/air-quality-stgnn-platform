<template>
  <div class="card">
    <h3>{{ stationName }}（{{ filter.stationId }}） · {{ filter.pollutant.toUpperCase() }} 预测</h3>
    <div v-if="loading" class="hint">正在推理…</div>
    <div ref="chartEl" class="chart"></div>
    <div v-if="error" class="hint err">{{ error }}</div>
    <div v-if="modelId" class="hint">模型 {{ modelId }} · 推理延迟 {{ latency }}ms · 分级依据 GB 3095-2012（辅助研判，非官方预警）</div>
  </div>
</template>

<script setup>
import { onMounted, onBeforeUnmount, ref, watch, computed } from 'vue'
import * as echarts from 'echarts'
import { useFilterStore } from '../stores/filter'
import { postPrediction } from '../api/client'

const filter = useFilterStore()
const chartEl = ref(null)
const loading = ref(false)
const error = ref('')
const modelId = ref('')
const latency = ref(0)
let chart = null

const stationName = computed(() =>
  filter.stations.find(s => s.code === filter.stationId)?.name || filter.stationId
)

async function load() {
  loading.value = true
  error.value = ''
  try {
    const data = await postPrediction({
      station_id: filter.stationId,
      pollutant: filter.pollutant,
      steps: [1, 3, 6, 12, 24],
    })
    modelId.value = data.model_id
    latency.value = data.latency_ms

    const hours = data.series.map(s => `+${s.h}h`)
    const values = data.series.map(s => s.value)
    const colors = data.series.map(s => s.grading?.color || '#5470c6')

    chart.setOption({
      title: { text: '未来 24h 预测（按 GB 3095-2012 分级着色）', left: 'center' },
      tooltip: {
        trigger: 'axis',
        formatter: ps => {
          const i = ps[0].dataIndex
          const it = data.series[i]
          return `+${it.h}h<br/>${it.value} μg/m³<br/>${it.grading?.level_name ?? ''}`
        },
      },
      grid: { left: 60, right: 30, top: 50, bottom: 40 },
      xAxis: { type: 'category', data: hours, name: '预测步长' },
      yAxis: { type: 'value', name: 'μg/m³' },
      series: [{
        type: 'bar',
        data: values.map((v, i) => ({ value: v, itemStyle: { color: colors[i] } })),
        label: { show: true, position: 'top' },
      }],
    }, true)
  } catch (e) {
    const msg = e.response?.data?.detail || e.message
    error.value = `预测失败：${msg}`
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  chart = echarts.init(chartEl.value)
  load()
  window.addEventListener('resize', () => chart && chart.resize())
})
onBeforeUnmount(() => { chart && chart.dispose() })
watch(() => [filter.stationId, filter.pollutant], load)
</script>
