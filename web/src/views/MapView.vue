<template>
  <div class="card">
    <h3>24 站点地图 · {{ filter.pollutant.toUpperCase() }} · {{ filter.horizon }}h 预测分级</h3>
    <div v-if="loading" class="hint">正在计算 24 站分级…</div>
    <div ref="chartEl" class="chart"></div>
    <div v-if="error" class="hint err">{{ error }}</div>
  </div>
</template>

<script setup>
import { onMounted, onBeforeUnmount, ref, watch } from 'vue'
import * as echarts from 'echarts'
import { useFilterStore } from '../stores/filter'
import { getGrading } from '../api/client'

const filter = useFilterStore()
const chartEl = ref(null)
const loading = ref(false)
const error = ref('')
let chart = null

async function load() {
  if (!filter.stations.length) return
  loading.value = true
  error.value = ''
  // 逐站拉取预测分级（后端 Dataset/模型均已常驻缓存）
  const results = (await Promise.all(
    filter.stations.map(async s => {
      try {
        const r = await getGrading({
          station_id: s.code, pollutant: filter.pollutant, steps: String(filter.horizon),
        })
        const item = r.series.find(x => x.h === filter.horizon)
        if (!item) return null
        return { code: s.code, name: s.name, lon: s.lon, lat: s.lat, value: item.value, grading: item.grading }
      } catch {
        return null
      }
    })
  )).filter(Boolean)
  loading.value = false

  if (!results.length) {
    error.value = '无可用分级结果（模型未就绪？）'
    return
  }

  chart.setOption({
    title: {
      text: `${filter.pollutant.toUpperCase()} ${filter.horizon}h 预测分布（${results.length}/24 站）`,
      left: 'center',
    },
    tooltip: {
      formatter: p => `${p.data.name}（${p.data.code}）<br/>浓度 ${p.data.value[2]} μg/m³<br/>等级 ${p.data.grading.level_name}`,
    },
    grid: { left: 50, right: 30, top: 50, bottom: 50 },
    xAxis: { name: '经度', type: 'value', min: 115.8, max: 117.2 },
    yAxis: { name: '纬度', type: 'value', min: 39.6, max: 40.6 },
    series: [{
      type: 'scatter',
      symbolSize: 24,
      data: results.map(r => ({
        code: r.code, name: r.name,
        value: [r.lon, r.lat, r.value],
        grading: r.grading,
        itemStyle: { color: r.grading.color },
      })),
      label: { show: true, formatter: p => p.data.name, position: 'top', fontSize: 10 },
    }],
  }, true)
}

onMounted(() => {
  chart = echarts.init(chartEl.value)
  load()
  window.addEventListener('resize', () => chart && chart.resize())
})
onBeforeUnmount(() => { chart && chart.dispose() })
watch(() => [filter.pollutant, filter.horizon, filter.stations.length], load)
</script>
