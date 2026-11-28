<template>
  <div class="layout">
    <header class="header">
      <h1>空气质量时空图预测与可视分析平台</h1>
      <select v-model="filter.pollutant">
        <option value="pm25">PM2.5</option>
        <option value="pm10">PM10</option>
        <option value="no2">NO₂</option>
        <option value="o3">O₃</option>
      </select>
      <select v-model="filter.stationId">
        <option v-for="s in filter.stations" :key="s.code" :value="s.code">
          {{ s.name }}（{{ s.code }}）
        </option>
      </select>
      <select v-model="filter.horizon">
        <option v-for="h in [1,3,6,12,24]" :key="h" :value="h">{{ h }}h</option>
      </select>
    </header>

    <nav style="background:#fff;padding:8px 24px;border-bottom:1px solid #eee;">
      <router-link to="/map" style="margin-right:16px">地图</router-link>
      <router-link to="/prediction" style="margin-right:16px">预测</router-link>
      <router-link to="/network" style="margin-right:16px">关联网络</router-link>
      <router-link to="/key-stations">关键站点</router-link>
    </nav>

    <main class="main">
      <router-view />
    </main>
  </div>
</template>

<script setup>
import { onMounted } from 'vue'
import { useFilterStore } from './stores/filter'

const filter = useFilterStore()
onMounted(() => { filter.loadStations() })
</script>
