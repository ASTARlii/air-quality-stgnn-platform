import { defineStore } from 'pinia'
import { getStations } from '../api/client'

// 前端默认的网络计算窗口（与后端缓存键一致）。
// 约 2 个月 ≈ 1460h，兼顾转移熵估计稳定性与构建耗时。
export const NETWORK_WINDOW = {
  from: '2024-10-01',
  to: '2024-11-30',
  surrogates: 100,
}

export const useFilterStore = defineStore('filter', {
  state: () => ({
    stationId: '1001A',
    pollutant: 'pm25',
    horizon: 24,
    stations: [],
  }),
  actions: {
    async loadStations() {
      const list = await getStations()
      this.stations = list.map(s => ({
        code: s.code, name: s.name, lon: s.lon, lat: s.lat, isReference: s.is_reference,
      }))
    },
  },
})
