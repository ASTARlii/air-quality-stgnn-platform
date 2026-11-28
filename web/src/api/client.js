import axios from 'axios'

const http = axios.create({ baseURL: '/api/v1', timeout: 60000 })

export const getStations = () => http.get('/stations').then(r => r.data)
export const getNetworks = (params) => http.get('/networks', { params }).then(r => r.data)
export const getTrace = (params) => http.get('/trace', { params }).then(r => r.data)
export const getKeyStations = (params) => http.get('/key-stations', { params }).then(r => r.data)
export const getGrading = (params) => http.get('/grading', { params }).then(r => r.data)
export const postPrediction = (payload, with_grading = true) =>
  http.post(`/predictions?with_grading=${with_grading}`, payload).then(r => r.data)
export const recomputeNetwork = (payload) =>
  http.post('/networks/recompute', payload).then(r => r.data)
export const getTask = (key) => http.get(`/tasks/${key}`).then(r => r.data)

/**
 * 确保关联网络已构建。
 * GET /networks 未命中缓存时返回 202 + task_key，此处轮询任务直到 success 后重新拉取。
 */
export async function ensureNetwork(params, onStatus) {
  for (let attempt = 0; attempt < 3; attempt++) {
    const r = await http.get('/networks', { params })
    if (r.status === 200 && r.data.nodes) return r.data

    const key = r.data.task_key
    onStatus?.('pending')
    let lost = false
    for (let i = 0; i < 600; i++) {
      await new Promise(res => setTimeout(res, 1000))
      let t
      try {
        t = await getTask(key)
      } catch {
        lost = true          // 任务不存在（如后端重启），重新提交
        break
      }
      onStatus?.(t.status)
      if (t.status === 'success') break
      if (t.status === 'failed') throw new Error(t.error || '网络构建失败')
    }
    if (lost) continue
  }
  throw new Error('网络构建超时（未在 10 分钟内完成）')
}
