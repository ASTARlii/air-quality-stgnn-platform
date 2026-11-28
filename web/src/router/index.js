import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { path: '/', redirect: '/map' },
  { path: '/map', component: () => import('../views/MapView.vue'), name: 'map' },
  { path: '/prediction', component: () => import('../views/PredictionView.vue'), name: 'prediction' },
  { path: '/network', component: () => import('../views/NetworkView.vue'), name: 'network' },
  { path: '/key-stations', component: () => import('../views/KeyStationsView.vue'), name: 'key-stations' },
]

export default createRouter({
  history: createWebHistory(),
  routes,
})
