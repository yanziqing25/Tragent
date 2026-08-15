import { createRouter, createWebHistory } from 'vue-router'
import type { RouteLocationNormalized } from 'vue-router'

import { pinia } from '@/stores'
import { useAuthStore } from '@/stores/auth'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    { path: '/', redirect: '/research' },
    {
      path: '/login',
      name: 'login',
      component: () => import('@/views/LoginView.vue'),
    },
    {
      path: '/research',
      name: 'research',
      component: () => import('@/views/ResearchListView.vue'),
      meta: { requiresAuth: true },
    },
    {
      path: '/doctor',
      name: 'doctor',
      component: () => import('@/views/DoctorView.vue'),
      meta: { requiresAuth: true },
    },
    {
      path: '/research/:threadId',
      name: 'research-detail',
      component: () => import('@/views/ResearchDetailView.vue'),
      meta: { requiresAuth: true },
    },
    {
      path: '/research/:threadId/report',
      name: 'research-report',
      component: () => import('@/views/ReportView.vue'),
      meta: { requiresAuth: true },
    },
  ],
})

export const authGuard = (to: Pick<RouteLocationNormalized, 'meta' | 'name'>) => {
  const auth = useAuthStore(pinia)
  if (to.meta.requiresAuth && !auth.isAuthenticated) return { name: 'login' }
  if (to.name === 'login' && auth.isAuthenticated) return { name: 'research' }
  return true
}

router.beforeEach(authGuard)

export default router
