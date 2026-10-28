import { createRouter, createWebHistory, type RouteLocationNormalized } from 'vue-router'

import { pinia } from '@/stores'
import { useAuthStore } from '@/stores/auth'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    { path: '/', redirect: '/account' },
    { path: '/login', name: 'login', component: () => import('@/views/LoginView.vue') },
    {
      path: '/account',
      name: 'account',
      component: () => import('@/views/BrokerAccountView.vue'),
      meta: { requiresAuth: true },
    },
    {
      path: '/strategies',
      name: 'strategies',
      component: () => import('@/views/StrategiesView.vue'),
      meta: { requiresAuth: true },
    },
    {
      path: '/research',
      name: 'research',
      component: () => import('@/views/ResearchListView.vue'),
      meta: { requiresAuth: true },
    },
    {
      path: '/research/:threadId',
      name: 'task',
      component: () => import('@/views/TaskView.vue'),
      meta: { requiresAuth: true },
    },
    {
      path: '/research/:threadId/report',
      name: 'report',
      component: () => import('@/views/ReportView.vue'),
      meta: { requiresAuth: true },
    },
    {
      path: '/diagnostics',
      name: 'diagnostics',
      component: () => import('@/views/DiagnosticsView.vue'),
      meta: { requiresAuth: true },
    },
  ],
  scrollBehavior: () => ({ top: 0 }),
})

export const authGuard = (to: Pick<RouteLocationNormalized, 'meta' | 'name'>) => {
  const auth = useAuthStore(pinia)
  if (to.meta.requiresAuth && !auth.isAuthenticated) return { name: 'login' }
  if (to.name === 'login' && auth.isAuthenticated) return { name: 'account' }
  return true
}

router.beforeEach(authGuard)

export default router
