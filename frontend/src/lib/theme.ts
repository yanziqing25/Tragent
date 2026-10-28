import { createSharedComposable, useDark, useToggle } from '@vueuse/core'

/** One app-wide light/dark switch; index.html applies the saved choice before first paint. */
export const useTheme = createSharedComposable(() => {
  const isDark = useDark({ storageKey: 'tragent.theme' })
  return { isDark, toggle: useToggle(isDark) }
})
