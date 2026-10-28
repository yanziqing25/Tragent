export { default as Alert } from './Alert.vue'
export { default as AlertDescription } from './AlertDescription.vue'

const BASE =
  'relative flex w-full items-start gap-3 rounded-xl border px-4 py-3 text-sm [&>svg]:size-4 [&>svg]:shrink-0 [&>svg]:translate-y-0.5'

const VARIANT = {
  default: 'border-border bg-surface text-foreground',
  info: 'border-transparent bg-info-soft text-info-ink',
  warning: 'border-transparent bg-warning-soft text-warning-ink',
  destructive: 'border-transparent bg-destructive-soft text-destructive-ink',
} as const

export const alertVariants = (props?: AlertVariants): string =>
  `${BASE} ${VARIANT[props?.variant ?? 'default']}`

export type AlertVariants = {
  variant?: keyof typeof VARIANT | null
}
