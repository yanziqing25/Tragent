export { default as Badge } from './Badge.vue'

const BASE =
  'inline-flex w-fit shrink-0 items-center gap-1.5 rounded-full font-medium whitespace-nowrap'

const VARIANT = {
  neutral: 'bg-muted text-muted-foreground',
  success: 'bg-success-soft text-success-ink',
  info: 'bg-info-soft text-info-ink',
  warning: 'bg-warning-soft text-warning-ink',
  danger: 'bg-destructive-soft text-destructive-ink',
  solid: 'rounded-sm bg-brand text-white dark:text-zinc-950',
} as const

const SIZE = {
  default: 'h-6 px-2.5 text-xs',
  sm: 'h-5 px-2 text-xs',
} as const

export const badgeVariants = (props?: BadgeVariants): string =>
  `${BASE} ${VARIANT[props?.variant ?? 'neutral']} ${SIZE[props?.size ?? 'default']}`

export type BadgeVariants = {
  variant?: keyof typeof VARIANT | null
  size?: keyof typeof SIZE | null
}
