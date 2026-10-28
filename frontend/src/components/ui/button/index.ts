export { default as Button } from './Button.vue'

const BASE =
  'inline-flex shrink-0 cursor-pointer items-center justify-center gap-2 rounded-lg font-medium whitespace-nowrap transition-[background-color,border-color,color,transform] duration-150 active:translate-y-px disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0'

const VARIANT = {
  default: 'bg-primary text-primary-foreground hover:bg-primary-hover',
  outline: 'border border-border bg-background text-foreground hover:bg-muted',
  ghost: 'text-muted-foreground hover:bg-muted hover:text-foreground',
  danger: 'text-destructive hover:bg-destructive-soft',
} as const

const SIZE = {
  default: 'h-9 px-4',
  sm: 'h-8 px-3 text-[13px]',
  lg: 'h-10 px-5',
  icon: 'size-9',
  'icon-sm': 'size-7 [&_svg]:size-3.5',
} as const

export const buttonVariants = (props?: ButtonVariants): string =>
  `${BASE} ${VARIANT[props?.variant ?? 'default']} ${SIZE[props?.size ?? 'default']}`

export type ButtonVariants = {
  variant?: keyof typeof VARIANT | null
  size?: keyof typeof SIZE | null
}
