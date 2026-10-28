import { twMerge } from 'tailwind-merge'
import { type HTMLAttributes, normalizeClass } from 'vue'

export const cn = (...inputs: HTMLAttributes['class'][]) => {
  return twMerge(normalizeClass(inputs))
}
