import 'vuetify/styles'

import { createVuetify } from 'vuetify'

export default createVuetify({
  theme: {
    defaultTheme: 'system',
    themes: {
      light: {
        dark: false,
        colors: {
          background: '#F4F7FB',
          surface: '#FFFFFF',
          'surface-variant': '#E9EEF7',
          'on-surface-variant': '#536078',
          primary: '#3157D5',
          secondary: '#0C8F86',
          success: '#16835A',
          info: '#3676DB',
          warning: '#B96E12',
          error: '#C44343',
        },
      },
      dark: {
        dark: true,
        colors: {
          background: '#0D111B',
          surface: '#151B27',
          'surface-variant': '#202A3A',
          'on-surface-variant': '#AFBBD0',
          primary: '#8AA4FF',
          secondary: '#58C9BC',
          success: '#5AC99B',
          info: '#81A8FF',
          warning: '#EDAE63',
          error: '#FF8A8A',
        },
      },
    },
  },
  defaults: {
    VCard: { rounded: 'xl', elevation: 0 },
    VBtn: { rounded: 'lg' },
    VTextField: { variant: 'outlined' },
    VTextarea: { variant: 'outlined' },
    VChip: { rounded: 'lg' },
  },
})
