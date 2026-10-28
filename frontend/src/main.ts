import { createApp } from 'vue'
import { PiniaColada } from '@pinia/colada'
import '@fontsource-variable/geist'
import '@fontsource-variable/geist-mono'

import App from './App.vue'
import router from './router'
import { pinia } from './stores'
import './styles/main.css'

createApp(App).use(pinia).use(PiniaColada).use(router).mount('#app')
