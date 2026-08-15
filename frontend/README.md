# Tragent Web

Vue 3 + TypeScript + Pinia + Vuetify 4 frontend for the Tragent API.
JWT is tab-scoped through Pinia and `sessionStorage`. Reports are rendered from `report.json`.

All commands run in WSL:

```bash
source /home/yanziqing25/.nvm/nvm.sh
npm install
npm run dev
```

Vite proxies `/api` to `http://127.0.0.1:8000`. With FastAPI running, regenerate only the
OpenAPI types—no SDK is generated:

```bash
npm run generate:api-types
```

Checks:

```bash
npm run type-check
npm run test:unit -- --run
npm run build
```
