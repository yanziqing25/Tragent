# Build the Vue app, then serve it from Nginx, which also fronts the API and the object store.
FROM node:lts-slim AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
# Leave empty to call the API on the app's own origin, as behind this image's Nginx.
ARG VITE_API_BASE_URL=""
RUN npm run build

FROM nginx:alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
