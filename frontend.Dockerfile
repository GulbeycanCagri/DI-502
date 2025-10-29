# ----- Stage 1: Build -----
FROM node:20-alpine AS builder

# Set working directory inside the container
WORKDIR /app

# Copy package files from frontend (since Dockerfile is outside it)
COPY frontend/package.json frontend/package-lock.json ./ 

# Install dependencies
RUN npm ci --no-audit --no-fund

# Copy the rest of the frontend source
COPY frontend ./

# Build the frontend (Vite creates /app/dist)
RUN npm run build

# ----- Stage 2: Serve with Nginx -----
FROM nginx:stable-alpine

# Copy build output from previous stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy custom nginx config (at project root)
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80 and start nginx
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
