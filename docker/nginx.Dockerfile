FROM nginx:alpine

# Copy nginx configuration
COPY docker/nginx/nginx.conf /etc/nginx/nginx.conf

EXPOSE 80 443

CMD ["nginx", "-g", "daemon off;"]