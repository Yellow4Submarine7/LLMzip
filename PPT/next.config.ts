import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'standalone',
  images: {
    unoptimized: true,
    domains: ['your-image-domain.com']
  }
};

export default nextConfig;
