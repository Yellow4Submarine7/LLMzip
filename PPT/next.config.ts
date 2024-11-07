import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  images: {
    domains: ['your-image-domain.com'], // 如果有外部图片源
  },
  // 其他配置...
};

export default nextConfig;
