import type { NextConfig } from "next";

const TARGET_SERVER_BASE_URL = process.env.SERVER_BASE_URL || 'http://localhost:8001';

const nextConfig: NextConfig = {
  /* config options here */
  output: 'standalone',
  // Optimize build for Docker
  experimental: {
    optimizePackageImports: ['@mermaid-js/mermaid', 'react-syntax-highlighter'],
  },
  // Reduce memory usage during build
  webpack: (config, { isServer }) => {
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
      };
    }
    // Optimize bundle size
    config.optimization = {
      ...config.optimization,
      splitChunks: {
        chunks: 'all',
        cacheGroups: {
          vendor: {
            test: /[\\/]node_modules[\\/]/,
            name: 'vendors',
            chunks: 'all',
          },
        },
      },
    };
    return config;
  },
  async rewrites() {
    return [
      {
        source: '/api/wiki_cache/:path*',
        destination: `${TARGET_SERVER_BASE_URL}/api/wiki_cache/:path*`,
      },
      {
        source: '/export/wiki/:path*',
        destination: `${TARGET_SERVER_BASE_URL}/export/wiki/:path*`,
      },
      {
        source: '/api/wiki_cache',
        destination: `${TARGET_SERVER_BASE_URL}/api/wiki_cache`,
      },
      {
        source: '/local_repo/structure',
        destination: `${TARGET_SERVER_BASE_URL}/local_repo/structure`,
      },
      {
        source: '/api/auth/status',
        destination: `${TARGET_SERVER_BASE_URL}/auth/status`,
      },
      {
        source: '/api/auth/validate',
        destination: `${TARGET_SERVER_BASE_URL}/auth/validate`,
      },
      {
        source: '/api/lang/config',
        destination: `${TARGET_SERVER_BASE_URL}/lang/config`,
      },
      {
        source: '/api/models/config',
        destination: `${TARGET_SERVER_BASE_URL}/models/config`,
      },
      {
        source: '/api/models/runtime_config',
        destination: `${TARGET_SERVER_BASE_URL}/models/runtime_config`,
      },
      {
        source: '/api/models/provider_url',
        destination: `${TARGET_SERVER_BASE_URL}/models/provider_url`,
      },
      {
        source: '/api/models/add_model',
        destination: `${TARGET_SERVER_BASE_URL}/models/add_model`,
      },
      {
        source: '/api/upload/repo',
        destination: `${TARGET_SERVER_BASE_URL}/upload/repo`,
      },
      {
        source: '/api/processed_projects',
        destination: `${TARGET_SERVER_BASE_URL}/api/processed_projects`,
      },
      {
        source: '/chat/completions/stream',
        destination: `${TARGET_SERVER_BASE_URL}/chat/completions/stream`,
      },
      {
        source: '/repo/structure',
        destination: `${TARGET_SERVER_BASE_URL}/repo/structure`,
      },
      {
        source: '/chat/direct/stream',
        destination: `${TARGET_SERVER_BASE_URL}/chat/direct/stream`,
      },
      {
        source: '/repo/files',
        destination: `${TARGET_SERVER_BASE_URL}/repo/files`,
      },
      {
        source: '/api/schedules/:path*',
        destination: `${TARGET_SERVER_BASE_URL}/api/schedules/:path*`,
      },
      {
        source: '/api/schedules',
        destination: `${TARGET_SERVER_BASE_URL}/api/schedules`,
      },
    ];
  },
};

export default nextConfig;
