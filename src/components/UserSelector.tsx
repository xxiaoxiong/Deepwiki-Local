'use client';

import React, { useState, useEffect } from 'react';
import { useLanguage } from '@/contexts/LanguageContext';

// Define the interfaces for our model configuration
interface Model {
  id: string;
  name: string;
}

interface Provider {
  id: string;
  name: string;
  models: Model[];
  supportsCustomModel?: boolean;
}

interface ModelConfig {
  providers: Provider[];
  defaultProvider: string;
}

interface ModelSelectorProps {
  provider: string;
  setProvider: (value: string) => void;
  model: string;
  setModel: (value: string) => void;
  isCustomModel: boolean;
  setIsCustomModel: (value: boolean) => void;
  customModel: string;
  setCustomModel: (value: string) => void;

  // Optional API key
  apiKey?: string;
  setApiKey?: (value: string) => void;

  // File filter configuration
  showFileFilters?: boolean;
  excludedDirs?: string;
  setExcludedDirs?: (value: string) => void;
  excludedFiles?: string;
  setExcludedFiles?: (value: string) => void;
  includedDirs?: string;
  setIncludedDirs?: (value: string) => void;
  includedFiles?: string;
  setIncludedFiles?: (value: string) => void;
}

export default function UserSelector({
  provider,
  setProvider,
  model,
  setModel,
  isCustomModel,
  setIsCustomModel,
  customModel,
  setCustomModel,

  // Optional API key
  apiKey = '',
  setApiKey,

  // File filter configuration
  showFileFilters = false,
  excludedDirs = '',
  setExcludedDirs,
  excludedFiles = '',
  setExcludedFiles,
  includedDirs = '',
  setIncludedDirs,
  includedFiles = '',
  setIncludedFiles
}: ModelSelectorProps) {
  // State to manage the visibility of the filters modal and filter section
  const [isFilterSectionOpen, setIsFilterSectionOpen] = useState(false);
  // State to manage filter mode: 'exclude' or 'include'
  const [filterMode, setFilterMode] = useState<'exclude' | 'include'>('exclude');
  const { messages: t } = useLanguage();

  // State for model configurations from backend
  const [modelConfig, setModelConfig] = useState<ModelConfig | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // State for viewing default values
  const [showDefaultDirs, setShowDefaultDirs] = useState(false);
  const [showDefaultFiles, setShowDefaultFiles] = useState(false);

  // State for provider URL editing
  const [providerUrls, setProviderUrls] = useState<Record<string, string>>({});
  const [urlSaving, setUrlSaving] = useState(false);
  const [urlSaveMsg, setUrlSaveMsg] = useState<string | null>(null);

  // Fetch model configurations from the backend
  useEffect(() => {
    const fetchModelConfig = async () => {
      try {
        setIsLoading(true);
        setError(null);

        const response = await fetch('/api/models/config');

        if (!response.ok) {
          throw new Error(`Error fetching model configurations: ${response.status}`);
        }

        const data = await response.json();
        setModelConfig(data);

        // Initialize provider and model with defaults from API if not already set
        if (!provider && data.defaultProvider) {
          setProvider(data.defaultProvider);

          // Find the default provider and set its default model
          const selectedProvider = data.providers.find((p: Provider) => p.id === data.defaultProvider);
          if (selectedProvider && selectedProvider.models.length > 0) {
            setModel(selectedProvider.models[0].id);
          }
        }
      } catch (err) {
        console.error('Failed to fetch model configurations:', err);
        setError('Failed to load model configurations. Using default options.');
      } finally {
        setIsLoading(false);
      }
    };

    fetchModelConfig();
  }, [provider, setModel, setProvider]);

  // Fetch runtime config (provider URLs)
  useEffect(() => {
    const fetchRuntimeConfig = async () => {
      try {
        const response = await fetch('/api/models/runtime_config');
        if (response.ok) {
          const data = await response.json();
          setProviderUrls(data.provider_urls || {});
        }
      } catch (err) {
        console.error('Failed to fetch runtime config:', err);
      }
    };
    fetchRuntimeConfig();
  }, []);

  // Save provider URL
  const handleSaveProviderUrl = async (providerId: string, url: string) => {
    setUrlSaving(true);
    setUrlSaveMsg(null);
    try {
      const response = await fetch('/api/models/provider_url', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider: providerId, base_url: url }),
      });
      if (response.ok) {
        setUrlSaveMsg('URL saved successfully');
        setTimeout(() => setUrlSaveMsg(null), 2000);
      } else {
        setUrlSaveMsg('Failed to save URL');
      }
    } catch (err) {
      void err;
      setUrlSaveMsg('Error saving URL');
    } finally {
      setUrlSaving(false);
    }
  };

  // Add custom model to provider
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const handleAddModel = async (providerId: string, modelId: string) => {
    if (!modelId.trim()) return;
    try {
      const response = await fetch('/api/models/add_model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider: providerId, model_id: modelId.trim() }),
      });
      if (response.ok) {
        // Refresh model config
        const configResponse = await fetch('/api/models/config');
        if (configResponse.ok) {
          const data = await configResponse.json();
          setModelConfig(data);
        }
        setModel(modelId.trim());
      }
    } catch (err) {
      console.error('Failed to add model:', err);
    }
  };

  // Handler for changing provider
  const handleProviderChange = (newProvider: string) => {
    setProvider(newProvider);
    setTimeout(() => {
      // Reset custom model state when changing providers
      setIsCustomModel(false);

      // Set default model for the selected provider
      if (modelConfig) {
        const selectedProvider = modelConfig.providers.find((p: Provider) => p.id === newProvider);
        if (selectedProvider && selectedProvider.models.length > 0) {
          setModel(selectedProvider.models[0].id);
        }
      }
    }, 10);
  };

  // Default excluded directories from config.py
  const defaultExcludedDirs =
`./.venv/
./venv/
./env/
./virtualenv/
./node_modules/
./bower_components/
./jspm_packages/
./.git/
./.svn/
./.hg/
./.bzr/
./__pycache__/
./.pytest_cache/
./.mypy_cache/
./.ruff_cache/
./.coverage/
./dist/
./build/
./out/
./target/
./bin/
./obj/
./docs/
./_docs/
./site-docs/
./_site/
./.idea/
./.vscode/
./.vs/
./.eclipse/
./.settings/
./logs/
./log/
./tmp/
./temp/
./.eng`;

  // Default excluded files from config.py
  const defaultExcludedFiles =
`package-lock.json
yarn.lock
pnpm-lock.yaml
npm-shrinkwrap.json
poetry.lock
Pipfile.lock
requirements.txt.lock
Cargo.lock
composer.lock
.lock
.DS_Store
Thumbs.db
desktop.ini
*.lnk
.env
.env.*
*.env
*.cfg
*.ini
.flaskenv
.gitignore
.gitattributes
.gitmodules
.github
.gitlab-ci.yml
.prettierrc
.eslintrc
.eslintignore
.stylelintrc
.editorconfig
.jshintrc
.pylintrc
.flake8
mypy.ini
pyproject.toml
tsconfig.json
webpack.config.js
babel.config.js
rollup.config.js
jest.config.js
karma.conf.js
vite.config.js
next.config.js
*.min.js
*.min.css
*.bundle.js
*.bundle.css
*.map
*.gz
*.zip
*.tar
*.tgz
*.rar
*.pyc
*.pyo
*.pyd
*.so
*.dll
*.class
*.exe
*.o
*.a
*.jpg
*.jpeg
*.png
*.gif
*.ico
*.svg
*.webp
*.mp3
*.mp4
*.wav
*.avi
*.mov
*.webm
*.csv
*.tsv
*.xls
*.xlsx
*.db
*.sqlite
*.sqlite3
*.pdf
*.docx
*.pptx`;

  // Display loading state
  if (isLoading) {
    return (
      <div className="flex flex-col gap-2">
        <div className="text-sm text-[var(--muted)]">Loading model configurations...</div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="space-y-4">
        {error && (
          <div className="text-sm text-red-500 mb-2">{error}</div>
        )}

        {/* Provider Selection */}
        <div>
          <label htmlFor="provider-dropdown" className="block text-xs font-medium text-[var(--foreground)] mb-1.5">
            {t.form?.modelProvider || 'Model Provider'}
          </label>
          <select
            id="provider-dropdown"
            value={provider}
            onChange={(e) => handleProviderChange(e.target.value)}
            className="input-japanese block w-full px-2.5 py-1.5 text-sm rounded-md bg-transparent text-[var(--foreground)] focus:outline-none focus:border-[var(--accent-primary)]"
          >
            <option value="" disabled>{t.form?.selectProvider || 'Select Provider'}</option>
            {modelConfig?.providers.map((providerOption) => (
              <option key={providerOption.id} value={providerOption.id}>
                {t.form?.[`provider${providerOption.id.charAt(0).toUpperCase() + providerOption.id.slice(1)}`] || providerOption.name}
              </option>
            ))}
          </select>
        </div>

        {/* Provider URL Configuration */}
        {provider && (
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="block text-xs font-medium text-[var(--foreground)]">
                {'API Base URL'}
              </label>
            </div>
            <div className="flex gap-1.5">
              <input
                type="text"
                value={providerUrls[provider] || ''}
                onChange={(e) => setProviderUrls(prev => ({ ...prev, [provider]: e.target.value }))}
                placeholder="e.g. http://10.0.0.1:8000/v1"
                className="input-japanese block w-full px-2.5 py-1.5 text-sm rounded-md bg-transparent text-[var(--foreground)] focus:outline-none focus:border-[var(--accent-primary)]"
              />
              <button
                type="button"
                onClick={() => handleSaveProviderUrl(provider, providerUrls[provider] || '')}
                disabled={urlSaving}
                className="px-3 py-1.5 text-xs rounded-md bg-[var(--accent-primary)] text-white hover:bg-[var(--accent-primary)]/80 disabled:opacity-50 whitespace-nowrap"
              >
                {urlSaving ? '...' : 'Save'}
              </button>
            </div>
            {urlSaveMsg && (
              <div className="text-xs text-green-500 mt-1">{urlSaveMsg}</div>
            )}
          </div>
        )}

        {/* Model Selection - always free text input */}
        <div>
          <label htmlFor="model-input" className="block text-xs font-medium text-[var(--foreground)] mb-1.5">
            {t.form?.modelSelection || 'Model Selection'}
          </label>
          <input
            id="model-input"
            type="text"
            value={customModel || model}
            onChange={(e) => {
              setModel(e.target.value);
              setCustomModel(e.target.value);
              if (!isCustomModel) setIsCustomModel(true);
            }}
            placeholder={t.form?.customModelPlaceholder || 'Enter model name'}
            className="input-japanese block w-full px-2.5 py-1.5 text-sm rounded-md bg-transparent text-[var(--foreground)] focus:outline-none focus:border-[var(--accent-primary)]"
          />
        </div>

        {/* API Key Input (optional) */}
        {setApiKey && (
          <div>
            <label htmlFor="api-key-input" className="block text-xs font-medium text-[var(--foreground)] mb-1.5">
              {t.form?.apiKey || 'API Key'}
              <span className="ml-1 text-[var(--muted)] font-normal">({t.form?.optional || 'optional'})</span>
            </label>
            <input
              id="api-key-input"
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder={t.form?.apiKeyPlaceholder || 'Enter API key for this provider'}
              className="input-japanese block w-full px-2.5 py-1.5 text-sm rounded-md bg-transparent text-[var(--foreground)] focus:outline-none focus:border-[var(--accent-primary)]"
            />
            <p className="text-xs text-[var(--muted)] mt-1">
              {t.form?.apiKeyHint || 'If set, overrides the server-side API key for this provider.'}
            </p>
          </div>
        )}


        {showFileFilters && (
          <div className="mt-4">
            <button
              type="button"
              onClick={() => setIsFilterSectionOpen(!isFilterSectionOpen)}
              className="flex items-center text-sm text-[var(--accent-primary)] hover:text-[var(--accent-primary)]/80 transition-colors"
            >
              <span className="mr-1.5 text-xs">{isFilterSectionOpen ? '▼' : '►'}</span>
              {t.form?.advancedOptions || 'Advanced Options'}
            </button>

            {isFilterSectionOpen && (
              <div className="mt-3 p-3 border border-[var(--border-color)]/70 rounded-md bg-[var(--background)]/30">
                {/* Filter Mode Selection */}
                <div className="mb-4">
                  <label className="block text-sm font-medium text-[var(--foreground)] mb-2">
                    {t.form?.filterMode || 'Filter Mode'}
                  </label>
                  <div className="flex gap-2">
                    <button
                      type="button"
                      onClick={() => setFilterMode('exclude')}
                      className={`flex-1 px-3 py-2 rounded-md border text-sm transition-colors ${
                        filterMode === 'exclude'
                          ? 'bg-[var(--accent-primary)]/10 border-[var(--accent-primary)] text-[var(--accent-primary)]'
                          : 'border-[var(--border-color)] text-[var(--foreground)] hover:bg-[var(--background)]'
                      }`}
                    >
                      {t.form?.excludeMode || 'Exclude Paths'}
                    </button>
                    <button
                      type="button"
                      onClick={() => setFilterMode('include')}
                      className={`flex-1 px-3 py-2 rounded-md border text-sm transition-colors ${
                        filterMode === 'include'
                          ? 'bg-[var(--accent-primary)]/10 border-[var(--accent-primary)] text-[var(--accent-primary)]'
                          : 'border-[var(--border-color)] text-[var(--foreground)] hover:bg-[var(--background)]'
                      }`}
                    >
                      {t.form?.includeMode || 'Include Only Paths'}
                    </button>
                  </div>
                  <p className="text-xs text-[var(--muted)] mt-1">
                    {filterMode === 'exclude'
                      ? (t.form?.excludeModeDescription || 'Specify paths to exclude from processing (default behavior)')
                      : (t.form?.includeModeDescription || 'Specify only the paths to include, ignoring all others')
                    }
                  </p>
                </div>

                {/* Directories Section */}
                <div className="mb-4">
                  <label className="block text-sm font-medium text-[var(--muted)] mb-1.5">
                    {filterMode === 'exclude'
                      ? (t.form?.excludedDirs || 'Excluded Directories')
                      : (t.form?.includedDirs || 'Included Directories')
                    }
                  </label>
                  <textarea
                    value={filterMode === 'exclude' ? excludedDirs : includedDirs}
                    onChange={(e) => {
                      if (filterMode === 'exclude') {
                        setExcludedDirs?.(e.target.value);
                      } else {
                        setIncludedDirs?.(e.target.value);
                      }
                    }}
                    rows={4}
                    className="block w-full rounded-md border border-[var(--border-color)]/50 bg-[var(--input-bg)] text-[var(--foreground)] px-3 py-2 text-sm focus:border-[var(--accent-primary)] focus:ring-1 focus:ring-opacity-50 shadow-sm"
                    placeholder={filterMode === 'exclude'
                      ? (t.form?.enterExcludedDirs || 'Enter excluded directories, one per line...')
                      : (t.form?.enterIncludedDirs || 'Enter included directories, one per line...')
                    }
                  />
                  {filterMode === 'exclude' && (
                    <>
                      <div className="flex mt-1.5">
                        <button
                          type="button"
                          onClick={() => setShowDefaultDirs(!showDefaultDirs)}
                          className="text-xs text-[var(--accent-primary)] hover:text-[var(--accent-primary)]/80 transition-colors"
                        >
                          {showDefaultDirs ? (t.form?.hideDefault || 'Hide Default') : (t.form?.viewDefault || 'View Default')}
                        </button>
                      </div>
                      {showDefaultDirs && (
                        <div className="mt-2 p-2 rounded bg-[var(--background)]/50 text-xs">
                          <p className="mb-1 text-[var(--muted)]">{t.form?.defaultNote || 'These defaults are already applied. Add your custom exclusions above.'}</p>
                          <pre className="whitespace-pre-wrap font-mono text-[var(--muted)] overflow-y-auto max-h-32">{defaultExcludedDirs}</pre>
                        </div>
                      )}
                    </>
                  )}
                </div>

                {/* Files Section */}
                <div>
                  <label className="block text-sm font-medium text-[var(--muted)] mb-1.5">
                    {filterMode === 'exclude'
                      ? (t.form?.excludedFiles || 'Excluded Files')
                      : (t.form?.includedFiles || 'Included Files')
                    }
                  </label>
                  <textarea
                    value={filterMode === 'exclude' ? excludedFiles : includedFiles}
                    onChange={(e) => {
                      if (filterMode === 'exclude') {
                        setExcludedFiles?.(e.target.value);
                      } else {
                        setIncludedFiles?.(e.target.value);
                      }
                    }}
                    rows={4}
                    className="block w-full rounded-md border border-[var(--border-color)]/50 bg-[var(--input-bg)] text-[var(--foreground)] px-3 py-2 text-sm focus:border-[var(--accent-primary)] focus:ring-1 focus:ring-opacity-50 shadow-sm"
                    placeholder={filterMode === 'exclude'
                      ? (t.form?.enterExcludedFiles || 'Enter excluded files, one per line...')
                      : (t.form?.enterIncludedFiles || 'Enter included files, one per line...')
                    }
                  />
                  {filterMode === 'exclude' && (
                    <>
                      <div className="flex mt-1.5">
                        <button
                          type="button"
                          onClick={() => setShowDefaultFiles(!showDefaultFiles)}
                          className="text-xs text-[var(--accent-primary)] hover:text-[var(--accent-primary)]/80 transition-colors"
                        >
                          {showDefaultFiles ? (t.form?.hideDefault || 'Hide Default') : (t.form?.viewDefault || 'View Default')}
                        </button>
                      </div>
                      {showDefaultFiles && (
                        <div className="mt-2 p-2 rounded bg-[var(--background)]/50 text-xs">
                          <p className="mb-1 text-[var(--muted)]">{t.form?.defaultNote || 'These defaults are already applied. Add your custom exclusions above.'}</p>
                          <pre className="whitespace-pre-wrap font-mono text-[var(--muted)] overflow-y-auto max-h-32">{defaultExcludedFiles}</pre>
                        </div>
                      )}
                    </>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
